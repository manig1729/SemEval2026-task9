import os
import re
import json
import random
from typing import Optional, List, Dict, Any

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ============================================================
#                     USER CONFIGURATION
# ============================================================

INPUT_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_1/train_subtask_1.csv"
OUTPUT_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_1/llm_aug/train_aug_groups.csv"
LEXICON_PATH = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/lexicon_data/spurious_lexicon_manually_curated.txt"

ID_COLUMN = "id"
TEXT_COLUMN = "text"
LABEL_COLUMN = "polarization"                # 0/1 column

N_AUG_PER_EXAMPLE = 3                        # how many variants per example
MAX_EXAMPLES = 100                           # max polarized+matched rows to augment (for speed/cost); None = all
RANDOM_SEED = 42

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.9
TOP_P = 0.95

AUGMENTED_COLUMN = "is_augmented"
ORIGINAL_ID_COLUMN = "original_id_map"


# ============================================================
#                   GROUP/ISSUE REGEX LEXICON
# ============================================================

def load_lexicon(path: str) -> list[str]:
    """
    Load a newline-separated lexicon file.
    Ignores empty lines and lines starting with '#'.
    """
    terms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            term = line.strip()
            if not term or term.startswith("#"):
                continue
            terms.append(term)
    return terms


def build_regex_patterns(terms: list[str]) -> list[re.Pattern]:
    """
    Turn each lexicon term into a compiled regex pattern.
    We:
      - escape the term (so '.' etc are literal),
      - add word boundaries at edges when appropriate,
      - compile with IGNORECASE.
    Works for both single words and multi-word phrases like
    'black lives matter' or 'fox news'.
    """
    patterns = []
    for term in terms:
        escaped = re.escape(term)

        # Add word boundaries if the term begins/ends with a word char.
        pattern_str = escaped
        if re.match(r"\w", term[0]):
            pattern_str = r"\b" + pattern_str
        if re.match(r"\w", term[-1]):
            pattern_str = pattern_str + r"\b"

        patterns.append(re.compile(pattern_str, flags=re.IGNORECASE))
    return patterns


# Build patterns at import time
LEXICON_TERMS = load_lexicon(LEXICON_PATH)
compiled_patterns = build_regex_patterns(LEXICON_TERMS)


# ============================================================
#                   LLAMA MODEL LOADING
# ============================================================

def load_llama_model():
    """
    Load local Llama-3.1-8B-Instruct in 4-bit mode using env var CURC_LLM_DIR.
    """
    CURC_LLM_DIR = os.getenv("CURC_LLM_DIR")
    if CURC_LLM_DIR is None:
        raise ValueError("Environment variable CURC_LLM_DIR is not set.")
    path_to_model = f"{CURC_LLM_DIR}/hf-transformers/Llama-3.1-8B-Instruct"
    print("Path to model: ", path_to_model)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("Config established!")

    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    print("Tokenizer loaded!")

    model = AutoModelForCausalLM.from_pretrained(
        path_to_model,
        device_map="cuda",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    print("Model loaded!")

    return tokenizer, model


# ============================================================
#                     HELPER FUNCTIONS
# ============================================================

def parse_augmented_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Given raw model output, try to find and parse one or more JSON objects.
    Return the most likely one that contains `augmented_texts`.

    Strategy:
    - Strip code fences.
    - Find ALL `{ ... }` blocks (non-greedy).
    - For each block, try json.loads.
    - Keep the last one that has an "augmented_texts" list.
    """
    if not text:
        return None

    stripped = text.strip()

    # Remove Markdown-style fences, if present
    fence_match = re.search(r"```(?:json)?(.*)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        stripped = fence_match.group(1).strip()

    # Find ALL candidate JSON objects
    candidates = re.finditer(r"\{[\s\S]*?\}", stripped)
    best_data = None

    for match in candidates:
        candidate = match.group(0).strip()
        try:
            data = json.loads(candidate)
        except Exception:
            continue

        # We only care about objects that have "augmented_texts" as a list
        if isinstance(data, dict) and isinstance(data.get("augmented_texts"), list):
            best_data = data  # keep the last valid one

    return best_data


def find_group_mentions(text: str) -> List[str]:
    """
    Return a list of unique substrings that match the group/issue patterns.
    """
    mentions = set()
    for pat in compiled_patterns:
        for m in pat.finditer(text):
            mentions.add(m.group(0))
    return sorted(mentions)


POLARIZATION_DEFINITION = """
We are studying polarization in social media posts.

- A "polarized" post clearly expresses a strong, divisive attitude or opinion,
  usually framing an "us vs them" dynamic, strongly supporting one side and/or
  attacking another.
- Here, all original posts you will see are labeled as POLARIZED (1).
""".strip()


def build_polarized_prompt(text: str, mentions: List[str], n_aug: int) -> str:
    mentions_str = ", ".join(f'"{m}"' for m in mentions) if mentions else "none"

    prompt = f"""
        {POLARIZATION_DEFINITION}

        You will be given a social media post that is labeled as POLARIZED (1).
        It expresses a strong, divisive attitude.

        Original post:
        {text}

        Groups or entities mentioned in the original post:
        [{mentions_str}]

        Your task:
        - Generate {n_aug} new posts that:
          1. Remain clearly POLARIZED (strong, divisive opinion).
          2. Keep a similar *type* of stance (e.g., critical, supportive, mocking),
             but you must change the specific groups, parties, countries, or movements
             mentioned to DIFFERENT BUT SOCIALLY COMPARABLE ones.
             For example:
             - If the original attacks "Democrats", you might attack "Republicans" or "liberals".
             - If the original talks about one country, you may switch to another country
               in a similar geopolitical context.
          3. The intensity of polarization should remain similar (don't make it neutral).
          4. Do NOT copy the original sentences; produce genuinely rephrased posts following the same syntactical structure.
          5. Do NOT mention exactly the same groups/entities as in the original.
          6. Do NOT add explanations; only return JSON.

        Output format (valid JSON):
        {{
          "augmented_texts": [
            "first new polarized post here",
            "second new polarized post here",
            "third new polarized post here"
          ]
        }}
    """.strip()

    return prompt


def build_nonpolarized_prompt(text: str, mentions: List[str], n_aug: int) -> str:
    mentions_str = ", ".join(f'"{m}"' for m in mentions) if mentions else "none"

    prompt = f"""
        {POLARIZATION_DEFINITION}

        You will be given a social media post that is labeled as POLARIZED (1),
        meaning it expresses a strong, divisive attitude.

        Original post:
        {text}

        Groups or entities mentioned in the original post:
        [{mentions_str}]

        Your task:
        - Generate {n_aug} new posts that:
          1. Are clearly NON-POLARIZED (neutral, balanced, or mildly opinionated).
          2. Should NOT frame a strong "us vs them" conflict or use aggressive, hostile language.
          3. Stay roughly on the same topic as the original.
          4. Mention the groups found ([{mentions_str}]),
             but only in a descriptive, informational, or balanced way.
          5. Do NOT copy the original sentences; produce genuinely rephrased posts that may have syntactically different structure.
          6. Do NOT add explanations; only return JSON.

        Output format (valid JSON):
        {{
          "augmented_texts": [
            "first new non-polarized post here",
            "second new non-polarized post here",
            "third new non-polarized post here"
          ]
        }}
    """.strip()

    return prompt



def generate_with_llama(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> str:
    """
    Use Llama in chat style to generate a completion for the given prompt.
    Returns the decoded generated text (assistant part only).
    """
    # Use chat template if available (Llama 3.1 typically has one)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a careful data augmentation assistant. "
                                          "You strictly follow the requested JSON output format."},
            {"role": "user", "content": prompt},
        ]
        model_input = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        # Fallback: plain prompt
        model_input = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            model_input,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and strip the prompt part if using chat template, tokenizer usually handles that
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Heuristic: if using chat template, the model output includes the prompt; we want only the assistant part.
    # Often, the assistant content appears after the last special token or after the prompt text.
    # For simplicity, we'll just take the text after the prompt if we can find it.
    if hasattr(tokenizer, "apply_chat_template"):
        # Reconstruct the prompt text without generation and split.
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if decoded.startswith(prompt_text):
            decoded = decoded[len(prompt_text):].strip()

    return decoded.strip()


def call_llm_for_augmentations(
    tokenizer,
    model,
    text: str,
    mentions: List[str],
    n_aug: int,
    target_label: int,
) -> List[str]:
    """
    Build a label-specific prompt (polarized vs non-polarized), call Llama,
    parse JSON, and return a list of augmented texts.
    """

    if target_label == 1:
        prompt = build_polarized_prompt(text, mentions, n_aug)
    elif target_label == 0:
        prompt = build_nonpolarized_prompt(text, mentions, n_aug)
    else:
        raise ValueError(f"Unsupported target_label {target_label}; expected 0 or 1.")

    raw_output = generate_with_llama(
        tokenizer,
        model,
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    # ---- NEW: parse all candidates, keep the best ----
    data = parse_augmented_json(raw_output)
    if data is None:
        print("Could not find valid JSON with 'augmented_texts'. Raw LLM output:")
        print(raw_output[:1000])
        raise RuntimeError("JSON parsing failed for this LLM output.")


    try:
        aug_texts = data.get("augmented_texts", [])
        aug_texts = [
            t.strip()
            for t in aug_texts
            if isinstance(t, str) and t.strip()
        ]
        return aug_texts[:n_aug]
    except Exception as e:
        print("Post-processing parsed JSON failed. Error:", e)
        print("Parsed data:", data)
        raise RuntimeError("JSON post-processing failed.")






# ============================================================
#                             MAIN
# ============================================================

def main():
    random.seed(RANDOM_SEED)

    print(f"Loading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    for col in [ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input CSV.")

    df[AUGMENTED_COLUMN] = 0
    df[ORIGINAL_ID_COLUMN] = df[ID_COLUMN]

    # Keep only polarized examples as input
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    df_pol = df[df[LABEL_COLUMN] == 1].copy()
    print(f"Total polarized examples (input to LLM): {len(df_pol)}")

    # Find group mentions for each polarized example
    df_pol["group_mentions"] = df_pol[TEXT_COLUMN].apply(find_group_mentions)
    df_pol_matched = df_pol[df_pol["group_mentions"].map(len) > 0].copy()
    print(f"Polarized examples with group mentions: {len(df_pol_matched)}")

    # Optional subsampling
    if MAX_EXAMPLES is not None and MAX_EXAMPLES < len(df_pol_matched):
        df_pol_matched = df_pol_matched.sample(
            n=MAX_EXAMPLES,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)
        print(f"Subsampled polarized examples to {len(df_pol_matched)} for augmentation.")

    # Load Llama
    tokenizer, model = load_llama_model()

    augmented_rows: List[Dict[str, Any]] = []

    print("Starting LLM-based augmentation for polarized inputs...")
    for idx, row in df_pol_matched.iterrows():
        orig_id = row[ID_COLUMN]
        orig_text = str(row[TEXT_COLUMN])
        mentions = row["group_mentions"]

        print(
            f"[{idx+1}/{len(df_pol_matched)}] "
            f"id={orig_id}, label=1, mentions={mentions}"
        )

        # 1) Generate new POLARIZED posts (label 1)
        pol_texts = call_llm_for_augmentations(
            tokenizer,
            model,
            orig_text,
            mentions,
            N_AUG_PER_EXAMPLE,
            target_label=1,
        )

        for j, new_text in enumerate(pol_texts):
            new_id = f"{orig_id}_pol_aug{j+1}"
            augmented_rows.append(
                {
                    ID_COLUMN: new_id,
                    ORIGINAL_ID_COLUMN: orig_id,
                    TEXT_COLUMN: new_text,
                    LABEL_COLUMN: 1,  # still polarized
                    AUGMENTED_COLUMN: 1,
                }
            )

        # 2) Generate new NON-POLARIZED posts (label 0)
        non_texts = call_llm_for_augmentations(
            tokenizer,
            model,
            orig_text,
            mentions,
            N_AUG_PER_EXAMPLE,
            target_label=0,
        )

        for j, new_text in enumerate(non_texts):
            new_id = f"{orig_id}_non_aug{j+1}"
            augmented_rows.append(
                {
                    ID_COLUMN: new_id,
                    ORIGINAL_ID_COLUMN: orig_id,
                    TEXT_COLUMN: new_text,
                    LABEL_COLUMN: 0,  # de-escalated
                    AUGMENTED_COLUMN: 1,
                }
            )

    print(f"Generated {len(augmented_rows)} augmented rows total.")

    df_aug = pd.DataFrame(augmented_rows)

    # Combine original + augmented
    full_df = pd.concat([df, df_aug], ignore_index=True)

    # Optional dedup by text+label
    full_df = full_df.drop_duplicates(subset=[TEXT_COLUMN, LABEL_COLUMN])

    print(f"Final dataset size (original + augmented, deduped): {len(full_df)}")

    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Saving augmented dataset to: {OUTPUT_CSV}")
    full_df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")



if __name__ == "__main__":
    main()
