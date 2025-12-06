import os
import re
import json
import random
from typing import List, Dict, Any

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

INPUT_CSV = "/Users/tejo9855/Documents/Classes/Fall '25/NLP - Martin/Assignments/SemEval2026-task9/data/subtask1/train/eng.csv"                 # original labeled data
OUTPUT_CSV = "/Users/tejo9855/Documents/Classes/Fall '25/NLP - Martin/Assignments/SemEval2026-task9/teagan_folder/src/output/train_aug_groups.csv"     # output with augmented rows appended

ID_COLUMN = "id"
TEXT_COLUMN = "text"
LABEL_COLUMN = "polarization"                # 0/1 column

N_AUG_PER_EXAMPLE = 3                        # how many variants per example
MAX_EXAMPLES = 100                           # max polarized+matched rows to augment (for speed/cost); None = all
RANDOM_SEED = 42

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.9
TOP_P = 0.95

# ============================================================
#                   GROUP/ISSUE REGEX LEXICON
# ============================================================

patterns_to_match = [
    # Political figures
    r"\bdonald\s+trump\b|\btrump(s)?\b",
    r"\bjoe\s+biden\b|\bbiden\b",
    r"\bbarack\s+obama\b|\bobama\b",
    r"\bhillary\s+clinton\b|\bclinton\b",
    r"\bnancy\s+pelosi\b|\bpelosi\b",
    r"\bchuck\s+schumer\b|\bschumer\b",
    r"\bmitch\s+mcconnell\b|\bmcconnell\b",
    r"\bkamala\s+harris\b|\bkamala\b",
    r"\bmike\s+pence\b|\bpence\b",
    r"\bron\s+desantis\b|\bdesantis\b",

    # Political parties & ideologies
    r"\bdemocrat(s)?\b",
    r"\brepublican(s)?\b",
    r"\bliberal(s)?\b",
    r"\bconservative(s)?\b",
    r"\bprogressive(s)?\b",
    r"\bleftist(s)?\b",
    r"\bright-?wing\b",

    # Countries & regions
    r"\bisrael\b|\bpalestine\b|\bgaza\b|\biran\b|\bukraine\b|\brussia\b|\bchina\b|\btaiwan\b",

    # Hot-button issues
    r"\babortion\b",
    r"\bgun(s)?\b",
    r"\bimmigration\b|\bborder\b",
    r"\bclimate\b",
    r"\bcovid\b|\bvaccine(s)?\b|\bmask mandate\b",

    # Movements & slogans
    r"\bblm\b|\bblack\s+lives\s+matter\b",
    r"\bmaga\b",
    r"\bantifa\b",
    r"\bwoke\b",
    r"\bcancel\s+culture\b",
    r"\bme\s+too\b",

    # Media outlets
    r"\bcnn\b",
    r"\bfox\b|\bfox\s+news\b",
    r"\bmsnbc\b",
    r"\bbreitbart\b",
    r"\bnytimes\b|\bnew\s+york\s+times\b|\bnyt\b",
    r"\bwashington\s+post\b",
    r"\btwitter\b|\bx\b",
    r"\bfacebook\b",
]

# Precompile regex for efficiency
compiled_patterns = [re.compile(p, flags=re.IGNORECASE) for p in patterns_to_match]


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

def find_group_mentions(text: str) -> List[str]:
    """
    Return a list of unique substrings that match the group/issue patterns.
    """
    mentions = set()
    for pat in compiled_patterns:
        for m in pat.finditer(text):
            mentions.add(m.group(0))
    return sorted(mentions)


POLARIZATION_EXPLANATION = """
We are studying polarization in social media posts.

- A "polarized" post clearly expresses a strong, divisive attitude or opinion,
  usually framing an "us vs them" dynamic, strongly supporting one side and/or
  attacking another.
- Here, all posts you will see are labeled as POLARIZED (1).
""".strip()


def build_prompt(text: str, mentions: List[str], n_aug: int) -> str:
    """
    Build an instruction for the LLM to generate n_aug variants:
    - still polarized
    - similar stance/topic
    - but mentioning different (comparable) groups.
    """
    mentions_str = ", ".join(f'"{m}"' for m in mentions) if mentions else "none"

    prompt = f"""
        {POLARIZATION_EXPLANATION}

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
            but you must change the specific groups, parties, countries,
            or movements mentioned to DIFFERENT BUT SOCIALLY COMPARABLE ones.
            For example:
            - If the original attacks "Democrats", you might attack "Republicans" or "liberals".
            - If the original talks about one country, you may switch to another country
                in a similar geopolitical context.
        3. The intensity of polarization should remain similar (don't turn it neutral).
        4. Do NOT copy the original sentences; produce genuinely rephrased posts.
        5. Do NOT mention exactly the same groups/entities as in the original.
        6. Do NOT add explanations; only return JSON.

        Output format (valid JSON):
        {{
        "augmented_texts": [
            "first new polarized post here",
            "second new polarized post here",
            ...
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
) -> List[str]:
    """
    Build a prompt, call Llama, parse JSON, return list of augmented texts.
    On parse failure, return empty list.
    """
    prompt = build_prompt(text, mentions, n_aug)
    raw_output = generate_with_llama(
        tokenizer,
        model,
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    # Try to parse JSON
    try:
        data = json.loads(raw_output)
        aug_texts = data.get("augmented_texts", [])
        aug_texts = [
            t.strip() for t in aug_texts
            if isinstance(t, str) and t.strip()
        ]
        return aug_texts[:n_aug]
    except Exception as e:
        print("JSON parse failed. Error:", e)
        print("Raw LLM output:\n", raw_output[:1000])
        return []


# ============================================================
#                             MAIN
# ============================================================

def main():
    random.seed(RANDOM_SEED)

    # Load data
    print(f"Loading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    for col in [ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input CSV.")

    # Basic cleaning
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    # Filter to polarized = 1
    df_pol = df[df[LABEL_COLUMN] == 1].copy()
    print(f"Total polarized examples: {len(df_pol)}")

    # Further filter to those that actually mention at least one group pattern
    df_pol["group_mentions"] = df_pol[TEXT_COLUMN].apply(find_group_mentions)
    df_pol_matched = df_pol[df_pol["group_mentions"].map(len) > 0].copy()
    print(f"Polarized examples with group mentions: {len(df_pol_matched)}")

    # Optional subsampling
    if MAX_EXAMPLES is not None and MAX_EXAMPLES < len(df_pol_matched):
        df_pol_matched = df_pol_matched.sample(
            n=MAX_EXAMPLES,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)
        print(f"Subsampled to {len(df_pol_matched)} examples for augmentation.")

    # Load Llama
    tokenizer, model = load_llama_model()

    augmented_rows: List[Dict[str, Any]] = []

    print("Starting LLM-based augmentation for polarized group-mention examples...")
    for idx, row in df_pol_matched.iterrows():
        orig_id = row[ID_COLUMN]
        text = str(row[TEXT_COLUMN])
        label = int(row[LABEL_COLUMN])  # always 1 here
        mentions = row["group_mentions"]

        print(f"[{idx+1}/{len(df_pol_matched)}] id={orig_id}, label={label}, mentions={mentions}")

        new_texts = call_llm_for_augmentations(
            tokenizer,
            model,
            text,
            mentions,
            N_AUG_PER_EXAMPLE,
        )

        for j, new_text in enumerate(new_texts):
            new_id = f"{orig_id}_aug{j+1}"
            augmented_rows.append(
                {
                    ID_COLUMN: new_id,
                    TEXT_COLUMN: new_text,
                    LABEL_COLUMN: label,  # keep label=1
                }
            )

    print(f"Generated {len(augmented_rows)} augmented rows.")

    df_aug = pd.DataFrame(augmented_rows)

    # Combine original + augmented
    full_df = pd.concat([df, df_aug], ignore_index=True)

    # Optional: drop exact duplicates by text+label
    full_df = full_df.drop_duplicates(subset=[TEXT_COLUMN, LABEL_COLUMN])

    print(f"Final dataset size (original + augmented, deduped): {len(full_df)}")

    # Save
    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"Saving augmented dataset to: {OUTPUT_CSV}")
    full_df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
