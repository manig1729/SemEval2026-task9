#!/usr/bin/env python

import os
import re
from typing import List
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ============================================================
#                   USER CONFIGURATION
# ============================================================

INPUT_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_3_test/base/train_base.csv"
OUTPUT_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_3_test/llm_aug/train_llm_aug.csv"

LEXICON_PATH = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/lexicon_data/spurious_lexicon_manually_curated.txt"
N_AUG_PER_EXAMPLE = 2

# Label columns for subtask 3
SUBTASK3_LABEL_COLS = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation",
]


# ============================================================
#                  LEXICON / REGEX HELPERS
# ============================================================

def load_lexicon(path: str) -> List[str]:
    terms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            term = line.strip()
            if not term or line.startswith("#"):
                continue
            terms.append(term)
    return terms


def build_regex_patterns(terms: List[str]) -> List[re.Pattern]:
    patterns = []
    for term in terms:
        escaped = re.escape(term)
        pattern_str = escaped
        if re.match(r"\w", term[0]):
            pattern_str = r"\b" + pattern_str
        if re.match(r"\w", term[-1]):
            pattern_str = pattern_str + r"\b"
        patterns.append(re.compile(pattern_str, flags=re.IGNORECASE))
    return patterns


def contains_spurious(text: str, patterns: List[re.Pattern]) -> bool:
    if not isinstance(text, str):
        return False
    for p in patterns:
        if p.search(text):
            return True
    return False


# ============================================================
#                LLAMA MODEL LOADING / GENERATION
# ============================================================

def load_llama_model():
    CURC_LLM_DIR = os.getenv("CURC_LLM_DIR")
    if CURC_LLM_DIR is None:
        raise ValueError("Environment variable CURC_LLM_DIR is not set.")
    path_to_model = f"{CURC_LLM_DIR}/hf-transformers/Llama-3.1-8B-Instruct"
    print("Path to model:", path_to_model)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    model = AutoModelForCausalLM.from_pretrained(
        path_to_model,
        device_map="cuda",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    return tokenizer, model


def generate_augmented_texts(
    text: str,
    tokenizer,
    model,
    n_aug: int = 1,
    max_new_tokens: int = 128,
) -> List[str]:

    system_msg = (
        "You generate counterfactual training data for polarization type classification.\n"
        "Given a sentence, you rewrite it so that it preserves the same meaning and "
        "polarization type, but replaces references to specific countries, political groups, "
        "or identity groups with different but comparable groups.\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Return ONLY the rewritten sentence.\n"
        "- Do NOT provide any explanation or commentary.\n"
        "- Do NOT number or bullet the output.\n"
        "- Output MUST be a single line of text.\n"
    )

    user_msg = (
        "Rewrite the following sentence according to the rules:\n\n"
        f"{text}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Build chat-formatted input
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            num_return_sequences=n_aug,
        )

    generated_texts: List[str] = []

    # model_inputs.shape[1] = length of the prompt in tokens
    prompt_len = model_inputs.shape[1]

    for i in range(n_aug):
        output_ids = outputs[i]

        # Keep only tokens generated *after* the prompt
        gen_ids = output_ids[prompt_len:]

        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # --- Post-processing to kill explanations / extra junk ---

        # 1) Take only up to the first newline (model sometimes adds extra lines)
        line = decoded.split("\n", 1)[0].strip()

        # # 2) Optionally trim at common hallucinated markers (if they ever appear)
        # for stop_token in ["</OUTPUT>", "</output>", "<END>", "Explanation:"]:
        #     if stop_token in line:
        #         line = line.split(stop_token, 1)[0].strip()

        # # 3) Optionally enforce "first sentence only" behavior
        # # If the model sometimes generates multiple sentences, uncomment this:
        # # if "." in line:
        # #     first = line.split(".", 1)[0].strip()
        # #     if first:  # keep trailing period if it seems like a real sentence
        # #         line = first + "."

        # Final cleanup
        line = line.strip()

        if line:
            generated_texts.append(line)

    return generated_texts


# ============================================================
#                          MAIN
# ============================================================

def main():
    print(f"Loading subtask 3 train data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in the input CSV.")

    for col in SUBTASK3_LABEL_COLS:
        if col not in df.columns:
            raise ValueError(
                f"Expected label column '{col}' in subtask 3 CSV. "
                "Edit SUBTASK3_LABEL_COLS if your names differ."
            )

    print(f"Loading lexicon from: {LEXICON_PATH}")
    terms = load_lexicon(LEXICON_PATH)
    patterns = build_regex_patterns(terms)
    print(f"Loaded {len(terms)} lexicon entries.")

    print("Loading LLaMA model...")
    tokenizer, model = load_llama_model()
    print("Model loaded.")

    augmented_rows = []

    for idx, row in df.iterrows():
        text = row["text"]
        if not isinstance(text, str):
            continue

        if not contains_spurious(text, patterns):
            continue

        try:
            aug_texts = generate_augmented_texts(
                text,
                tokenizer=tokenizer,
                model=model,
                n_aug=N_AUG_PER_EXAMPLE,
            )
        except Exception as e:
            print(f"Warning: generation failed for row {idx}: {e}")
            continue

        for i, aug_text in enumerate(aug_texts):
            new_row = row.copy()
            new_row["id"] = f"{row['id']}_aug{i+1}"
            new_row["text"] = aug_text
            augmented_rows.append(new_row)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)} rows...")

    print(f"Generated {len(augmented_rows)} augmented rows.")
    aug_df = pd.DataFrame(augmented_rows)
    out_df = pd.concat([df, aug_df], ignore_index=True)

    print(f"Saving augmented data to: {OUTPUT_CSV}")
    out_df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
