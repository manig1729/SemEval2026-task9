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

INPUT_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_2/train_subtask2.csv"
OUTPUT_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_2/llm_aug/subtask2_train_llm_aug.csv"

LEXICON_PATH = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/lexicon_data/spurious_lexicon_manually_curated.txt"
N_AUG_PER_EXAMPLE = 2  # how many augmented variants to generate per example

# Label columns for subtask 2
SUBTASK2_LABEL_COLS = [
    "political",
    "racial/ethnic",
    "religious",
    "gender/sexual",
    "other",
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
    """
    Load local Llama-3.1-8B-Instruct in 4-bit mode using env var CURC_LLM_DIR.
    """
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
    """
    Use the LLM to generate n_aug counterfactual versions of 'text'
    where group references are replaced with different but comparable groups.
    """
    prompt = (
        "You are helping create counterfactual training data for polarization type classification.\n"
        "Given the following sentence, generate alternative versions that preserve the same meaning\n"
        "and polarization type, but replace references to specific countries, political groups,\n"
        "or identity groups with different but comparable groups.\n\n"
        f"Original sentence:\n{text}\n\n"
        f"Generate {n_aug} alternative versions, one per line."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_part = full_text[len(prompt):].strip()

    lines = [l.strip() for l in generated_part.split("\n") if l.strip()]
    return lines[:n_aug]


# ============================================================
#                          MAIN
# ============================================================

def main():
    print(f"Loading subtask 2 train data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in the input CSV.")

    for col in SUBTASK2_LABEL_COLS:
        if col not in df.columns:
            raise ValueError(
                f"Expected label column '{col}' in subtask 2 CSV. "
                "Edit SUBTASK2_LABEL_COLS if your names differ."
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
            continue  # only augment examples with spurious tokens

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
