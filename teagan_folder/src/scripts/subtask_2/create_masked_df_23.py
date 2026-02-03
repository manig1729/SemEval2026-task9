
import re
from typing import List
import torch
import pandas as pd
import random

# ============================================================
#                   USER CONFIGURATION
# ============================================================

# Set which subtask this file is for: 2 or 3
TASK = 3

INPUT_CSV = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_{TASK}_test/base/train_base.csv"
OUTPUT_CSV = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_{TASK}_test/masked/train_subtask{TASK}_masked.csv"

LEXICON_PATH = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/lexicon_data/spurious_lexicon_manually_curated.txt"
MASK_TOKEN = "MASKED_TOKEN"
P_MASK = 0.5  # probability of masking a matched span


# ============================================================
#                  LEXICON / REGEX HELPERS
# ============================================================

def load_lexicon(path: str) -> List[str]:
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


def build_regex_patterns(terms: List[str]) -> List[re.Pattern]:
    """
    Turn each lexicon term into a compiled regex pattern.
    - Escapes special characters
    - Adds word boundaries if term starts/ends with a word char
    - Case-insensitive
    Works for unigrams and multi-word phrases.
    """
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


# ============================================================
#                     MASKING FUNCTION
# ============================================================

def mask_spurious_words(
    text: str,
    patterns: List[re.Pattern],
    mask_token: str = MASK_TOKEN,
    p: float = P_MASK,
) -> str:
    """
    Mask each matched spurious token/phrase with probability p.

    We:
    - Find all match spans
    - Sample which ones to mask
    - Merge overlapping spans
    - Rebuild the text with mask_token in those spans
    """
    if not isinstance(text, str):
        return text

    spans = []

    for pattern in patterns:
        for m in pattern.finditer(text):
            if random.random() < p:
                spans.append((m.start(), m.end()))

    if not spans:
        return text

    spans.sort()
    merged = []
    current_start, current_end = spans[0]

    for start, end in spans[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))

    output_chunks = []
    last_idx = 0
    for start, end in merged:
        output_chunks.append(text[last_idx:start])
        output_chunks.append(mask_token)
        last_idx = end
    output_chunks.append(text[last_idx:])

    return "".join(output_chunks)


# ============================================================
#                          MAIN
# ============================================================

def main():
    print(f"Subtask: {TASK}")
    print(f"Loading lexicon from: {LEXICON_PATH}")
    terms = load_lexicon(LEXICON_PATH)
    print(f"Loaded {len(terms)} lexicon entries.")

    print("Compiling regex patterns...")
    patterns = build_regex_patterns(terms)
    print(f"Compiled {len(patterns)} patterns.")

    print(f"Loading train CSV from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in the input CSV.")

    print("Masking spurious terms...")
    df["text_masked"] = df["text"].apply(
        lambda t: mask_spurious_words(
            t,
            patterns=patterns,
            mask_token=MASK_TOKEN,
            p=P_MASK,
        )
    )

    print(f"Saving masked data to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
