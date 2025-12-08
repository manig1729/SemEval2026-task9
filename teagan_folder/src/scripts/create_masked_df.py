import torch
import pandas as pd
import re
import random


# ============================================================
#                    USER CONFIGURATION
# ============================================================

INPUT_PATH = "/projects/tejo9855/Projects/SemEval2026-task9/data/subtask1/train/eng.csv"
OUTPUT_PATH = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/output/preprocessed_train_data/train_masked_manually_curated_lexicon.csv"

P_MASK = 0.5   # probability of masking each matched token
MASK_TOKEN = "MASKED_TOKEN"
LEXICON_PATH = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/lexicon_data/spurious_lexicon_manually_curated.txt" 

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
      - escape the term (so '.' etc are treated literally),
      - wrap with word boundaries where appropriate,
      - compile with IGNORECASE.
    """
    patterns = []
    for term in terms:
        escaped = re.escape(term)

        # Add word boundaries at edges if the term starts/ends with a word char.
        # This keeps phrases like "black lives matter" working, but avoids
        # partial matches inside words.
        pattern_str = escaped
        if re.match(r"\w", term[0]):
            pattern_str = r"\b" + pattern_str
        if re.match(r"\w", term[-1]):
            pattern_str = pattern_str + r"\b"

        patterns.append(re.compile(pattern_str, flags=re.IGNORECASE))
    return patterns


# Build your patterns once at import time
LEXICON_TERMS = load_lexicon(LEXICON_PATH)
patterns_to_mask = build_regex_patterns(LEXICON_TERMS)


# ============================================================
#            MASKING FUNCTION (probabilistic)
# ============================================================

def mask_spurious_words(text, p=P_MASK):
    """
    Mask each matched spurious token with probability p.

    We apply regex finditer to locate spans and then rebuild the text.
    This avoids overlapping substitutions or accidentally masking too much.
    """
    if not isinstance(text, str):
        return text

    # Track spans for masking
    spans = []  # list of (start, end) intervals to mask

    for pattern in patterns_to_mask:
        for match in pattern.finditer(text):
            if random.random() < p:
                spans.append((match.start(), match.end()))

    if not spans:
        return text  # no changes

    # Merge overlapping spans
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

    # Rebuild text using merged spans
    output = []
    last_idx = 0

    for start, end in merged:
        output.append(text[last_idx:start])
        output.append(MASK_TOKEN)
        last_idx = end

    output.append(text[last_idx:])  # remainder
    return "".join(output)


# ============================================================
#                          MAIN
# ============================================================

def main():
    print(f"Loading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")

    print("Applying probabilistic masking...")
    df["masked_text"] = df["text"].apply(mask_spurious_words)

    print(f"Saving masked df to: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
