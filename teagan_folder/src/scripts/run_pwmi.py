import math
import re
from collections import Counter
import torch
import pandas as pd

# ============================================================
#                   USER CONFIGURATION
# ============================================================

# Choose "unigram" or "bigram"
NGRAM_TYPE = "bigram"

INPUT_PATH = "/projects/tejo9855/Projects/SemEval2026-task9/data/subtask1/train/eng.csv"
OUTPUT_PATH = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/lexicon_data/token_pwmi_{NGRAM_TYPE}.csv"
SPURIOUS_LEXICON_PATH = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/lexicon_data/spurious_lexicon_{NGRAM_TYPE}_pmi_only.txt"

SPURIOUS_LABEL = 1
SPURIOUS_PMI_THRESHOLD = 0.75

MIN_TOKEN_COUNT = 10     # ignore tokens/ngrams appearing fewer than this overall
MIN_JOINT_COUNT = 5      # ignore token/ngram–label pairs that occur fewer times than this

# Simple regex tokenizer; replace with spaCy or HF tokenizer if desired
TOKEN_PATTERN = r"\b\w+\b"


# ============================================================
#                      TOKENIZATION
# ============================================================

def tokenize(text: str):
    """
    Very simple tokenizer:
    - lowercase
    - alphanumeric tokens using regex
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    return re.findall(TOKEN_PATTERN, text)


def extract_bigrams(tokens):
    """
    Given a list of tokens, return list of bigrams as strings: "tok1 tok2".
    """
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]


def get_units(tokens):
    """
    Return the sequence units (unigrams or bigrams) depending on NGRAM_TYPE.
    """
    if NGRAM_TYPE == "unigram":
        return tokens
    elif NGRAM_TYPE == "bigram":
        return extract_bigrams(tokens)
    else:
        raise ValueError(f"Unsupported NGRAM_TYPE: {NGRAM_TYPE}. Use 'unigram' or 'bigram'.")


# ============================================================
#                 MAIN PMI COMPUTATION
# ============================================================

def main():
    print(f"Running PMI with NGRAM_TYPE='{NGRAM_TYPE}'")
    print(f"Loading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    required_cols = {"id", "text", "polarization"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["polarization"] = df["polarization"].astype(int)

    # Counters
    token_counts = Counter()        # counts of units (unigrams or bigrams)
    label_token_counts = Counter()  # total number of units seen with each label
    joint_counts = Counter()        # counts of (unit, label) pairs

    print("Tokenizing and counting...")
    n = len(df)
    for idx, row in df.iterrows():
        if (idx + 1) % 1000 == 0 or idx == n - 1:
            print(f"  Processed {idx + 1}/{n}", end="\r")

        text = row["text"]
        label = row["polarization"]

        tokens = tokenize(text)
        units = get_units(tokens)  # unigrams or bigrams

        for u in units:
            token_counts[u] += 1
            label_token_counts[label] += 1
            joint_counts[(u, label)] += 1

    print("\nCounting complete.")

    total_tokens = sum(label_token_counts.values())
    print(f"Total units (according to NGRAM_TYPE='{NGRAM_TYPE}'): {total_tokens}")

    # P(label)
    label_probs = {
        label: label_token_counts[label] / total_tokens
        for label in label_token_counts
    }

    output_rows = []
    labels = sorted(label_token_counts.keys())

    print("Computing PMI...")

    for unit, unit_count in token_counts.items():
        if unit_count < MIN_TOKEN_COUNT:
            continue

        p_unit = unit_count / total_tokens

        for label in labels:
            joint = joint_counts.get((unit, label), 0)

            if joint < MIN_JOINT_COUNT:
                continue

            p_joint = joint / total_tokens
            p_label = label_probs[label]

            denom = p_unit * p_label
            if p_joint > 0 and denom > 0:
                pmi = math.log2(p_joint / denom)
            else:
                pmi = float("nan")

            output_rows.append({
                "token": unit,  # this will be a unigram or bigram string
                "label": label,
                "pmi": pmi,
                "joint_count": joint,
                "token_count": unit_count,
                "tokens_with_label": label_token_counts[label],
                "p_token": p_unit,
                "p_label": p_label,
                "p_joint": p_joint,
                "ngram_type": NGRAM_TYPE,
            })

    print(f"Computed PMI for {len(output_rows)} token–label pairs.")

    out_df = pd.DataFrame(output_rows)
    out_df = out_df.sort_values(by=["label", "pmi"], ascending=[True, False])

    print(f"Saving PMI results to: {OUTPUT_PATH}")
    out_df.to_csv(OUTPUT_PATH, index=False)

    # ============================================================
    #        BUILD AND SAVE SPURIOUS LEXICON
    # ============================================================
    print("Building spurious lexicon from PMI results...")

    spurious_df = out_df[
        (out_df["label"] == SPURIOUS_LABEL) &
        (out_df["pmi"] >= SPURIOUS_PMI_THRESHOLD)
    ].copy()

    # Sort by PMI descending, drop duplicates so each token appears once
    spurious_df = spurious_df.sort_values(by="pmi", ascending=False)
    spurious_tokens = (
        spurious_df["token"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Selected {len(spurious_tokens)} tokens for the spurious lexicon "
        f"(label={SPURIOUS_LABEL}, PMI >= {SPURIOUS_PMI_THRESHOLD}, "
        f"ngram_type={NGRAM_TYPE})."
    )

    # Write newline-separated lexicon file
    with open(SPURIOUS_LEXICON_PATH, "w", encoding="utf-8") as f:
        for tok in spurious_tokens:
            f.write(tok + "\n")

    print(f"Saved spurious lexicon to: {SPURIOUS_LEXICON_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
