import math
import re
from collections import Counter
import pandas as pd


# ============================================================
#                   USER CONFIGURATION
# ============================================================

INPUT_PATH = "/Users/tejo9855/Documents/Classes/Fall '25/NLP - Martin/Assignments/SemEval2026-task9/data/subtask1/train/eng.csv"            # path to your CSV
OUTPUT_PATH = "/Users/tejo9855/Documents/Classes/Fall '25/NLP - Martin/Assignments/SemEval2026-task9/teagan_folder/src/output/token_pwmi.csv"    # where results will go

MIN_TOKEN_COUNT = 10     # ignore tokens appearing fewer than this overall
MIN_JOINT_COUNT = 5      # ignore token–label pairs that occur fewer times than this

# Simple regex tokenizer; replace with spaCy or HF tokenizer if desired
TOKEN_PATTERN = r"\b\w+\b"


# ============================================================
#                      TOKENIZER
# ============================================================

def tokenize(text):
    """
    Very simple tokenizer:
    - lowercase
    - alphanumeric tokens using regex
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    return re.findall(TOKEN_PATTERN, text)


# ============================================================
#                 MAIN PMI COMPUTATION
# ============================================================

def main():
    print(f"Loading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    required_cols = {"id", "text", "polarization"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["polarization"] = df["polarization"].astype(int)

    # Counters
    token_counts = Counter()
    label_token_counts = Counter()
    joint_counts = Counter()

    print("Tokenizing and counting...")
    n = len(df)
    for idx, row in df.iterrows():
        if (idx + 1) % 1000 == 0 or idx == n - 1:
            print(f"  Processed {idx + 1}/{n}", end="\r")

        text = row["text"]
        label = row["polarization"]

        tokens = tokenize(text)
        for tok in tokens:
            token_counts[tok] += 1
            label_token_counts[label] += 1
            joint_counts[(tok, label)] += 1

    print("\nCounting complete.")

    total_tokens = sum(label_token_counts.values())
    print(f"Total tokens: {total_tokens}")

    # P(label)
    label_probs = {
        label: label_token_counts[label] / total_tokens
        for label in label_token_counts
    }

    output_rows = []
    labels = sorted(label_token_counts.keys())

    print("Computing PMI...")

    for token, token_count in token_counts.items():
        if token_count < MIN_TOKEN_COUNT:
            continue

        p_token = token_count / total_tokens

        for label in labels:
            joint = joint_counts.get((token, label), 0)

            if joint < MIN_JOINT_COUNT:
                continue

            p_joint = joint / total_tokens
            p_label = label_probs[label]

            denom = p_token * p_label
            if p_joint > 0 and denom > 0:
                pmi = math.log2(p_joint / denom)
            else:
                pmi = float("nan")

            output_rows.append({
                "token": token,
                "label": label,
                "pmi": pmi,
                "joint_count": joint,
                "token_count": token_count,
                "tokens_with_label": label_token_counts[label],
                "p_token": p_token,
                "p_label": p_label,
                "p_joint": p_joint,
            })

    print(f"Computed PMI for {len(output_rows)} token–label pairs.")

    out_df = pd.DataFrame(output_rows)
    out_df = out_df.sort_values(by=["label", "pmi"], ascending=[True, False])

    print(f"Saving results to: {OUTPUT_PATH}")
    out_df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
