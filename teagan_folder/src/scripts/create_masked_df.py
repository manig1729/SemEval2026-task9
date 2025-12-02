import pandas as pd
import re
import random


# ============================================================
#                    USER CONFIGURATION
# ============================================================

INPUT_PATH = "/Users/tejo9855/Documents/Classes/Fall '25/NLP - Martin/Assignments/SemEval2026-task9/data/subtask1/train/eng.csv"
OUTPUT_PATH = "/Users/tejo9855/Documents/Classes/Fall '25/NLP - Martin/Assignments/SemEval2026-task9/teagan_folder/src/output/train_masked.csv"

P_MASK = 0.5   # probability of masking each matched token
MASK_TOKEN = "MASKED_TOKEN"


# Your spurious regex patterns
patterns_to_mask = [
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
    r"\bfacebook\b"
]


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
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
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
