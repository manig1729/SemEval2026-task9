import pandas as pd
import re
import random


# ============================================================
#                USER CONFIGURATION
# ============================================================

INPUT_PATH = "/Users/tejo9855/Documents/Classes/Fall '25/NLP - Martin/Assignments/SemEval2026-task9/data/subtask1/train/eng.csv"
OUTPUT_PATH = "/Users/tejo9855/Documents/Classes/Fall '25/NLP - Martin/Assignments/SemEval2026-task9/teagan_folder/src/output/train_counterfactual_aug.csv"

# Probability of swapping each matched token
P_SWAP = 0.7

# List of neutral group names for replacement
GROUP_NAMES = [
    "GroupA",
    "GroupB",
    "GroupC",
    "GroupD",
    "GroupE",
    "GroupF"
]

# OR — Uncomment this for *realistic* counterfactual augmentation:
"""
GROUP_NAMES = [
    "Democrats", "Republicans",
    "Liberals", "Conservatives",
    "Christians", "Muslims",
    "Atheists", "Israelis", "Palestinians",
    "Women", "Men",
    "Immigrants", "Citizens",
    "Scientists", "Teachers",
    "Students", "Parents",
]
"""

# Your spurious regex patterns
patterns_to_swap = [
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
#             RANDOM SWAP FUNCTION (probabilistic)
# ============================================================

def random_group_replacement(text, p=P_SWAP):
    """
    Replace each matched spurious token with a random group name
    with probability p.

    Uses span merging to avoid duplicate or overlapping replacements.
    """
    if not isinstance(text, str):
        return text

    spans = []  # spans to replace (start, end)

    for pattern in patterns_to_swap:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            if random.random() < p:
                spans.append((m.start(), m.end()))

    if not spans:
        return text

    # Merge overlapping spans
    spans.sort()
    merged = []
    cur_start, cur_end = spans[0]

    for start, end in spans[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    # Rebuild with random replacements
    output = []
    last = 0

    for start, end in merged:
        output.append(text[last:start])

        replacement = random.choice(GROUP_NAMES)
        output.append(replacement)

        last = end

    output.append(text[last:])

    return "".join(output)


# ============================================================
#                          MAIN
# ============================================================

def main():
    print(f"Loading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    print("Applying random group replacements...")

    df["swapped_text"] = df["text"].apply(random_group_replacement)

    print(f"Saving augmented data to: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
