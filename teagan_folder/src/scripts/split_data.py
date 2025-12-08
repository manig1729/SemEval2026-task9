#!/usr/bin/env python
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
#                 USER CONFIGURATION
# ============================================================

TASK = 3
INPUT_CSV = f"/projects/tejo9855/Projects/SemEval2026-task9/data/subtask{TASK}/train/eng.csv"

TRAIN_OUT = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_{TASK}/train_subtask_{TASK}.csv"
VAL_OUT   = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_{TASK}/val_subtask_{TASK}.csv"
TEST_OUT  = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_{TASK}/test_subtask_{TASK}.csv"

# Percentages for splitting (must sum to 1.0 or less — remaining goes to train)
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# For reproducibility
RANDOM_SEED = 42


# ============================================================
#                        MAIN LOGIC
# ============================================================

def main():

    print(f"Loading full training data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows.")

    if TRAIN_RATIO + VAL_RATIO + TEST_RATIO > 1.0:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must be ≤ 1.0.")

    # ------------------------------------------------------------
    # Step 1: Train vs (Val+Test)
    # ------------------------------------------------------------
    val_test_ratio = VAL_RATIO + TEST_RATIO

    if val_test_ratio > 0:
        df_train, df_val_test = train_test_split(
            df,
            test_size=val_test_ratio,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
    else:
        df_train = df
        df_val_test = pd.DataFrame(columns=df.columns)

    # ------------------------------------------------------------
    # Step 2: Split (Val+Test) into Val and Test
    # ------------------------------------------------------------
    if val_test_ratio > 0:
        relative_val_ratio = VAL_RATIO / val_test_ratio if val_test_ratio > 0 else 0.0

        df_val, df_test = train_test_split(
            df_val_test,
            test_size=(1 - relative_val_ratio),
            random_state=RANDOM_SEED,
            shuffle=True,
        )
    else:
        df_val = pd.DataFrame(columns=df.columns)
        df_test = pd.DataFrame(columns=df.columns)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    print(f"Saving train set ({len(df_train)} rows) → {TRAIN_OUT}")
    df_train.to_csv(TRAIN_OUT, index=False)

    print(f"Saving validation set ({len(df_val)} rows) → {VAL_OUT}")
    df_val.to_csv(VAL_OUT, index=False)

    print(f"Saving test set ({len(df_test)} rows) → {TEST_OUT}")
    df_test.to_csv(TEST_OUT, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
