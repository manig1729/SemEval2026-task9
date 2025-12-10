import torch
import pandas as pd

PATH = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_1/masked/train_masked.csv"

def count_blindness_masks(
    csv_path,
    text_col="text_masked",
    mask_token="MASKED_TOKEN"
):

    df = pd.read_csv(csv_path)

    # Count occurrences of mask token in each row
    df["mask_count"] = df[text_col].fillna("").str.count(mask_token)

    # Rows where at least one mask was applied
    masked_rows = df[df["mask_count"] > 0]

    num_masked_instances = len(masked_rows)
    total_masks_inserted = df["mask_count"].sum()

    print("========== Blindness Preprocessing Stats ==========")
    print(f"Total rows in dataset: {len(df)}")
    print(f"Rows containing at least one mask: {num_masked_instances}")
    print(f"Total MASKED_TOKEN occurrences: {total_masks_inserted}")
    print("\nTop masked examples (mask_count):")
    print(masked_rows.sort_values("mask_count", ascending=False).head(10))
    print("===================================================")

    return {
        "total_rows": len(df),
        "num_masked_instances": num_masked_instances,
        "total_masks_inserted": total_masks_inserted,
        "per_row_mask_counts": df["mask_count"]
    }

# stats = count_blindness_masks(PATH)

# print(stats)

print(len(pd.read_csv("/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_1/val_subtask_1.csv")))