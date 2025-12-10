import torch
import pandas as pd

PATH = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_2/llm_aug/train_llm_aug.csv"

def count_augmentations(csv_path, 
                        id_col="original_id_map",
                        original_id_col="original_id_map",
                        augmented_col="is_augmented"):

    df = pd.read_csv(csv_path)

    # All augmented rows
    df_aug = df[df[augmented_col] == 1]

    total_augmented_rows = len(df_aug)

    # Original instances that produced at least one augmentation
    unique_augmented_originals = df_aug[original_id_col].nunique()

    # Optional: augmented rows per original id
    per_original_counts = (
        df_aug.groupby(original_id_col)[augmented_col].count().sort_values(ascending=False)
    )

    print("========== Augmentation Statistics ==========")
    print(f"Total augmented rows: {total_augmented_rows}")
    print(f"Unique original examples augmented: {unique_augmented_originals}")
    print("\nAugmented rows per original example:")
    print(per_original_counts)
    print("=============================================")

    return {
        "total_augmented_rows": total_augmented_rows,
        "unique_augmented_originals": unique_augmented_originals,
        "per_original_aug_counts": per_original_counts
    }

stats = count_augmentations(PATH)

print(stats)