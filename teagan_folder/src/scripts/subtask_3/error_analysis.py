import torch
import pandas as pd

test_labels = pd.read_csv("/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_3/test_subtask_3.csv")
masked_predictions = pd.read_csv("/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/output/subtask_3/pred_masked.csv")

df_merged = test_labels.merge(masked_predictions, on="id", how="left")

print(df_merged)

df_merged.to_excel("/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/output/subtask_3/merged_test_masked.xlsx", index=False)
