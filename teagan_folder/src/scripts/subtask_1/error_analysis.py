import torch
import pandas as pd

test_labels = pd.read_csv("/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_1/test_subtask_1.csv")
masked_predictions = pd.read_csv("/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/output/subtask_1/pred_masked.csv")

df_merged = test_labels.merge(masked_predictions, on="id", how="left")

df_merged.to_excel("/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/output/subtask_1/merged_test_masked.xlsx", index=False)
