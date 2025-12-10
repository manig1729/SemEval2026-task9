import matplotlib.pyplot as plt 
import torch
import pandas as pd
from collections import Counter

subtask_3_full_df = pd.read_csv("/projects/tejo9855/Projects/SemEval2026-task9/data/subtask3/train/eng.csv")

subtask_3_cols = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation"
]

counts = {}
for col in subtask_3_cols:
    counts[col] = Counter(subtask_3_full_df[col])[1]

total_count = subtask_3_full_df.shape[0]

print("Dict count: ", counts)
print("Total count: ", total_count)