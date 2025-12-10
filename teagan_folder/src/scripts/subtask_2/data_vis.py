import matplotlib.pyplot as plt 
import torch
import pandas as pd
from collections import Counter

subtask_2_full_df = pd.read_csv("/projects/tejo9855/Projects/SemEval2026-task9/data/subtask2/train/eng.csv")

subtask_2_cols = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]

counts = {}
for col in subtask_2_cols:
    counts[col] = Counter(subtask_2_full_df[col])[1]

total_count = subtask_2_full_df.shape[0]

print("Dict count: ", counts)
print("Total count: ", total_count)