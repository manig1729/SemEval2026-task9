import matplotlib.pyplot as plt 
import torch
import pandas as pd
from collections import Counter

subtask_1_full_df = pd.read_csv("/projects/tejo9855/Projects/SemEval2026-task9/data/subtask1/train/eng.csv")

count_polarization = Counter(subtask_1_full_df['polarization'])
total_count = subtask_1_full_df.shape[0]

print("Polarization count: ", count_polarization)
print("Total count: ", total_count)