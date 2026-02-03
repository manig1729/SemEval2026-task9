import os
import torch
import numpy as np
import pandas as pd
import json

from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score

# MODEL_NAME = "masked"
MODEL_NAME = "llm_aug"
# MODEL_NAME = "base"

MODEL_DIR = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/models/subtask_2_test/{MODEL_NAME}_model"
DEV_CSV  = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_2_test/val_subtask_2.csv"
THRESHOLDS_DIR = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/output/subtask_2_test/per_label_thresholds_{MODEL_NAME}.json"

MAX_LENGTH = 256
BATCH_SIZE = 16

SUBTASK2_LABEL_COLS = [
    "political",
    "racial/ethnic",
    "religious",
    "gender/sexual",
    "other",
]


class DevTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        if "text" not in df.columns:
            raise ValueError("Dev CSV must contain a 'text' column.")

        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def main():
    print(f"Loading dev CSV from: {DEV_CSV}")
    df = pd.read_csv(DEV_CSV)

    # Extract gold labels as a numpy array (shape: [num_examples, num_labels])
    y_true = df[SUBTASK2_LABEL_COLS].values.astype(int)

    print(f"Loading model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    dataset = DevTextDataset(df, tokenizer, max_length=MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "tmp_thresh"),
        per_device_eval_batch_size=BATCH_SIZE,
        do_predict=True,
    )

    trainer = Trainer(model=model, args=training_args)

    print("Running prediction on dev...")
    preds_output = trainer.predict(dataset)
    logits = preds_output.predictions

    # Probabilities
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    if probs.shape[1] != len(SUBTASK2_LABEL_COLS):
        raise ValueError(
            f"Model outputs {probs.shape[1]} labels but expected "
            f"{len(SUBTASK2_LABEL_COLS)}."
        )

    best_thresholds = {}

    # Simple grid search per label
    thresholds_to_try = np.linspace(0.05, 0.95, 19)

    for label_idx, label_name in enumerate(SUBTASK2_LABEL_COLS):
        label_true = y_true[:, label_idx]
        label_probs = probs[:, label_idx]

        best_f1 = -1.0
        best_t = 0.5

        for t in thresholds_to_try:
            label_pred = (label_probs >= t).astype(int)

            # Skip degenerate case if all zeros
            if label_pred.sum() == 0 and label_true.sum() == 0:
                f1 = 1.0
            else:
                f1 = f1_score(label_true, label_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        best_thresholds[label_name] = best_t
        print(f"Label: {label_name:15s}  best_t: {best_t:.2f}  best F1: {best_f1:.4f}")

    with open(THRESHOLDS_DIR, "w") as f:
        json.dump(best_thresholds, f, indent=2)

    print(f"Saved per-label thresholds to: {THRESHOLDS_DIR}")


if __name__ == "__main__":
    main()
