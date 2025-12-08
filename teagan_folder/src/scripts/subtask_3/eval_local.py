import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, precision_score, recall_score


# ==============================================================
#               USER SETTINGS (EDIT THESE ONLY)
# ==============================================================

MODEL_NAME = "masked"
# MODEL_NAME = "llm_aug"
# MODEL_NAME = "base"

MODEL_DIR = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/models/subtask_3/{MODEL_NAME}_model"
TEST_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_3/test_subtask_3.csv"

OUTPUT_CSV = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/output/subtask_3/pred_{MODEL_NAME}.csv"
METRICS_JSON = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/output/subtask_3/metrics_{MODEL_NAME}.json"


MAX_LENGTH = 128
BATCH_SIZE = 16

SUBTASK3_LABEL_COLS = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation",
]

if MODEL_NAME == "base":
    PER_LABEL_THRESHOLDS = {
        "stereotype": 0.3,
        "vilification": 0.2,
        "dehumanization": 0.05,
        "extreme_language": 0.49999999999999994,
        "lack_of_empathy": 0.25,
        "invalidation": 0.1
    }
elif MODEL_NAME == "masked":
    PER_LABEL_THRESHOLDS = {
        "stereotype": 0.25,
        "vilification": 0.44999999999999996,
        "dehumanization": 0.39999999999999997,
        "extreme_language": 0.39999999999999997,
        "lack_of_empathy": 0.25,
        "invalidation": 0.1
    }
else:
    PER_LABEL_THRESHOLDS = {
        "stereotype": 0.05,
        "vilification": 0.1,
        "dehumanization": 0.25,
        "extreme_language": 0.35,
        "lack_of_empathy": 0.25,
        "invalidation": 0.1
    }


# ==============================================================
#                           DATASET
# ==============================================================

class TestTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        if "text" not in df.columns:
            raise ValueError("Test CSV must contain 'text'.")
        if "id" not in df.columns:
            raise ValueError("Test CSV must contain 'id'.")

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


# ==============================================================
#                             MAIN
# ==============================================================

def main():
    print(f"Loading test CSV from: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)

    # Gold labels
    for col in SUBTASK3_LABEL_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing label column '{col}' in test CSV.")

    y_true = df[SUBTASK3_LABEL_COLS].values.astype(int)

    print(f"Loading model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    dataset = TestTextDataset(df, tokenizer, max_length=MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "tmp_eval_metrics"),
        per_device_eval_batch_size=BATCH_SIZE,
        do_predict=True,
    )

    trainer = Trainer(model=model, args=training_args)

    print("Running prediction...")
    preds_output = trainer.predict(dataset)
    logits = preds_output.predictions

    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    if probs.shape[1] != len(SUBTASK3_LABEL_COLS):
        raise ValueError(
            f"Model outputs {probs.shape[1]} labels but expected "
            f"{len(SUBTASK3_LABEL_COLS)}."
        )

    # Apply thresholds
    preds = np.zeros_like(probs, dtype=int)
    for i, col in enumerate(SUBTASK3_LABEL_COLS):
        t = PER_LABEL_THRESHOLDS[col]
        preds[:, i] = (probs[:, i] >= t).astype(int)

    # ---------------------------
    # Compute metrics
    # ---------------------------

    metrics = {}

    # Global metrics (micro over all labels)
    metrics["precision_micro"] = precision_score(
        y_true, preds, average="micro", zero_division=0
    )
    metrics["recall_micro"] = recall_score(
        y_true, preds, average="micro", zero_division=0
    )
    metrics["f1_macro"] = f1_score(
        y_true, preds, average="macro", zero_division=0
    )

    # Per-label metrics
    per_label_metrics = {}
    for i, col in enumerate(SUBTASK3_LABEL_COLS):
        y_true_i = y_true[:, i]
        y_pred_i = preds[:, i]

        precision_i = precision_score(
            y_true_i, y_pred_i, zero_division=0
        )
        recall_i = recall_score(
            y_true_i, y_pred_i, zero_division=0
        )
        f1_i = f1_score(
            y_true_i, y_pred_i, zero_division=0
        )
        support_i = int(y_true_i.sum())

        per_label_metrics[col] = {
            "precision": precision_i,
            "recall": recall_i,
            "f1": f1_i,
            "support": support_i,
        }

    metrics["per_label"] = per_label_metrics

    # Save metrics JSON
    os.makedirs(os.path.dirname(METRICS_JSON), exist_ok=True)
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nSaved metrics to: {METRICS_JSON}")

    # ---------------------------
    # Save predictions CSV
    # ---------------------------

    out_df = pd.DataFrame({"id": df["id"]})
    for i, col in enumerate(SUBTASK3_LABEL_COLS):
        out_df[col] = preds[:, i]
        out_df[col + "_prob"] = probs[:, i]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved predictions to: {OUTPUT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
