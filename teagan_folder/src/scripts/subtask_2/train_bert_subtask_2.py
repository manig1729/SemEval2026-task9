#!/usr/bin/env python

from typing import List, Dict, Any

import numpy as np
import torch
import pandas as pd

import json
import os

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, precision_score, recall_score


# ============================================================
#                   USER CONFIGURATION
# ============================================================

# MODEL_NAME = "masked"
# MODEL_NAME = "llm_aug"
MODEL_NAME = "base"

TRAIN_CSV = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_2/{MODEL_NAME}/train_{MODEL_NAME}.csv" 
VAL_CSV   = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_2/val_subtask_2.csv" 
          
OUTPUT_DIR = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/models/subtask_2/{MODEL_NAME}_model"
if MODEL_NAME == "masked":
    TEXT_COLUMN = "text_masked"      
else:
    TEXT_COLUMN = "text"  

MODEL_NAME = "vinai/bertweet-base"
BATCH_SIZE = 8
NUM_EPOCHS = 3.0
MAX_LENGTH = 128

SUBTASK2_LABEL_COLS = [
    "political",
    "racial/ethnic",
    "religious",
    "gender/sexual",
    "other",
]


# ============================================================
#                        DATASET
# ============================================================

class MultiLabelDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        label_cols: List[str],
        max_length: int = 256,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_cols = label_cols
        self.max_length = max_length

        if TEXT_COLUMN not in df.columns:
            raise ValueError(f"Expected a {TEXT_COLUMN} column in the dataframe.")

        for col in self.label_cols:
            if col not in df.columns:
                raise ValueError(f"Missing label column: {col}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        text = row[TEXT_COLUMN]
        labels = row[self.label_cols].values.astype(float)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(labels, dtype=torch.float)
        return item


# ============================================================
#                    METRICS FUNCTION
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).int().numpy()
    labels = labels.int().numpy()

    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


# ============================================================
#                          MAIN
# ============================================================

def main():
    print(f"Loading train data from: {TRAIN_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)

    print(f"Loading validation data from: {VAL_CSV}")
    val_df = pd.read_csv(VAL_CSV)

    if TEXT_COLUMN not in val_df.columns:
        if "text" in val_df.columns:
            val_df[TEXT_COLUMN] = val_df["text"]
            print(f"VAL: '{TEXT_COLUMN}' not found, using 'text' column instead.")
        else:
            raise ValueError(
                f"Validation CSV must contain '{TEXT_COLUMN}' "
                f"or a 'text' column."
            )

    for col in SUBTASK2_LABEL_COLS:
        if col not in train_df.columns:
            raise ValueError(f"Train CSV missing label column: {col}")
        if col not in val_df.columns:
            raise ValueError(f"Val CSV missing label column: {col}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = MultiLabelDataset(
        train_df,
        tokenizer=tokenizer,
        label_cols=SUBTASK2_LABEL_COLS,
        max_length=MAX_LENGTH,
    )
    val_dataset = MultiLabelDataset(
        val_df,
        tokenizer=tokenizer,
        label_cols=SUBTASK2_LABEL_COLS,
        max_length=MAX_LENGTH,
    )

    num_labels = len(SUBTASK2_LABEL_COLS)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="no",
        save_steps=None,
        load_best_model_at_end=False,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        # no_cuda=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")

    print(f"Saving best model to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")

    with open(os.path.join(OUTPUT_DIR, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)


if __name__ == "__main__":
    main()
