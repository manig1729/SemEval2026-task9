import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# ==============================================================
#               USER SETTINGS (EDIT THESE ONLY)
# ==============================================================

MODEL_NAME = "masked"
# MODEL_NAME = "llm_aug"
# MODEL_NAME = "base"

MODEL_DIR = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/models/subtask_2/{MODEL_NAME}_model"
TEST_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/data/subtask2/dev/eng.csv"

OUTPUT_CSV = f"/projects/tejo9855/Projects/SemEval2026-task9/predictions/subtask_2/pred_{MODEL_NAME}.csv"
METRICS_JSON = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/output/subtask_2/metrics_{MODEL_NAME}.json"

MAX_LENGTH = 128
BATCH_SIZE = 16

SUBTASK2_LABEL_COLS = [
    "political",
    "racial/ethnic",
    "religious",
    "gender/sexual",
    "other",
]

if MODEL_NAME == "base":
    PER_LABEL_THRESHOLDS = {
        "political": 0.15,
        "racial/ethnic": 0.25,
        "religious": 0.1,
        "gender/sexual": 0.05,
        "other": 0.05
    }
elif MODEL_NAME == "masked":
    PER_LABEL_THRESHOLDS = {
        "political": 0.1,
        "racial/ethnic": 0.25,
        "religious": 0.2,
        "gender/sexual": 0.05,
        "other": 0.2
    }
else:
    PER_LABEL_THRESHOLDS = {
        "political": 0.3,
        "racial/ethnic": 0.2,
        "religious": 0.39999999999999997,
        "gender/sexual": 0.05,
        "other": 0.05
        }


# ==============================================================
#                   DATASET (NO LABELS)
# ==============================================================

class TestTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        if "text" not in df.columns:
            raise ValueError("Test CSV must contain a 'text' column.")
        if "id" not in df.columns:
            raise ValueError("Test CSV must contain an 'id' column.")

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
#                            MAIN
# ==============================================================

def main():
    print(f"Loading test CSV from: {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)

    print(f"Loading model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # Build dataset
    dataset = TestTextDataset(df, tokenizer, max_length=MAX_LENGTH)

    # Dummy training args for Trainer.predict()
    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "tmp_eval"),
        per_device_eval_batch_size=BATCH_SIZE,
        do_predict=True,
    )

    trainer = Trainer(model=model, args=training_args)

    # Predict
    print("Running prediction...")
    preds_output = trainer.predict(dataset)
    logits = preds_output.predictions

    # Convert logits → probabilities → 0/1 predictions
    probs = torch.sigmoid(torch.tensor(logits))
    probs_np = probs.numpy()
    preds = np.zeros_like(probs_np, dtype=int)

    for i, col in enumerate(SUBTASK2_LABEL_COLS):
        t = PER_LABEL_THRESHOLDS[col]
        preds[:, i] = (probs_np[:, i] >= t).astype(int)

    # Build output DataFrame
    out_df = pd.DataFrame()
    out_df["id"] = df["id"]

    if preds.shape[1] != len(SUBTASK2_LABEL_COLS):
        raise ValueError(
            f"Model outputs {preds.shape[1]} labels but expected "
            f"{len(SUBTASK2_LABEL_COLS)}. Check model training."
        )

    # Add prediction columns
    for i, col in enumerate(SUBTASK2_LABEL_COLS):
        out_df[col] = preds[:, i]

    # Save result
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    print(f"Saving predictions to {OUTPUT_CSV}")
    out_df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
