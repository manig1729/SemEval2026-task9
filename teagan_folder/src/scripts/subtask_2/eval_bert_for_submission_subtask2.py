import os
import torch
import pandas as pd
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

MODEL_DIR = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/models/subtask_2/subtask_2_model_masked"  
TEST_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/data/subtask2/dev/eng.csv"   
OUTPUT_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/predictions/subtask_2/pred_eng_masked.csv"

MAX_LENGTH = 256
BATCH_SIZE = 16

SUBTASK2_LABEL_COLS = [
    "political",
    "racial/ethnic",
    "religious",
    "gender/sexual",
    "other",
]


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
    preds = (probs >= 0.5).int().numpy()

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
