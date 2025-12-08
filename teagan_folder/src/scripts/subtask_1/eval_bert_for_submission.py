import os
import numpy as np
import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)


# ============================================================
#                    USER CONFIGURATION
# ============================================================

MODEL_DIR = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/models/subtask_1/masked_model"  
TEST_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/data/subtask1/dev/eng.csv"       
ID_COLUMN = "id"                        
TEXT_COLUMN = "text"                

OUTPUT_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/predictions/subtask_1/pred_eng_masked.csv"     # SemEval submission file

MAX_LENGTH = 128
USE_CPU_ONLY = True                      # set True if you want to force CPU


# ============================================================
#                     DATASET CLASS
# ============================================================

class TestPolarizationDataset(torch.utils.data.Dataset):
    """
    Simple dataset for inference without labels.
    """
    def __init__(self, texts, tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: tensor.squeeze(0) for key, tensor in encoding.items()}
        return item


# ============================================================
#                            MAIN
# ============================================================

def main():
    # ---------- Load test data ----------
    print(f"Loading test data from: {TEST_CSV}")
    df_test = pd.read_csv(TEST_CSV)

    if ID_COLUMN not in df_test.columns:
        raise ValueError(f"Column '{ID_COLUMN}' not found in test CSV.")
    if TEXT_COLUMN not in df_test.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found in test CSV.")

    ids = df_test[ID_COLUMN].tolist()
    texts = df_test[TEXT_COLUMN].astype(str).tolist()

    # ---------- Load model & tokenizer ----------
    print(f"Loading model and tokenizer from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # ---------- Build dataset & data collator ----------
    test_dataset = TestPolarizationDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ---------- Dummy training args for prediction ----------
    # We don't actually train here; Trainer is used for its predict() method.
    training_args = TrainingArguments(
        output_dir="./tmp_pred",
        per_device_eval_batch_size=32,
        do_train=False,
        do_eval=False,
        do_predict=True,
        no_cuda=USE_CPU_ONLY,
        report_to="none",
        logging_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
    )

    # ---------- Predict ----------
    print("Running prediction on test set...")
    preds_output = trainer.predict(test_dataset)
    logits = preds_output.predictions  # shape: (num_examples, num_labels)

    # Argmax over logits -> label (0 or 1)
    pred_labels = np.argmax(logits, axis=1)

    # ---------- Build submission dataframe ----------
    submission_df = pd.DataFrame({
        "id": ids,
        "polarization": pred_labels,
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    print(f"Saving predictions to: {OUTPUT_CSV}")
    submission_df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
