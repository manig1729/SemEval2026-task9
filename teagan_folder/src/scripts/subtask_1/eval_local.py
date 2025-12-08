import os
import json
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ============================================================
#                    USER CONFIGURATION
# ============================================================

MODEL_DIR = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/models/subtask_1/masked_model_uni_bi_gram_pmi"
TEST_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_1/test_subtask_1.csv"

ID_COLUMN = "id"
TEXT_COLUMN = "text"
LABEL_COLUMN = "polarization"   # gold label column

OUTPUT_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/output/subtask_1/pred_eng_uni_bi_gram_pmi.csv"
METRICS_JSON = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/output/subtask_1/metrics_eng_uni_bi_gram_pmi.json"

MAX_LENGTH = 128
USE_CPU_ONLY = True  # set True if you want to force CPU


# ============================================================
#                     DATASET CLASS
# ============================================================

class TestPolarizationDataset(torch.utils.data.Dataset):
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
        return {key: tensor.squeeze(0) for key, tensor in encoding.items()}


# ============================================================
#                            MAIN
# ============================================================

def main():
    print(f"Loading test data from: {TEST_CSV}")
    df_test = pd.read_csv(TEST_CSV)

    for col in [ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN]:
        if col not in df_test.columns:
            raise ValueError(f"Column '{col}' not found in test CSV.")

    ids = df_test[ID_COLUMN].tolist()
    texts = df_test[TEXT_COLUMN].astype(str).tolist()
    y_true = df_test[LABEL_COLUMN].astype(int).values

    print(f"Loading model and tokenizer from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    test_dataset = TestPolarizationDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./tmp_pred_subtask1",
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

    print("Running prediction on test set...")
    preds_output = trainer.predict(test_dataset)
    logits = preds_output.predictions  # (num_examples, 2)

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pos_probs = probs[:, 1]

    y_pred = np.argmax(logits, axis=1)

    # ---------------- Metrics (binary, positive class = 1) ----------------
    accuracy = accuracy_score(y_true, y_pred)

    precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    # support for positive class = count of gold positives
    support_pos = int((y_true == 1).sum())

    metrics = {
        "accuracy": float(accuracy),
        "precision_positive": float(precision_pos),
        "recall_positive": float(recall_pos),
        "f1_positive": float(f1_pos),
        "support_positive": support_pos,
    }

    os.makedirs(os.path.dirname(METRICS_JSON), exist_ok=True)
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved metrics to: {METRICS_JSON}")

    # ---------------- Save predictions ----------------
    submission_df = pd.DataFrame({
        "id": ids,
        "polarization": y_pred,
        "polarization_prob": pos_probs,
    })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    submission_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved predictions to: {OUTPUT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
