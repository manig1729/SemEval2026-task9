import os
import random

import numpy as np
import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ============================================================
#                     USER CONFIGURATION
# ============================================================

# INPUT_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_1/llm_aug/train_aug_groups.csv"   # path to your training CSV
# VAL_CSV   = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_1/val_subtask_1.csv" 
# TEXT_COLUMN = "text"                
# LABEL_COLUMN = "polarization"         # 0/1 column
# OUTPUT_DIR = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/models/subtask_1/llm_aug_model"     

INPUT_CSV = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_1/masked/train_masked_subtask_1.csv"   # path to your training CSV
VAL_CSV   = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/preprocessed_data/subtask_1/val_subtask_1.csv" 
TEXT_COLUMN = "masked_text"                 
LABEL_COLUMN = "polarization"         # 0/1 column
OUTPUT_DIR = "/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/models/subtask_1/masked_model"          # where to save final model

MODEL_NAME = "vinai/bertweet-base"    # HF model name
MAX_LENGTH = 128

TEST_SIZE = 0.2                       # fraction for validation
RANDOM_SEED = 42                      # for reproducibility

NUM_EPOCHS = 6
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 2
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

USE_CPU_ONLY = False                  # set True if you explicitly want no CUDA

# ============================================================
#                REPRODUCIBILITY HELPERS
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
#                   DATASET CLASSES
# ============================================================

class PolarizationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,  # dynamic padding via DataCollator
            max_length=self.max_length,
            return_tensors="pt",
        )

        # squeeze() to remove the batch dimension (1, seq_len) -> (seq_len,)
        item = {key: tensor.squeeze(0) for key, tensor in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)

        return item


# ============================================================
#                   METRICS FUNCTION
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"f1_macro": f1_macro}


# ============================================================
#                            MAIN
# ============================================================

def main():
    set_seed(RANDOM_SEED)

    # ---------- Load training data ----------
    print(f"Loading training data from: {INPUT_CSV}")
    df_train = pd.read_csv(INPUT_CSV).dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])

    train_texts = df_train[TEXT_COLUMN].astype(str).tolist()
    train_labels = df_train[LABEL_COLUMN].astype(int).tolist()

    # ---------- Load validation data ----------
    print(f"Loading validation data from: {VAL_CSV}")
    df_val = pd.read_csv(VAL_CSV)

    if TEXT_COLUMN not in df_val.columns:
        if "text" in df_val.columns:
            df_val[TEXT_COLUMN] = df_val["text"]
            print(f"VAL: '{TEXT_COLUMN}' not found, using 'text' instead.")
        else:
            raise ValueError(
                f"Validation CSV must contain '{TEXT_COLUMN}' or a 'text' column."
            )

    df_val = df_val.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    val_texts = df_val[TEXT_COLUMN].astype(str).tolist()
    val_labels = df_val[LABEL_COLUMN].astype(int).tolist()

    print(f"Train size: {len(train_texts)}")
    print(f"Val size:   {len(val_texts)}")

    # ---------- Tokenizer & model ----------
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    # For BERTweet specifically, you often want use_fast=False, but AutoTokenizer
    # will handle it; adjust if you see warnings.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # ---------- Datasets ----------
    train_dataset = PolarizationDataset(
        train_texts, train_labels, tokenizer, max_length=MAX_LENGTH
    )
    val_dataset = PolarizationDataset(
        val_texts, val_labels, tokenizer, max_length=MAX_LENGTH
    )

    # ---------- Data collator ----------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ---------- Training arguments ----------
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=False,  # could set True if you add a metric to monitor
        no_cuda=USE_CPU_ONLY,
        report_to="none",  # disable WandB/MLflow, etc.
    )

    # ---------- Trainer ----------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # ---------- Train ----------
    print("Starting training...")
    trainer.train()

    # ---------- Final eval ----------
    print("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    print("Eval results:", eval_results)

    # ---------- Save model & tokenizer ----------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving model and tokenizer to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    log_history = trainer.state.log_history
    df = pd.DataFrame(log_history)
    out_path = os.path.join(OUTPUT_DIR, "eval/loss_log_history.csv")
    df.to_csv(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
