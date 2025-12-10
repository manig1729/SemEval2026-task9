import json

subtask = "1"
# model_name = "base"
# model_name = "llm_aug"
model_name = "masked"

LOG_PATH = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/models/subtask_{subtask}/{model_name}_model/log_history.json"

OUT_PATH = f"/projects/tejo9855/Projects/SemEval2026-task9/teagan_folder/src/output/subtask_{subtask}/train_val_loss_{model_name}_model.png"

with open(LOG_PATH, "r") as f:
    log_history = json.load(f)

print(f"Loaded {len(log_history)} log entries.")

train_steps, train_loss = [], []
eval_steps, eval_loss, eval_f1 = [], [], []
train_epochs, eval_epochs = [], []
    
for entry in log_history:
    
    # Training logs (must contain "loss" but not "eval_loss")
    if "loss" in entry and "eval_loss" not in entry:
        train_steps.append(entry["step"])
        train_loss.append(entry["loss"])
        train_epochs.append(entry["epoch"])
    
    # Evaluation logs
    if "eval_loss" in entry:
        eval_steps.append(entry["step"])
        eval_loss.append(entry["eval_loss"])
        eval_f1.append(entry.get("eval_f1_macro"))
        eval_epochs.append(entry["epoch"])

import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
plt.plot(train_epochs, train_loss, marker="o", label="Train Loss")
plt.plot(eval_epochs, eval_loss, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.tight_layout()

# --- SAVE ---
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
# plt.savefig("train_val_loss.pdf", bbox_inches="tight")  # optional PDF
plt.close()

print("Saved")