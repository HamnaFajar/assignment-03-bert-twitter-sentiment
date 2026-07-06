"""
Assignment 03 - BERT Training on Twitter Sentiment Dataset
Course: Natural Language Processing Lab
-----------------------------------------------------------
Run this on Google Colab (free GPU) or any machine with a GPU.
It will:
  1. Load dataset/twitter_sentiment.csv
  2. Fine-tune bert-base-uncased for sentiment classification
  3. Evaluate (accuracy, precision, recall, F1, confusion matrix)
  4. Save 4 result graphs into results/
  5. Save the trained model + tokenizer into saved_bert_model/
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    Trainer, TrainingArguments
)

# ----------------------------
# 0. Config
# ----------------------------
DATA_PATH = "dataset/twitter_sentiment.csv"
TEXT_COL = "text"                # matches Twitter Airline Sentiment dataset
LABEL_COL = "airline_sentiment"  # matches Twitter Airline Sentiment dataset
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
SAVE_DIR = "saved_bert_model"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv(DATA_PATH, encoding="latin-1")
df = df[[TEXT_COL, LABEL_COL]].dropna()
df[TEXT_COL] = df[TEXT_COL].astype(str)

# Map labels -> integers (works for 2 or 3 class datasets)
label_names = sorted(df[LABEL_COL].unique().tolist())
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}
df["label_id"] = df[LABEL_COL].map(label2id)
num_labels = len(label_names)
print("Labels found:", label2id)

# Class distribution graph
plt.figure(figsize=(6, 4))
sns.countplot(x=LABEL_COL, data=df, order=label_names)
plt.title("Class Distribution of Dataset")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/class_distribution.png")
plt.close()

# ----------------------------
# 2. Train / test split
# ----------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[TEXT_COL].tolist(), df["label_id"].tolist(),
    test_size=0.2, random_state=42, stratify=df["label_id"]
)

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)


class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=MAX_LEN
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


train_dataset = TweetDataset(train_texts, train_labels)
val_dataset = TweetDataset(val_texts, val_labels)

# ----------------------------
# 3. Model
# ----------------------------
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels
)
model.to(device)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


training_args = TrainingArguments(
    output_dir="checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ----------------------------
# 4. Train
# ----------------------------
train_result = trainer.train()

# ----------------------------
# 5. Loss / Accuracy graphs from log history
# ----------------------------
logs = trainer.state.log_history
train_loss = [l["loss"] for l in logs if "loss" in l]
eval_loss = [l["eval_loss"] for l in logs if "eval_loss" in l]
eval_acc = [l["eval_accuracy"] for l in logs if "eval_accuracy" in l]

plt.figure(figsize=(6, 4))
plt.plot(eval_loss, label="Validation Loss", marker="o")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/loss_curve.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(eval_acc, label="Validation Accuracy", marker="o", color="green")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/accuracy_curve.png")
plt.close()

# ----------------------------
# 6. Final evaluation + confusion matrix
# ----------------------------
preds_output = trainer.predict(val_dataset)
y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = preds_output.label_ids

print(classification_report(y_true, y_pred, target_names=label_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
plt.close()

# ----------------------------
# 7. Save model + tokenizer + label map (used by the PyQt app)
# ----------------------------
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

import json
with open(os.path.join(SAVE_DIR, "label_map.json"), "w") as f:
    json.dump(id2label, f)

print(f"\nDone. Model saved to '{SAVE_DIR}/'. Graphs saved to '{RESULTS_DIR}/'.")
