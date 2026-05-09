import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("data/hybrid_redsuitetrain.csv")

df = df.dropna()

# =====================================
# LABEL MAP
# =====================================

label2id = {
    "safe": 0,
    "unsafe": 1
}

id2label = {
    0: "safe",
    1: "unsafe"
}

df["label"] = df["label"].map(label2id)

# =====================================
# HF DATASET
# =====================================

dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

# =====================================
# TOKENIZER
# =====================================

MODEL_NAME = "microsoft/deberta-v3-small"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

def tokenize(batch):

    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

dataset = dataset.map(tokenize)

# =====================================
# MODEL
# =====================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# =====================================
# METRICS
# =====================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = logits.argmax(axis=-1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# =====================================
# TRAINING ARGS
# =====================================

training_args = TrainingArguments(

    output_dir="./deberta_results",

    do_eval=True,

    save_steps=50,

    logging_steps=10,

    learning_rate=2e-5,

    num_train_epochs=3,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    weight_decay=0.01,

    report_to="none",

    fp16=False
)

# =====================================
# TRAINER
# =====================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset["train"],

    eval_dataset=dataset["test"],

    compute_metrics=compute_metrics
)

# =====================================
# TRAIN
# =====================================

print("\n===== TRAINING STARTED =====\n")

trainer.train()

# =====================================
# SAVE MODEL
# =====================================

trainer.save_model("./smartguard_model")

tokenizer.save_pretrained("./smartguard_model")

print("\n===== MODEL SAVED SUCCESSFULLY =====")