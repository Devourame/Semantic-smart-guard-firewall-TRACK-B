import pandas as pd
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================================
# STEP 1: LOAD MODEL
# =========================================

print("\n[1] Loading model...")

MODEL_PATH = "./smartguard_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

print("[1 DONE] Model loaded successfully")

print("\n[2] Loading evaluation dataset...")

df = pd.read_csv(r"data\redteam_suiteinjection.csv")

print("[2 DONE] Dataset loaded")
print(df.head())

print("\n[3] Detecting columns...")

if "prompt" in df.columns:
    text_col = "prompt"
elif "text" in df.columns:
    text_col = "text"
else:
    raise Exception("No prompt/text column found")

if "label" in df.columns:
    label_col = "label"
elif "category" in df.columns:
    label_col = "category"
elif "true_label" in df.columns:
    label_col = "true_label"
else:
    raise Exception("No label column found")

print(f"TEXT COLUMN  : {text_col}")
print(f"LABEL COLUMN : {label_col}")

# =========================================
# STEP 4: CONVERT TO BINARY
# =========================================

print("\n[4] Converting labels to binary...")

def convert_to_binary(x):

    x = str(x).lower()

    if x == "safe":
        return "safe"

    return "unsafe"

df["binary_label"] = df[label_col].apply(convert_to_binary)

print(df["binary_label"].value_counts())

print("[4 DONE]")

# =========================================
# STEP 5: RUN PREDICTIONS
# =========================================

print("\n[5] Running inference...")

predictions = []
confidences = []
latencies = []

id2label = {
    0: "safe",
    1: "unsafe",
    2: "unsafe",
    3: "unsafe"
}

for i, text in enumerate(df[text_col]):

    start = time.time()

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    confidence, pred = torch.max(probs, dim=1)
    print(pred.item())
    pred_label = id2label[pred.item()]

    end = time.time()

    latency_ms = (end - start) * 1000

    predictions.append(pred_label)
    confidences.append(round(confidence.item(), 4))
    latencies.append(round(latency_ms, 2))

    if i % 10 == 0:
        print(f"Processed {i}/{len(df)} prompts")

print("[5 DONE] Inference complete")

# =========================================
# STEP 6: SAVE RESULTS
# =========================================

print("\n[6] Saving results CSV...")

df["prediction"] = predictions
df["confidence"] = confidences
df["latency_ms"] = latencies

df["correct"] = (
    df["binary_label"] == df["prediction"]
)

df.to_csv(
    "results/eval_results.csv",
    index=False
)

print("[6 DONE] Results saved")

# =========================================
# STEP 7: METRICS
# =========================================

print("\n[7] Computing metrics...")

accuracy = accuracy_score(
    df["binary_label"],
    df["prediction"]
)

precision = precision_score(
    df["binary_label"],
    df["prediction"],
    pos_label="unsafe",
    zero_division=0
)

recall = recall_score(
    df["binary_label"],
    df["prediction"],
    pos_label="unsafe",
    zero_division=0
)

f1 = f1_score(
    df["binary_label"],
    df["prediction"],
    pos_label="unsafe",
    zero_division=0
)

print("\n===== FINAL METRICS =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\n===== CLASSIFICATION REPORT =====")

print(classification_report(
    df["binary_label"],
    df["prediction"],
    zero_division=0
))

# =========================================
# STEP 8: CONFUSION MATRIX
# =========================================

print("\n[8] Generating confusion matrix...")

cm = confusion_matrix(
    df["binary_label"],
    df["prediction"]
)

print(cm)

plt.figure(figsize=(5,5))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("True")

plt.colorbar()

plt.savefig(
    "results/confusion_matrix.png"
)

print("[8 DONE] Confusion matrix saved")

# =========================================
# STEP 9: STRICTNESS CURVE
# =========================================

print("\n[9] Generating strictness curve...")

thresholds = np.arange(0.1, 1.0, 0.1)

accuracies = []
recalls = []

for t in thresholds:

    temp_preds = []

    for conf, pred in zip(
        df["confidence"],
        df["prediction"]
    ):

        if conf >= t:
            temp_preds.append(pred)
        else:
            temp_preds.append("safe")

    acc = accuracy_score(
        df["binary_label"],
        temp_preds
    )

    rec = recall_score(
        df["binary_label"],
        temp_preds,
        pos_label="unsafe",
        zero_division=0
    )

    accuracies.append(acc)
    recalls.append(rec)

plt.figure(figsize=(8,5))

plt.plot(
    thresholds,
    accuracies,
    marker="o",
    label="Accuracy"
)

plt.plot(
    thresholds,
    recalls,
    marker="o",
    label="Recall"
)

plt.xlabel("Threshold Strictness")
plt.ylabel("Score")

plt.title("Accuracy vs Strictness")

plt.legend()

plt.savefig(
    "results/accuracy_strictness_curve.png"
)

print("[9 DONE] Curve saved")

print("\n===== LATENCY =====")

print(
    f"Average latency: {np.mean(latencies):.2f} ms"
)

print(
    f"P95 latency: {np.percentile(latencies,95):.2f} ms"
)

print("\n===== EVERYTHING FINISHED SUCCESSFULLY =====")