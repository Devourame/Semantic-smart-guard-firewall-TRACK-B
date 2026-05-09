import json
import time
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from app.classifier import ai_classifier

with open("data/redteam_advanced.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)


results = []

y_true = []
y_pred = []

latencies = []

for item in dataset:

    prompt = item["prompt"]
    true_label = item["label"]

    start = time.time()

    verdict, category, confidence = ai_classifier(prompt)

    end = time.time()

    latency = (end - start) * 1000

    latencies.append(latency)

    predicted = "safe"

    if verdict == "unsafe":
        predicted = "unsafe"


    actual = "safe"

    if true_label != "safe":
        actual = "unsafe"


    y_true.append(actual)
    y_pred.append(predicted)


    results.append({
        "prompt": prompt,
        "true_label": true_label,
        "predicted": predicted,
        "category": category,
        "confidence": confidence,
        "latency_ms": latency
    })


# METRICS
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred, pos_label="unsafe")

recall = recall_score(y_true, y_pred, pos_label="unsafe")

f1 = f1_score(y_true, y_pred, pos_label="unsafe")

avg_latency = sum(latencies) / len(latencies)


# PRINT RESULTS
print("\n===== EVALUATION RESULTS =====")

print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1 Score : {f1:.2f}")
print(f"Avg Latency: {avg_latency:.2f} ms")


# SAVE CSV
df = pd.DataFrame(results)

df.to_csv("results/eval_results.csv", index=False)

print("\nResults saved to results/eval_results.csv")
import matplotlib.pyplot as plt
import seaborn as sns


# CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Safe", "Unsafe"],
    yticklabels=["Safe", "Unsafe"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("results/confusion_matrix.png")

print("\nConfusion matrix saved.")