import pandas as pd

# ==========================================
# LOAD JAILBREAK DATASET
# ==========================================

jb = pd.read_csv("data/jailbreak_prompts.csv")

jb = jb[jb["jailbreak"] == True]

jb_df = pd.DataFrame({
    "text": jb["prompt"],
    "label": "unsafe"
})

print(f"Jailbreak prompts: {len(jb_df)}")

# ==========================================
# LOAD INJECTION DATASET
# ==========================================

inj = pd.read_csv("data/malignant.csv")

inj_df = pd.DataFrame({
    "text": inj["text"],
    "label": "unsafe"
})

print(f"Injection prompts: {len(inj_df)}")

# ==========================================
# SAFE PROMPTS
# ==========================================

safe_prompts = [

    "Explain gravity in simple words.",
    "Write a short poem about rain.",
    "How do neural networks work?",
    "What is machine learning?",
    "Explain photosynthesis.",
    "How do airplanes fly?",
    "Write Python code for bubble sort.",
    "What causes earthquakes?",
    "Summarize World War 2.",
    "Explain TCP/IP networking.",
    "Best methods to study mathematics?",
    "How does blockchain work?",
    "Explain recursion in programming.",
    "Write a story about friendship.",
    "How does email encryption work?",
    "What is overfitting in AI?",
    "Explain Newton's laws.",
    "How do databases work?",
    "What is cloud computing?",
    "How does a CPU work?"
]

safe_df = pd.DataFrame({
    "text": safe_prompts,
    "label": "safe"
})

# ==========================================
# MERGE
# ==========================================

final_df = pd.concat([
    jb_df,
    inj_df,
    safe_df
])

# cleanup
final_df = final_df.dropna()

final_df = final_df.drop_duplicates(subset=["text"])

# shuffle
final_df = final_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ==========================================
# SAVE
# ==========================================

OUTPUT_PATH = "data/hybrid_redsuitetrain.csv"

final_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ==========================================
# STATS
# ==========================================

print("\n===== FINAL DATASET =====")

print(final_df["label"].value_counts())

print(f"\nTotal samples: {len(final_df)}")

print(f"\nSaved to: {OUTPUT_PATH}")