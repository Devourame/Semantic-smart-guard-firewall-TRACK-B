import pandas as pd

# ==========================================
# LOAD JAILBREAK DATASET
# ==========================================

jb = pd.read_csv("data/jailbreak_prompts.csv")

# keep only true jailbreaks
jb = jb[jb["jailbreak"] == True]

# keep only needed columns
jb_df = pd.DataFrame({
    "text": jb["prompt"],
    "label": "jailbreak"
})

print(f"Jailbreak prompts: {len(jb_df)}")

# ==========================================
# LOAD PROMPT INJECTION DATASET
# ==========================================

inj = pd.read_csv("data/malignant.csv")

inj_df = pd.DataFrame({
    "text": inj["text"],
    "label": "prompt_injection"
})

print(f"Injection prompts: {len(inj_df)}")

# ==========================================
# COMBINE
# ==========================================

final_df = pd.concat([
    jb_df,
    inj_df
])

# remove null rows
final_df = final_df.dropna()

# remove duplicates
final_df = final_df.drop_duplicates(subset=["text"])

# shuffle dataset
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