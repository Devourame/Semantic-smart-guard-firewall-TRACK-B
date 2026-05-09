import pandas as pd
from pathlib import Path


# =========================================
# PATHS
# =========================================

BASE_DIR = Path(__file__).resolve().parent

PROMPTS_DIR = BASE_DIR / "benchmark" / "llm-red-teaming-dataset" / "prompts"



OUTPUT_FILE = BASE_DIR / "data" / "merged_dataset.csv"


# =========================================
# CATEGORY MAP
# =========================================

CATEGORY_MAP = {

    "redteam_jailbreaking.csv": "jailbreak",

    "redteam_privacy_violations.csv": "pii",

    "redteam_harmful_content.csv": "toxic",

    "redteam_manipulation.csv": "prompt_injection",

    "redteam_misinformation.csv": "toxic",

    "redteam_bias_stereotypes.csv": "toxic",

    "redteam_illegal_activities.csv": "jailbreak",

    "redteam_sexual_content.csv": "toxic"
}


# =========================================
# LOAD REDTEAM FILES
# =========================================

all_rows = []

print("\n===== LOADING REDTEAM DATASETS =====\n")


for filename, label in CATEGORY_MAP.items():

    file_path = PROMPTS_DIR / filename

    if not file_path.exists():

        print(f"[WARNING] Missing file: {filename}")

        continue

    try:

        df = pd.read_csv(file_path)

        print(f"[LOADED] {filename} -> {len(df)} rows")

        # FIND PROMPT COLUMN
        possible_columns = [

            "prompt",
            "text",
            "input",
            "instruction"
        ]

        prompt_column = None

        for col in possible_columns:

            if col in df.columns:

                prompt_column = col

                break

        if prompt_column is None:

            print(f"[SKIPPED] No prompt column in {filename}")

            continue

        temp_df = pd.DataFrame({

            "text": df[prompt_column].astype(str),

            "label": label,

            "source": filename,

            "difficulty": "medium"
        })

        all_rows.append(temp_df)

    except Exception as e:

        print(f"[ERROR] {filename}")

        print(e)

# =========================================
# MERGE DATASETS
# =========================================

print("\n===== MERGING DATASETS =====\n")

merged_df = pd.concat(

    all_rows,

    ignore_index=True
)


# =========================================
# CLEANING
# =========================================

print("\n===== CLEANING DATA =====\n")

# REMOVE EMPTY
merged_df = merged_df.dropna(subset=["text"])

# STRIP WHITESPACE
merged_df["text"] = merged_df["text"].str.strip()

# REMOVE EMPTY STRINGS
merged_df = merged_df[

    merged_df["text"] != ""
]

# REMOVE DUPLICATES
before = len(merged_df)

merged_df = merged_df.drop_duplicates(

    subset=["text"]
)

after = len(merged_df)

print(f"Removed {before - after} duplicates")


# =========================================
# SHUFFLE
# =========================================

merged_df = merged_df.sample(

    frac=1,

    random_state=42
).reset_index(drop=True)


# =========================================
# SAVE
# =========================================

merged_df.to_csv(

    OUTPUT_FILE,

    index=False
)

print("\n===== DATASET BUILT =====\n")

print(f"Saved to: {OUTPUT_FILE}")

print(f"Total rows: {len(merged_df)}")


# =========================================
# LABEL DISTRIBUTION
# =========================================

print("\n===== LABEL DISTRIBUTION =====\n")

print(

    merged_df["label"].value_counts()
)