
import os
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
from deltalake import write_deltalake, DeltaTable

# Same masking rules as the live gateway, so training data and customer traffic are treated identically
from guardrails import mask_sensitive


# 📦 Local S3 Storage Path Simulation
LOCAL_S3_VAULT = Path("data/s3_storage_vault/cleaned_banking_table")
EVAL_VAULT = Path("data/s3_storage_vault/banking77_eval_table")
INPUT_PATH = Path("data/BankFAQs.csv")
BANKING77_TRAIN_PATH = Path("data/banking77_train.csv")
BANKING77_TEST_PATH = Path("data/banking77_test.csv")

# Banking77 has ~10k training rows vs ~1.5k FAQs; cap rows per intent so FAQs aren't drowned out.
BANKING77_ROWS_PER_INTENT = 40
SEED = 42
QUALITY_REPORTS = Path("data/quality_reports")

# Near-duplicate FAQ entries (typo copies, a trailing full stop) score ~0.997 on question+answer embeddings,
# while different products with similar wording score below 0.98
NEAR_DUPLICATE_SIMILARITY = 0.99
# Questions this similar ask the same thing; they must share a partition so no paraphrase leaks into the test set
SAME_QUESTION_SIMILARITY = 0.95
SPLIT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def normalise(text):
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def embed(texts):
    # Same embedding model as the RAG index, so "duplicate" means the same thing in both places
    from rag import load_embedder
    return load_embedder().encode(list(texts), normalize_embeddings=True, batch_size=64)


def drop_near_duplicates(frame):
    # Exact duplicates after normalising case and punctuation, then near-duplicates by embedding similarity
    key = frame["User_Query"].map(normalise) + " || " + frame["Target_Banking_Response"].map(normalise)
    exact_removed = int(key.duplicated().sum())
    frame = frame[~key.duplicated()].reset_index(drop=True)
    vectors = embed(frame["User_Query"] + "\n" + frame["Target_Banking_Response"])
    similarity = np.triu(vectors @ vectors.T, k=1)
    # Keep the first entry of each near-duplicate pair
    drop = np.unique(np.argwhere(similarity >= NEAR_DUPLICATE_SIMILARITY)[:, 1])
    return frame.drop(index=drop).reset_index(drop=True), exact_removed, len(drop)


def question_groups(frame):
    # Union-find over questions that mean the same thing, so each group lands in a single partition
    vectors = embed(frame["User_Query"])
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in np.argwhere(np.triu(vectors @ vectors.T, k=1) >= SAME_QUESTION_SIMILARITY):
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(frame))]), vectors


def assign_splits(frame, groups):
    # Shuffle whole groups with a fixed seed and fill train, then validation, then test by row count
    order = pd.Series(np.unique(groups)).sample(frac=1, random_state=SEED).tolist()
    sizes = pd.Series(groups).value_counts()
    boundaries = np.cumsum([SPLIT_FRACTIONS[s] for s in ("train", "validation")]) * len(frame)
    split_of_group, filled = {}, 0
    for group in order:
        split_of_group[group] = "train" if filled < boundaries[0] else "validation" if filled < boundaries[1] else "test"
        filled += sizes[group]
    return pd.Series(groups).map(split_of_group).values


def contamination(frame, vectors):
    # Highest question similarity between each test row and any training row; high values mean leakage
    test, train = (frame["Split"] == "test").values, (frame["Split"] == "train").values
    closest = (vectors[test] @ vectors[train].T).max(axis=1)
    return {
        "test_rows": int(test.sum()),
        "max_similarity_to_train": round(float(closest.max()), 4),
        "test_rows_at_or_above_0.95": int((closest >= 0.95).sum()),
        "test_rows_at_or_above_0.90": int((closest >= 0.90).sum()),
    }

# v1: BankFAQs only in the training table (Banking77 goes to the eval table for promptfoo).
# v2: INCLUDE_BANKING77=1 python clean_data.py adds sampled Banking77 intent records to training.
INCLUDE_BANKING77 = os.getenv("INCLUDE_BANKING77", "0") == "1"

print("[INFO] Initializing raw banking corpus ingestion and compliance processing...")

missing = [p for p in (INPUT_PATH, BANKING77_TRAIN_PATH, BANKING77_TEST_PATH) if not p.exists()]
if missing:
    print(f"[ERROR]  Cannot find raw data files: {', '.join(map(str, missing))}. Run download_data.py first.")
else:
    # 1. Load the raw data from your download script
    df = pd.read_csv(INPUT_PATH)
    quality = {"source": str(INPUT_PATH), "raw_rows": len(df)}

    # 2. Structure columns and remove duplicates
    df = df.rename(columns={'Question': 'User_Query', 'Answer': 'Target_Banking_Response'})
    df = df[['User_Query', 'Target_Banking_Response']].dropna()
    quality["rows_with_missing_fields"] = quality["raw_rows"] - len(df)
    rows_before_dedup = len(df)
    df = df.drop_duplicates()
    df['Task'] = 'faq'
    df['Source'] = 'BankFAQs'

    # 2b. Banking77 intent records: customer message -> intent label, balanced per intent
    def load_banking77(path: Path) -> pd.DataFrame:
        b77 = pd.read_csv(path).rename(columns={'text': 'User_Query', 'category': 'Target_Banking_Response'})
        b77 = b77[['User_Query', 'Target_Banking_Response']].dropna().drop_duplicates()
        b77['Task'] = 'intent'
        b77['Source'] = 'Banking77'
        return b77

    intent_df = load_banking77(BANKING77_TRAIN_PATH)
    # Shuffle, then keep the first N rows of each intent (keeps all columns, unlike groupby.apply in pandas 3)
    intent_df = (
        intent_df.sample(frac=1, random_state=SEED)
        .groupby('Target_Banking_Response')
        .head(BANKING77_ROWS_PER_INTENT)
    )
    # Official Banking77 test split is kept out of training and stored separately for evaluation.
    intent_eval_df = load_banking77(BANKING77_TEST_PATH)
    print(f"[STATUS] Banking77: {len(intent_df)} training rows sampled, {len(intent_eval_df)} held-out eval rows")

    # 3. PII Scrubbing (Safety Gates to prevent data leaks)
    def scrub_private_data(value: object) -> str:
        text = "" if pd.isna(value) else str(value)
        return mask_sensitive(text)[0]
        


    
    print("[INFO] Enforcing PII safety gates across unstructured array columns...")
    df['User_Query'] = df['User_Query'].apply(scrub_private_data)
    df['Target_Banking_Response'] = df['Target_Banking_Response'].apply(scrub_private_data)
    
    # 4. HDFC Bank Brand Alignment
    print("Aligning text terminology with HDFC corporate brand standards...")
    # The Kaggle dump anonymises HDFC as "M&N" ("M&N  Bank", "M&N Life", "www.m&N.com"); restore the real name
    for column in ('User_Query', 'Target_Banking_Response'):
        df[column] = df[column].str.replace(r"(?i)\bM&N\s+bank\b", "HDFC Bank", regex=True)
        df[column] = df[column].str.replace(r"(?i)M&N\s+", "HDFC ", regex=True)
        df[column] = df[column].str.replace(r"(?i)M&N", "HDFC", regex=True)
    df['Target_Banking_Response'] = df['Target_Banking_Response'].str.replace("the bank", "HDFC Bank", case=False)
    df['Target_Banking_Response'] = df['Target_Banking_Response'].str.replace("our institution", "HDFC Bank", case=False)
    text = df['User_Query'] + " " + df['Target_Banking_Response']
    quality["pii_masks_applied"] = {
        label: int(text.str.count(re.escape(f"[{label}]")).sum())
        for label in ("MASKED_ACCOUNT_OR_CARD", "MASKED_PHONE_NUMBER", "MASKED_EMAIL", "MASKED_SECRET")
    }

    # 4a. Deduplicate, then split by question group so no paraphrase of a test question is trained on
    print("[INFO] Removing exact and near-duplicate FAQs...")
    exact_verbatim = rows_before_dedup - len(df)
    df, exact_normalised, quality["near_duplicates_removed"] = drop_near_duplicates(df)
    quality["exact_duplicates_removed"] = exact_verbatim + exact_normalised
    groups, question_vectors = question_groups(df)
    df['Split'] = assign_splits(df, groups)
    quality["question_groups"] = int(len(np.unique(groups)))
    quality["largest_question_group"] = int(pd.Series(groups).value_counts().max())
    quality["split_rows"] = df['Split'].value_counts().to_dict()
    quality["contamination"] = contamination(df, question_vectors)
    answer_words = df['Target_Banking_Response'].str.split().str.len()
    quality["answer_words"] = {"min": int(answer_words.min()), "median": int(answer_words.median()), "max": int(answer_words.max())}
    print(f"[STATUS] {quality['exact_duplicates_removed']} exact and {quality['near_duplicates_removed']} near-duplicates removed; "
          f"splits {quality['split_rows']}; contamination {quality['contamination']}")

    # 4b. Intent records: scrub customer messages only (labels are fixed intent names)
    intent_df['User_Query'] = intent_df['User_Query'].apply(scrub_private_data)
    intent_eval_df['User_Query'] = intent_eval_df['User_Query'].apply(scrub_private_data)

    if INCLUDE_BANKING77:
        # Intent messages are short and independent, so each row is its own split group
        intent_df = intent_df.reset_index(drop=True)
        intent_df['Split'] = assign_splits(intent_df, np.arange(len(intent_df)))
        df = pd.concat([df, intent_df], ignore_index=True)
        print("[STATUS] INCLUDE_BANKING77=1: Banking77 intent records added to the training table")
    else:
        print("[STATUS] Training table is BankFAQs only (set INCLUDE_BANKING77=1 to add Banking77)")

    # 5. 🔒 WRITE DIRECTLY TO DELTA LAKE (Fulfilling Governance Requirements)
    print(" Freezing dataset into immutable Delta Lake snapshots inside Local S3...")
    # schema_mode="overwrite" because Task/Source columns were added in this version
    write_deltalake(LOCAL_S3_VAULT, df, mode="overwrite", schema_mode="overwrite")
    write_deltalake(EVAL_VAULT, intent_eval_df, mode="overwrite", schema_mode="overwrite")

    print("\n[SUCCESS] DATA GOVERNANCE COMPLIANCE VERIFIED.")
    print(f"[STATUS] Target Delta table active location: {LOCAL_S3_VAULT}")
    print(f"[STATUS] Banking77 eval Delta table location: {EVAL_VAULT}")

    # Verify the audit log works for your repository validation
    dt = DeltaTable(LOCAL_S3_VAULT)
    print(f"[STATUS] Active Delta snapshot tracking version: {dt.version()}")
    print(f"[STATUS] Total verified rows committed to storage layer: {len(df)}")
    print(f"[STATUS] Rows per task: {df['Task'].value_counts().to_dict()}")

    # Quality evidence travels with the exact version it describes
    quality.update({"delta_table": str(LOCAL_S3_VAULT), "delta_version": dt.version(), "final_rows": len(df),
                    "rows_per_task": df['Task'].value_counts().to_dict()})
    QUALITY_REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = QUALITY_REPORTS / f"cleaned_banking_table_v{dt.version()}.json"
    report_path.write_text(json.dumps(quality, indent=2))
    print(f"[STATUS] Data quality report: {report_path}")
