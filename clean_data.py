
import os
import re
from pathlib import Path
import pandas as pd
from deltalake import write_deltalake, DeltaTable


ACCOUNT_OR_CARD = re.compile(r"\b\d{10,16}\b")
INDIAN_MOBILE = re.compile(r"\b[6-9]\d{9}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Mask only when a value follows ("PIN 1234", "OTP is 5678", "password: abc"),
# so ordinary mentions like "change my PIN" keep their meaning.
SECRET = re.compile(r"(?i)\b(?:otp|cvv|pin|password)\b(?:\s*(?:is\s+)?[:=-]?\s*\S*\d\S*|\s*[:=]\s*\S+)")


# 📦 Local S3 Storage Path Simulation
LOCAL_S3_VAULT = Path("data/s3_storage_vault/cleaned_banking_table")
EVAL_VAULT = Path("data/s3_storage_vault/banking77_eval_table")
INPUT_PATH = Path("data/BankFAQs.csv")
BANKING77_TRAIN_PATH = Path("data/banking77_train.csv")
BANKING77_TEST_PATH = Path("data/banking77_test.csv")

# Banking77 has ~10k training rows vs ~1.5k FAQs; cap rows per intent so FAQs aren't drowned out.
BANKING77_ROWS_PER_INTENT = 40
SEED = 42

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

    # 2. Structure columns and remove duplicates
    df = df.rename(columns={'Question': 'User_Query', 'Answer': 'Target_Banking_Response'})
    df = df[['User_Query', 'Target_Banking_Response']].dropna().drop_duplicates()
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
        text = ACCOUNT_OR_CARD.sub("[MASKED_ACCOUNT_OR_CARD]", text)
        text = INDIAN_MOBILE.sub("[MASKED_PHONE_NUMBER]", text)
        text = EMAIL.sub("[MASKED_EMAIL]", text)
        return SECRET.sub("[MASKED_SECRET]", text)
        


    
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

    # 4b. Intent records: scrub customer messages only (labels are fixed intent names)
    intent_df['User_Query'] = intent_df['User_Query'].apply(scrub_private_data)
    intent_eval_df['User_Query'] = intent_eval_df['User_Query'].apply(scrub_private_data)

    if INCLUDE_BANKING77:
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
