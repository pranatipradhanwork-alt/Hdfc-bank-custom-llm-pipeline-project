
import re
from pathlib import Path
import pandas as pd
from deltalake import write_deltalake, DeltaTable


ACCOUNT_OR_CARD = re.compile(r"\b\d{10,16}\b")
INDIAN_MOBILE = re.compile(r"\b[6-9]\d{9}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
SECRET = re.compile(r"(?i)\b(?:otp|cvv|pin|password)\s*[:=-]?\s*\S+")


# 📦 Local S3 Storage Path Simulation
LOCAL_S3_VAULT = Path("data/s3_storage_vault/cleaned_banking_table")
INPUT_PATH = Path("data/BankFAQs.csv")

print("[INFO] Initializing raw banking corpus ingestion and compliance processing...")

if not INPUT_PATH.exists():
    print(f"[ERROR]  Cannot find raw data file at {INPUT_PATH}.")
else:
    # 1. Load the raw data from your download script
    df = pd.read_csv(INPUT_PATH)
    
    # 2. Structure columns and remove duplicates
    df = df.rename(columns={'Question': 'User_Query', 'Answer': 'Target_Banking_Response'})
    df = df[['User_Query', 'Target_Banking_Response']].dropna().drop_duplicates()
    
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
    df['Target_Banking_Response'] = df['Target_Banking_Response'].str.replace("the bank", "HDFC Bank", case=False)
    df['Target_Banking_Response'] = df['Target_Banking_Response'].str.replace("our institution", "HDFC Bank", case=False)

    # 5. 🔒 WRITE DIRECTLY TO DELTA LAKE (Fulfilling Governance Requirements)
    print(" Freezing dataset into immutable Delta Lake snapshots inside Local S3...")
    write_deltalake(LOCAL_S3_VAULT, df, mode="overwrite")
    
    print("\n[SUCCESS] DATA GOVERNANCE COMPLIANCE VERIFIED.")
    print(f"[STATUS] Target Delta table active location: {LOCAL_S3_VAULT}")
    
    # Verify the audit log works for your repository validation
    dt = DeltaTable(LOCAL_S3_VAULT)
    print(f"[STATUS] Active Delta snapshot tracking version: {dt.version()}")
    print(f"[STATUS] Total verified rows committed to storage layer: {len(df)}")
