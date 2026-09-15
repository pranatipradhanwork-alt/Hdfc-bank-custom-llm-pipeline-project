import os
import re
import pandas as pd
from deltalake import write_deltalake, DeltaTable

# 📦 Local S3 Storage Path Simulation
LOCAL_S3_VAULT = "data/s3_storage_vault/cleaned_banking_table"
INPUT_PATH = "data/BankFAQs.csv"

print("🚀 Launching data cleaning and Local Delta Lake ingestion pipeline...")

if not os.path.exists(INPUT_PATH):
    print(f"❌ Error: Cannot find raw data file at {INPUT_PATH}.")
else:
    # 1. Load the raw data from your download script
    df = pd.read_csv(INPUT_PATH)
    
    # 2. Structure columns and remove duplicates
    df = df.rename(columns={'Question': 'User_Query', 'Answer': 'Target_Banking_Response'})
    df = df[['User_Query', 'Target_Banking_Response']].dropna().drop_duplicates()
    
    # 3. PII Scrubbing (Safety Gates to prevent data leaks)
    def scrub_private_data(text):
        if not isinstance(text, str): return ""
        # Mask account or credit card number patterns (10 to 16 digits)
        text = re.sub(r'\b\d{10,16}\b', '[MASKED_ACCOUNT_OR_CARD]', text)
        # Mask Indian mobile formats (10 digits starting with 6-9)
        text = re.sub(r'\b[6-9]\d{9}\b', '[MASKED_PHONE_NUMBER]', text)
        return text

    print("🛡️ Applying PII safety gates...")
    df['User_Query'] = df['User_Query'].apply(scrub_private_data)
    df['Target_Banking_Response'] = df['Target_Banking_Response'].apply(scrub_private_data)
    
    # 4. HDFC Bank Brand Alignment
    print("🏦 Aligning text terminology with HDFC corporate brand standards...")
    df['Target_Banking_Response'] = df['Target_Banking_Response'].str.replace("the bank", "HDFC Bank", case=False)
    df['Target_Banking_Response'] = df['Target_Banking_Response'].str.replace("our institution", "HDFC Bank", case=False)

    # 5. 🔒 WRITE DIRECTLY TO DELTA LAKE (Fulfilling Governance Requirements)
    print("🧱 Freezing dataset into immutable Delta Lake snapshots inside Local S3...")
    write_deltalake(LOCAL_S3_VAULT, df, mode="overwrite")
    
    print("\n✅ SUCCESS! DATA GOVERNANCE COMPLIANT.")
    print(f"📁 Delta Table active at: {LOCAL_S3_VAULT}")
    
    # Verify the audit log works for your repository validation
    dt = DeltaTable(LOCAL_S3_VAULT)
    print(f"📜 Current Delta Table Version: {dt.version()}")
    print(f"📊 Total immutable data rows logged: {len(df)}")
