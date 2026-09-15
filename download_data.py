import os
import shutil
import kagglehub

print("📥 Fetching the raw BankFAQs dataset from Kaggle...")

# 1. Download the latest version into the hidden local cache folder
cache_path = kagglehub.dataset_download("somanathkshirasagar/bankfaqs")
print(f"📍 Hidden Cache Location: {cache_path}")

# 2. Define your project's local data directory path
destination_dir = "data"
destination_file = os.path.join(destination_dir, "BankFAQs.csv")

# Create the data folder automatically if it doesn't exist yet
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# 3. Physically pull the CSV file out of the hidden cache and save it locally
source_file = os.path.join(cache_path, "BankFAQs.csv")

if os.path.exists(source_file):
    shutil.copy(source_file, destination_file)
    print(f"✅ Success! Raw dataset copied straight to: {destination_file}")
    print(f"📊 Workspace Layout: data/BankFAQs.csv is ready.")
else:
    print("⚠️ Error: BankFAQs.csv was not found inside the cache path.")
