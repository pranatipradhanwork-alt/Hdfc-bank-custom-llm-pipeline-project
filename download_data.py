from pathlib import Path
import shutil
import kagglehub
import requests

print("[INFO] Downloading BankFAQs dataset...")

cache_path = Path(
    kagglehub.dataset_download("somanathkshirasagar/bankfaqs")
)

destination = Path("data/BankFAQs.csv")
destination.parent.mkdir(parents=True, exist_ok=True)

matches = list(cache_path.rglob("*.csv"))

if not matches:
    raise FileNotFoundError(
        f"[ERROR] No CSV file found inside: {cache_path}"
    )

source = matches[0]
shutil.copy2(source, destination)

print(f"[SUCCESS] Dataset saved to: {destination}")

# Banking77 (PolyAI, CC BY 4.0): customer messages labelled with 77 intents.
# The official train split is used for fine-tuning, the official test split only for evaluation.
BANKING77_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{split}.csv"

print("[INFO] Downloading Banking77 dataset...")
for split in ("train", "test"):
    response = requests.get(BANKING77_URL.format(split=split), timeout=60)
    response.raise_for_status()
    split_destination = Path(f"data/banking77_{split}.csv")
    split_destination.write_bytes(response.content)
    print(f"[SUCCESS] Banking77 {split} split saved to: {split_destination}")
