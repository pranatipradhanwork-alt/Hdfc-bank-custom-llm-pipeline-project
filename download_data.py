from pathlib import Path
import shutil
import kagglehub

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