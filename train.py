import os
import sys
from dotenv import load_dotenv
import yaml
from pathlib import Path
import torch
from deltalake import DeltaTable
from datasets import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training 

#ENVIRONMENT SETTING AND  HARDWARE CONFIGURATION

print("[INFO] Setting up environment and hardware configuration...")
load_dotenv()
LOCAL_S3_VAULT=Path(os.getenv("LOCAL_S3_VAULT","data/s3_storage_vault/cleaned_banking_table"))
LORA_OUTPUT_DIR=Path(os.getenv("LORA_OUTPUT_DIR","models/hdfc_lora_adapter"))

#Auto-detect GPU availability and set device accordingly

if torch.backends.mps.is_available():
    config_profile="configs/training/mac-lora.yaml"
    device_target="mps"
    print("[STATUS] MPS (Apple Silicon GPU) detected. Using mac-lora.yaml configuration profile.")
elif torch.cuda.is_available():
    config_profile="configs/training/cuda-qlora.yaml"
    device_target="cuda"
    print("[STATUS] CUDA (NVIDIA GPU) detected. Using cuda-qlora.yaml configuration profile.")
else:
    config_profile="configs/training/cpu-lora.yaml"
    device_target="cpu"
    print("[STATUS] No GPU detected. Using CPU configuration profile.")

if not Path(config_profile).exists():
    raise FileNotFoundError(f"[ERROR] Configuration profile not found at: {config_profile}")
with open(config_profile, "r") as f:
    cfg=yaml.safe_load(f)


#LOCAL AUDITED DATA EXTRACTION & SPLIT

print("\n[INFO] Extracting compliance records out of Delta Lake data architecture...")
if not LOCAL_S3_VAULT.exists():
    raise FileNotFoundError(f"[ERROR] Local S3 vault registry not found at: {LOCAL_S3_VAULT}")


#Establish connection to vesion-controllled data engine

dt=DeltaTable(str(LOCAL_S3_VAULT))
df=dt.to_pandas()
print(f"[STATUS] Delta table connected. Current dataset version: {dt.version()}. ")
print(f"[STATUS] Total records extracted: {len(df)}")

#converting pandas dataframe to huggingface dataset

raw_dataset=Dataset.from_pandas(df, preserve_index=False)

#splitting dataset into training and testing sets (80% train, 20% test) with a fixed random seed for reproducibility

split_dataset=raw_dataset.train_test_split(test_size=0.2,seed=42)
train_data = split_dataset["train"]
test_data = split_dataset["test"]
print(f"[STATUS] Data partitions allocation: {len(train_data)} rows training / {len(test_data)} rows testing")

#MODEL INSTATIATION AND TOKENIZER SETUP

model_id=cfg["base_model_name"]
print(f"\n[INFO] Loading base model: {model_id}...")

#Initialize automated text-to-token matrix translator

tokenizer=AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side="right"

#Select mathematically float calculation scale dynamic

torch_precision = torch.float32 if cfg["torch_dtype"] == "float32" else torch.float16

#Apply the infrastructure routing control rules

if cfg["use_quantization"]:
    from transformers import BitsAndBytesConfig
    q_cfg=cfg["quantization"]
    bnb_config=BitsAndBytesConfig(
        load_in_4bit=q_cfg["load_in_4bit"],
        bnb_4bit_use_double_quant=q_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=q_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch_precision
    )
    model=AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch_precision
        )
    model=prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="cpu",  
        low_cpu_mem_usage=True,  
        torch_dtype=torch_precision
    )

    if torch.backends.mps.is_available():
        model = model.to("mps")
        active_hardware = "MPS"
    else:
        active_hardware = "CPU"

print(f"[SUCCESS]Core trmsformer model loaded successfully on target device engine: {active_hardware}.")








