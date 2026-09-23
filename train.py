import os
import sys
from dotenv import load_dotenv
import yaml
from pathlib import Path
import torch
from deltalake import DeltaTable
from datasets import Dataset
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ENVIRONMENT SETTING AND HARDWARE CONFIGURATION

print("[INFO] Setting up environment and hardware configuration...")
load_dotenv()
LOCAL_S3_VAULT = Path(os.getenv("LOCAL_S3_VAULT", "data/s3_storage_vault/cleaned_banking_table"))
LORA_OUTPUT_DIR = Path(os.getenv("LORA_OUTPUT_DIR", "models/hdfc_lora_adapter"))

# Optional quick sanity-check mode: QUICK_TEST=1 python train.py
# Trains on a handful of rows for a couple of steps so you can see the
# pipeline run end to end on CPU without waiting hours. Unset for a real run.
QUICK_TEST = os.getenv("QUICK_TEST", "0") == "1"

# Auto-detect GPU availability and set device accordingly

if torch.backends.mps.is_available():
    config_profile = "configs/training/mac-lora.yaml"
    device_target = "mps"
    print("[STATUS] MPS (Apple Silicon GPU) detected. Using mac-lora.yaml configuration profile.")
elif torch.cuda.is_available():
    config_profile = "configs/training/cuda-qlora.yaml"
    device_target = "cuda"
    print("[STATUS] CUDA (NVIDIA GPU) detected. Using cuda-qlora.yaml configuration profile.")
else:
    config_profile = "configs/training/cpu-demo.yaml"
    device_target = "cpu"
    print("[STATUS] No GPU detected. Using CPU configuration profile.")

if not Path(config_profile).exists():
    raise FileNotFoundError(f"[ERROR] Configuration profile not found at: {config_profile}")
with open(config_profile, "r") as f:
    cfg = yaml.safe_load(f)

# LOCAL AUDITED DATA EXTRACTION & SPLIT

print("\n[INFO] Extracting compliance records out of Delta Lake data architecture...")
if not LOCAL_S3_VAULT.exists():
    raise FileNotFoundError(
        f"[ERROR] Local S3 vault registry not found at: {LOCAL_S3_VAULT}. "
        "Run download_data.py then clean_data.py first."
    )

# Establish connection to version-controlled data engine

dt = DeltaTable(str(LOCAL_S3_VAULT))
df = dt.to_pandas()
print(f"[STATUS] Delta table connected. Current dataset version: {dt.version()}. ")
print(f"[STATUS] Total records extracted: {len(df)}")

# converting pandas dataframe to huggingface dataset

raw_dataset = Dataset.from_pandas(df, preserve_index=False)

# splitting dataset into training and testing sets (80% train, 20% test) with a fixed random seed for reproducibility

split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
train_data = split_dataset["train"]
test_data = split_dataset["test"]

if QUICK_TEST:
    train_data = train_data.select(range(min(20, len(train_data))))
    test_data = test_data.select(range(min(10, len(test_data))))
    print("[STATUS] QUICK_TEST=1: subsetting data for a fast sanity-check run.")

print(f"[STATUS] Data partitions allocation: {len(train_data)} rows training / {len(test_data)} rows testing")

# MODEL INSTANTIATION AND TOKENIZER SETUP

model_id = cfg["base_model_name"]
print(f"\n[INFO] Loading base model: {model_id}...")

# Initialize automated text-to-token matrix translator

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Select mathematically float calculation scale dynamic

torch_precision = torch.float32 if cfg["torch_dtype"] == "float32" else torch.float16

# Apply the infrastructure routing control rules

active_hardware = "CPU"

if cfg["use_quantization"]:
    from transformers import BitsAndBytesConfig
    q_cfg = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_cfg["load_in_4bit"],
        bnb_4bit_use_double_quant=q_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=q_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch_precision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        dtype=torch_precision,
    )
    model = prepare_model_for_kbit_training(model)
    active_hardware = device_target.upper()
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        low_cpu_mem_usage=True,
        dtype=torch_precision,
    )

    if torch.backends.mps.is_available():
        model = model.to("mps")
        active_hardware = "MPS"
    else:
        active_hardware = "CPU"

print(f"[SUCCESS] Core transformer model loaded successfully on target device engine: {active_hardware}.")

# LORA ADAPTER SETUP

print("\n[INFO] Attaching LoRA adapters...")
peft_cfg = cfg["peft"]
lora_config = LoraConfig(
    r=peft_cfg["r"],
    lora_alpha=peft_cfg["lora_alpha"],
    target_modules=peft_cfg["target_modules"],
    lora_dropout=peft_cfg["lora_dropout"],
    bias=peft_cfg["bias"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# PROMPT FORMATTING AND TOKENIZATION

PROMPT_TEMPLATE = "### Customer Query:\n{query}\n\n### HDFC Bank Response:\n{response}"
MAX_LENGTH = 256


def format_and_tokenize(example):
    text = PROMPT_TEMPLATE.format(
        query=example["User_Query"],
        response=example["Target_Banking_Response"],
    ) + tokenizer.eos_token
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    return tokenized


print("\n[INFO] Tokenizing training and evaluation splits...")
tokenized_train = train_data.map(format_and_tokenize, remove_columns=train_data.column_names)
tokenized_test = test_data.map(format_and_tokenize, remove_columns=test_data.column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# TRAINING ARGUMENTS AND TRAINER

train_args_cfg = dict(cfg["training_arguments"])
# YAML quirk: "2e-4" (no decimal point) parses as a string, not a float,
# which crashes the optimizer's lr check. Coerce the numeric fields explicitly.
train_args_cfg["learning_rate"] = float(train_args_cfg["learning_rate"])
train_args_cfg["weight_decay"] = float(train_args_cfg["weight_decay"])
train_args_cfg["per_device_train_batch_size"] = int(train_args_cfg["per_device_train_batch_size"])
train_args_cfg["gradient_accumulation_steps"] = int(train_args_cfg["gradient_accumulation_steps"])
train_args_cfg["num_train_epochs"] = int(train_args_cfg["num_train_epochs"])
train_args_cfg["logging_steps"] = int(train_args_cfg["logging_steps"])
if QUICK_TEST:
    train_args_cfg["num_train_epochs"] = 1
    train_args_cfg["max_steps"] = 5
    train_args_cfg["logging_steps"] = 1

LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(LORA_OUTPUT_DIR),
    per_device_train_batch_size=train_args_cfg["per_device_train_batch_size"],
    gradient_accumulation_steps=train_args_cfg["gradient_accumulation_steps"],
    learning_rate=train_args_cfg["learning_rate"],
    num_train_epochs=train_args_cfg["num_train_epochs"],
    weight_decay=train_args_cfg["weight_decay"],
    logging_steps=train_args_cfg["logging_steps"],
    max_steps=train_args_cfg.get("max_steps", -1),
    eval_strategy="epoch",
    save_strategy="no",
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
)

print("\n[INFO] Starting fine-tuning run...")
train_result = trainer.train()
print(f"\n[SUCCESS] Training complete. Final training loss: {train_result.training_loss:.4f}")

print("\n[INFO] Evaluating on held-out test split...")
eval_metrics = trainer.evaluate()
print(f"[STATUS] Eval metrics: {eval_metrics}")

# SAVE THE ADAPTER

print(f"\n[INFO] Saving LoRA adapter to {LORA_OUTPUT_DIR}...")
model.save_pretrained(str(LORA_OUTPUT_DIR))
tokenizer.save_pretrained(str(LORA_OUTPUT_DIR))
print(f"[SUCCESS] Adapter and tokenizer saved to: {LORA_OUTPUT_DIR}")
