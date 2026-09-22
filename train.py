import os
import sys
from dotenv import load_dotenv
import yaml
from pathlib import Path
import torch
from deltalake import DeltaTable
from datasets import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling,DataCollatorForSeq2Seq

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training 
from trl import SFTTrainer

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

if cfg["use_quantization"] and device_target == "cuda":
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
        dtype=torch_precision
        )
    model=prepare_model_for_kbit_training(model)
    active_hardware = "CUDA (Quantized)"
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        #device_map="auto",  
        low_cpu_mem_usage=True,  
        dtype=torch_precision
    )

    if torch.backends.mps.is_available():
        model = model.to("mps")
        active_hardware = "MPS"
    else:
        active_hardware = "CPU"

print(f"[SUCCESS]Core trmsformer model loaded successfully on target device engine: {active_hardware}.")


#MODEL FINETUNING WITH MODULAR ADAPTERS(PEFT/LORA)

print("\n[INFO] Configuring LoRA adapters for parameter-efficient fine-tuning...")
lora_cfg=cfg["lora"]

peft_config=LoraConfig(
    
    task_type="CAUSAL_LM",
    r=lora_cfg["r"],
    lora_alpha=lora_cfg["lora_alpha"],
    lora_dropout=lora_cfg["lora_dropout"],
    target_modules=lora_cfg["target_modules"],
    bias="none"
)

#Attach lora adapters to the base model

model=get_peft_model(model, peft_config)
trainable_count,total_count = model.get_nb_trainable_parameters()
print(f"[SUCCESS] LoRA adapters attached to the base model. Total trainable parameters: {trainable_count:,}")


#SFTTRAINER PIPELINE SETUP WITH MLFLOW LINKS

print("\n[INFO] Setting up SFTTrainer pipeline for supervised fine-tuning...")

def formatting_prompts(example):
    output_texts=[]
    for i in range(len(example["User_Query"])):
        text = f"Context: Standard HDFC protocol apply.\nUser Query: {example['User_Query'][i]}\nHDFC Authorized Support Response: {example['Target_Banking_Response'][i]}{tokenizer.eos_token}"
        output_texts.append(text)
    return {"text": output_texts}
train_cfg=cfg["training_arguments"]

#Instantiate the centralized execution matrix agruments

training_arguments = TrainingArguments(
    output_dir=str(LORA_OUTPUT_DIR),
    max_steps=5,
    per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
    per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
    gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
    num_train_epochs=train_cfg.get("num_train_epochs", 1),
    learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
    warmup_steps=train_cfg.get("warmup_steps", 10),
    logging_steps=train_cfg.get("logging_steps", 5),
    save_strategy=train_cfg.get("save_strategy", "steps"),
    save_total_limit=train_cfg.get("save_total_limit", 1),
    eval_strategy="no",
    eval_steps=train_cfg.get("eval_steps", 50),
    load_best_model_at_end=False,
    metric_for_best_model="loss",
    greater_is_better=False,
    fp16=(torch_precision == torch.float16) if device_target == "cuda" else False,
    push_to_hub=False,
    bf16=True if device_target == "cuda" and torch.cuda.is_bf16_supported() else False,
    report_to="mlflow"
)

#Training engine: Combining the model, tokenizer, training arguments, and datasets into a single SFTTrainer instance
train_mapped = train_data.map(formatting_prompts, batched=True, remove_columns=train_data.column_names)
test_mapped = test_data.map(formatting_prompts, batched=True, remove_columns=test_data.column_names)

trainer=SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_arguments,
    train_dataset=train_mapped,
    eval_dataset=test_mapped,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model ,padding=True),
    peft_config=None
)

print(f"[LAUNCH] Beginning fine-tuning process with SFTTrainer on {device_target.upper()} device...")

try:
    trainer.train()

    print(f"[INFO] Packaging and saving the fine-tuned LoRA adapter model to: {LORA_OUTPUT_DIR}...")
    trainer.model.save_pretrained(str(LORA_OUTPUT_DIR))
    tokenizer.save_pretrained(str(LORA_OUTPUT_DIR))
    print(f"[SUCCESS] Fine-tuned LoRA adapter model saved successfully...")
except Exception as e:
    print(f"[FAIL] Fine_tuning loop failed:{str(e)}")





