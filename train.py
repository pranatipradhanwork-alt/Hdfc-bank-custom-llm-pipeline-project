import os
import sys
import json
import subprocess
import mlflow
from dotenv import load_dotenv
import yaml
from pathlib import Path
import pandas as pd
import torch
from deltalake import DeltaTable
from datasets import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

#ENVIRONMENT SETTING AND  HARDWARE CONFIGURATION

print("[INFO] Setting up environment and hardware configuration...")
load_dotenv()
LOCAL_S3_VAULT=Path(os.getenv("LOCAL_S3_VAULT","data/s3_storage_vault/cleaned_banking_table"))
LORA_OUTPUT_DIR=Path(os.getenv("LORA_OUTPUT_DIR","models/hdfc_lora_adapter"))

#Staged runs: MAX_STEPS=5 (smoke test) -> MAX_STEPS=150 (~1 epoch) -> unset (full run from config epochs)
MAX_STEPS=int(os.getenv("MAX_STEPS","-1"))

#One seed for data split, LoRA init and trainer shuffling, so runs are reproducible
SEED=int(os.getenv("SEED","42"))
set_seed(SEED)

#MLflow experiment tracking: local SQLite store by default (view with: mlflow ui --backend-store-uri sqlite:///mlflow.db)
#Each run is named after its output folder, e.g. stage2_150 or hdfc_lora_v1
MLFLOW_TRACKING_URI=os.getenv("MLFLOW_TRACKING_URI","sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME=os.getenv("MLFLOW_EXPERIMENT_NAME","hdfc-bankfaq-lora")
MLFLOW_RUN_NAME=os.getenv("MLFLOW_RUN_NAME",LORA_OUTPUT_DIR.name)
os.environ["MLFLOW_TRACKING_URI"]=MLFLOW_TRACKING_URI
os.environ["MLFLOW_EXPERIMENT_NAME"]=MLFLOW_EXPERIMENT_NAME
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def current_git_commit():
    #Code version for the run manifest; "-dirty" means uncommitted changes were present
    try:
        commit=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
        dirty=subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True).strip()
        return commit+("-dirty" if dirty else "")
    except Exception:
        return "unknown"

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
    config_profile="configs/training/cpu-demo.yaml"
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

#Tables written before the Task column existed are FAQ-only
if "Task" not in df.columns:
    df["Task"]="faq"
print(f"[STATUS] Records per task: {df['Task'].value_counts().to_dict()}")

#splitting dataset into train / validation / test (80/10/10) per task with a fixed seed.
#Validation is scored during training; test is held out for the final score only.

splits={"train":[],"validation":[],"test":[]}
for task,group in df.groupby("Task"):
    group=group.sample(frac=1,random_state=SEED)
    n_train=int(len(group)*0.8)
    n_val=int(len(group)*0.1)
    splits["train"].append(group.iloc[:n_train])
    splits["validation"].append(group.iloc[n_train:n_train+n_val])
    splits["test"].append(group.iloc[n_train+n_val:])

#converting pandas dataframes to huggingface datasets (shuffled so tasks are mixed)

train_data,val_data,test_data=(
    Dataset.from_pandas(pd.concat(parts).sample(frac=1,random_state=SEED),preserve_index=False)
    for parts in splits.values()
)
print(f"[STATUS] Data partitions allocation: {len(train_data)} train / {len(val_data)} validation / {len(test_data)} test")

#MODEL INSTATIATION AND TOKENIZER SETUP

model_id=cfg["base_model_name"]
print(f"\n[INFO] Loading base model: {model_id}...")

#Initialize automated text-to-token matrix translator

tokenizer=AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
if tokenizer.pad_token is None:
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

SYSTEM_PROMPT=(
    "You are HDFC Bank's customer support assistant. Answer banking questions clearly "
    "and concisely. Never ask for or reveal OTPs, PINs, CVVs, passwords or full account numbers."
)
INTENT_INSTRUCTION="Classify the intent of this customer message. Reply with only the intent label.\n\nMessage: {query}"

#Prompt/completion chat format: SFTTrainer applies the model's chat template, computes loss on the
#completion (answer) only, and appends the EOS token so the model learns when to stop.
#Inference and promptfoo must build the prompt with the same system prompt and template.

def formatting_prompts(example):
    user_content=INTENT_INSTRUCTION.format(query=example["User_Query"]) if example["Task"]=="intent" else example["User_Query"]
    return {
        "prompt":[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":user_content},
        ],
        "completion":[{"role":"assistant","content":example["Target_Banking_Response"]}],
    }
train_cfg=cfg["training_arguments"]

#Mixed precision: bf16 where the GPU supports it, otherwise fp16 on CUDA. Never both at once.
use_bf16=device_target=="cuda" and torch.cuda.is_bf16_supported()
use_fp16=device_target=="cuda" and not use_bf16 and torch_precision==torch.float16

#Instantiate the centralized execution matrix agruments

training_arguments = SFTConfig(
    output_dir=str(LORA_OUTPUT_DIR),
    max_steps=MAX_STEPS,
    per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 2)),
    per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 2)),
    gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 4)),
    num_train_epochs=int(train_cfg.get("num_train_epochs", 1)),
    learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
    weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    warmup_steps=int(train_cfg.get("warmup_steps", 10)),
    logging_steps=int(train_cfg.get("logging_steps", 5)),
    save_strategy=train_cfg.get("save_strategy", "steps"),
    save_steps=int(train_cfg.get("save_steps", 50)),
    save_total_limit=train_cfg.get("save_total_limit", 1),
    eval_strategy="steps",
    eval_steps=int(train_cfg.get("eval_steps", 50)),
    load_best_model_at_end=False,
    metric_for_best_model="loss",
    greater_is_better=False,
    fp16=use_fp16,
    bf16=use_bf16,
    push_to_hub=False,
    max_length=512,
    completion_only_loss=True,
    seed=SEED,
    report_to="mlflow"
)

#Training engine: Combining the model, tokenizer, training arguments, and datasets into a single SFTTrainer instance
train_mapped = train_data.map(formatting_prompts, remove_columns=train_data.column_names)
val_mapped = val_data.map(formatting_prompts, remove_columns=val_data.column_names)
test_mapped = test_data.map(formatting_prompts, remove_columns=test_data.column_names)

trainer=SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_arguments,
    train_dataset=train_mapped,
    eval_dataset=val_mapped,
    peft_config=None
)

print(f"[LAUNCH] Beginning fine-tuning process with SFTTrainer on {device_target.upper()} device (max_steps={MAX_STEPS})...")

#Open the MLflow run ourselves so training, validation and test metrics plus artifacts all land in ONE run.
#The Trainer's MLflow callback reuses this active run and does not close it.
mlflow.start_run(run_name=MLFLOW_RUN_NAME)
mlflow.set_tags({
    "base_model":model_id,
    "config_profile":config_profile,
    "device":active_hardware,
    "git_commit":current_git_commit(),
    "dataset_path":str(LOCAL_S3_VAULT),
    "dataset_version":str(dt.version()),
    "tasks":",".join(sorted(df["Task"].unique())),
    "seed":str(SEED),
    "max_steps":str(MAX_STEPS),
    "output_dir":str(LORA_OUTPUT_DIR),
})
#Dataset lineage: records the exact Delta snapshot (with a content digest) used for training
mlflow.log_input(
    mlflow.data.from_pandas(df, source=str(LOCAL_S3_VAULT), name=f"cleaned_banking_table_v{dt.version()}"),
    context="training",
)
mlflow.log_artifact(config_profile, artifact_path="config")

try:
    train_result=trainer.train()

    print("\n[INFO] Evaluating on held-out test split...")
    test_metrics=trainer.evaluate(eval_dataset=test_mapped, metric_key_prefix="test")
    print(f"[STATUS] Test metrics: {test_metrics}")

    print(f"[INFO] Packaging and saving the fine-tuned LoRA adapter model to: {LORA_OUTPUT_DIR}...")
    trainer.model.save_pretrained(str(LORA_OUTPUT_DIR))
    tokenizer.save_pretrained(str(LORA_OUTPUT_DIR))

    #Keep the numbers next to the adapter so a finished run is never "loss unknown"
    metrics={
        "base_model":model_id,
        "config_profile":config_profile,
        "device":active_hardware,
        "seed":SEED,
        "max_steps":MAX_STEPS,
        "dataset_version":dt.version(),
        "rows":{"train":len(train_data),"validation":len(val_data),"test":len(test_data)},
        "rows_per_task":df["Task"].value_counts().to_dict(),
        "train_loss":train_result.training_loss,
        **test_metrics,
    }
    metrics["mlflow_run_id"]=mlflow.active_run().info.run_id
    with open(LORA_OUTPUT_DIR / "metrics.json","w") as f:
        json.dump(metrics,f,indent=2)

    #Attach the adapter itself to the run (tokenizer is left out: it comes unchanged from the base model)
    for artifact in ("adapter_config.json","adapter_model.safetensors","metrics.json"):
        mlflow.log_artifact(str(LORA_OUTPUT_DIR / artifact), artifact_path="adapter")
    mlflow.end_run(status="FINISHED")
    print(f"[SUCCESS] Fine-tuned LoRA adapter model and metrics.json saved successfully...")
    print(f"[STATUS] MLflow run '{MLFLOW_RUN_NAME}' ({metrics['mlflow_run_id']}) logged to experiment '{MLFLOW_EXPERIMENT_NAME}'")
except KeyboardInterrupt:
    mlflow.end_run(status="KILLED")
    print("[FAIL] Fine-tuning stopped by user (Ctrl+C). MLflow run marked KILLED.")
    raise
except Exception as e:
    mlflow.end_run(status="FAILED")
    print(f"[FAIL] Fine_tuning loop failed:{str(e)}")
    raise
