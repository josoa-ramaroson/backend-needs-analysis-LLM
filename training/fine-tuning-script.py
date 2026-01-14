# ------------- Library import -------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Third party service authentification
from huggingface_hub import login
login(token="<HF_TOKEN>")  

import wandb
wandb.login(key="<WAN_DB_TOKEN>")
# PROJECT name- Fine-tune-LLAMA3.2-Instruct
run = wandb.init(
    project='Fine-tune-LLAMA3.2-Instruct',
    job_type="training",
    anonymous="allow"
)

from datasets import Dataset
import json
import random

data_path = "<DATABASE_PATH_ON_KAGGLE>"
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)
# Shuffle
random.seed(45)
random.shuffle(data)

# Create Hugging Face Dataset
dataset = Dataset.from_list(data)


model_id = "meta-llama/Llama-3.2-3B-Instruct"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# enforce right padding (recommended for training in fp16)
tokenizer.padding_side = "right"

# load model WITHOUT bitsandbytes/quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,   # use torch_dtype param name
)

# prepare for peft
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=162,
    target_modules=['q_proj','k_proj','v_proj','o_proj','up_proj','down_proj'],
    lora_dropout=0.05,
    bias="none",
    3#
    task_type="CAUSAL_LM"
)


# -------------- training args --------------
training_args = TrainingArguments(
    output_dir="./lora_no_bnb",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
     gradient_checkpointing=True,
    fp16=True,
    bf16=False,
    optim="adamw_torch",
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    remove_unused_columns=True,   # set to True to avoid collator issues
    report_to="none",
)

# -------------- trainer: pass dataset_text_field and max_seq_length --------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
    processing_class=tokenizer
)

trainer.train()