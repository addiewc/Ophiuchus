"""
Fine-tuning adopted from tutorial at: https://www.datacamp.com/tutorial/fine-tuning-llama-2.
"""

import sys
import os
import argparse
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

sys.path.append(os.path.join(sys.path[0], "../"))
from config import HUGGINGFACE_ACCESS_TOKEN


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, help="Huggingface model name to start pretraining from.")
parser.add_argument("--dataset", type=str, help="Path to dataset for finetuning")
parser.add_argument("-n", "--name", type=str, help="Additional qualifier information (e.g., dataset info) for saving models.")
parser.add_argument("--single-gpu", action="store_true", help="Run finetuning on one gpu only")
parser.add_argument("-d", "--device", type=int, default=None, help="GPU device number (if single-gpu finetuning)")
parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size for finetuning")
parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--scheduler", type=str, default="linear", help="Learning rate scheduler type")
args = parser.parse_args()

wandb.init(project="Finetuning_political_LMs")

# Load the dataset
base_model = args.model
new_model = f"finetuned_{args.name}_{args.model}"
full_dataset = load_dataset(
    args.dataset,
    split="train")
print(f"Loaded full dataset with {len(full_dataset)} finetuning samples.")

# Load the model for lower-precision finetuning for computational efficiency
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto" if not args.single_gpu else None,
    token=HUGGINGFACE_ACCESS_TOKEN,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(
    base_model, trust_remote_code=True, token=HUGGINGFACE_ACCESS_TOKEN)
special_tokens = []
for speaker in ["Q", "CLINTON", "TRUMP", "KAINE", "PENCE", "HOST"]:
    special_tokens.append(f"[START_{speaker}]")
    special_tokens.append(f"[END_{speaker}]")
tokenizer.add_tokens(special_tokens, special_tokens=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.resize_token_embeddings(len(tokenizer))

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set up training
training_params = TrainingArguments(
    output_dir=f"finetuned_{base_model}/on_clinton/test/",
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=args.lr,
    weight_decay=1e-3,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb",
    run_name=f"{base_model}:fine_tuning_{args.name}"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=full_dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# Finetune the model
print("...beginning finetuning.")
trainer.train()
print("...finetuning completed.")

# Save the new result
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)
print(f"Completed model saved to {new_model}.")
