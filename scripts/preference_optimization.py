"""
POROver: Preference Optimization for Reducing Overrefusal.

Runs Direct Preference Optimization (DPO) on a preference dataset of
(prompt, chosen, rejected) triples, as described in Section 3.2 of the paper.
The preference data mixes two sources:
  - seemingly toxic prompts, where the losing completion is the target
    model's own overrefusal and the winning completion is a compliant,
    helpful response (Figure 2 / Algorithm 1);
  - toxic prompts, where the winning/losing completions are the
    safest/least-safe completions among overgenerated candidates
    (Algorithm 2).

Expects --train_file / --val_file to be JSON(L) files with records of the
form {"prompt": ..., "chosen": ..., "rejected": ...}.

Example:
    python preference_optimization.py \
        --model_name_or_path ./checkpoints/sft-llama-3.1-8b \
        --train_file dpo_train.json \
        --val_file dpo_val.json \
        --output_dir ./checkpoints/porover-llama-3.1-8b \
        --beta 0.025
"""

import argparse

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True, help="SFT checkpoint to align with DPO")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--beta", type=float, default=0.025, help="DPO temperature; lower = less divergence from the reference model")
    parser.add_argument("--loss_type", default="sigmoid")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = {"": PartialState().process_index}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model.gradient_checkpointing_enable()

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    ).eval()

    dataset = load_dataset(
        "json", data_files={"train": args.train_file, "validation": args.val_file}
    )

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        bf16=True,
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        learning_rate=args.learning_rate,
        report_to=None,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
