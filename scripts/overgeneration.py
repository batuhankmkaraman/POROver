"""
Overgeneration (+ rejection sampling) for instruction finetuning data.

Given a set of prompts, sample K candidate completions per prompt from a
teacher model and keep one per prompt according to a rejection-sampling
criterion:
  - "random": pick a uniformly random candidate (isolates the effect of
    overgeneration itself from any scoring criterion).
  - "reward_model": score every candidate with a sequence-classification
    reward model and keep the highest-scoring one (e.g. OpenAssistant's
    DeBERTa reward model or ArmoRM).

Example:
    python overgeneration.py \
        --model_name meta-llama/Meta-Llama-3-70B-Instruct \
        --input_json prompts.json \
        --output_json completions.json \
        --num_completions 8 \
        --criterion reward_model \
        --reward_model_name OpenAssistant/reward-model-deberta-v3-large-v2
"""

import argparse
import json
import random

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def generate_candidates(model, tokenizer, prompts, args):
    """Sample `num_completions` completions per prompt, batched over prompts."""
    all_candidates = [[] for _ in prompts]
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.num_completions,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        completions = tokenizer.batch_decode(
            outputs[:, prompt_len:], skip_special_tokens=True
        )
        for i in range(len(batch_prompts)):
            all_candidates[start + i] = [
                c.strip()
                for c in completions[
                    i * args.num_completions : (i + 1) * args.num_completions
                ]
            ]
        print(f"generated candidates for prompts {start}-{start + len(batch_prompts) - 1}")
    return all_candidates


@torch.no_grad()
def score_candidates(reward_model, reward_tokenizer, prompt, candidates, device):
    scores = []
    for candidate in candidates:
        inputs = reward_tokenizer(
            prompt, candidate, return_tensors="pt", truncation=True
        ).to(device)
        score = reward_model(**inputs).logits[0].mean().item()
        scores.append(score)
    return scores


def select_completion(prompt, candidates, args, reward_model=None, reward_tokenizer=None):
    if args.criterion == "random":
        return random.choice(candidates)
    scores = score_candidates(reward_model, reward_tokenizer, prompt, candidates, reward_model.device)
    return candidates[max(range(len(candidates)), key=lambda i: scores[i])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True, help="Teacher model used to overgenerate completions")
    parser.add_argument("--input_json", required=True, help='JSON list of {"prompt": ...} objects')
    parser.add_argument("--output_json", required=True, help='Output JSON list of {"prompt", "completion", "candidates"}')
    parser.add_argument("--num_completions", type=int, default=8, help="Number of completions (K) to overgenerate per prompt")
    parser.add_argument("--criterion", choices=["random", "reward_model"], default="random")
    parser.add_argument("--reward_model_name", default=None, help="Required when --criterion reward_model")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.criterion == "reward_model" and not args.reward_model_name:
        parser.error("--reward_model_name is required when --criterion reward_model")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    reward_model, reward_tokenizer = None, None
    if args.criterion == "reward_model":
        reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        reward_model.eval()

    data = read_json(args.input_json)
    prompts = [d["prompt"] for d in data]

    candidates = generate_candidates(model, tokenizer, prompts, args)

    results = []
    for prompt, cands in zip(prompts, candidates):
        best = select_completion(prompt, cands, args, reward_model, reward_tokenizer)
        results.append({"prompt": prompt, "completion": best, "candidates": cands})

    write_json(results, args.output_json)
    print(f"saved {len(results)} overgenerated completions to {args.output_json}")


if __name__ == "__main__":
    main()
