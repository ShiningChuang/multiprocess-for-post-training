#!/usr/bin/env python3
"""
GSM8K evaluation harness. Reusable for base model or any LoRA adapter.

Usage:
  python eval_gsm8k.py                      # eval base model
  python eval_gsm8k.py --adapter PATH       # eval base + LoRA adapter
  python eval_gsm8k.py --n 200 --device cuda:0
"""
import os, re, time, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH = "/data/models/Qwen2.5-1.5B-Instruct"

def extract_answer(text):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip().replace(",", "")
    hash_match = re.findall(r'####\s*(.+)', text)
    if hash_match:
        return hash_match[-1].strip().replace(",", "")
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else ""

def extract_gt(answer_text):
    m = re.findall(r'####\s*(.+)', answer_text)
    return m[-1].strip().replace(",", "") if m else ""

def build_prompt(q):
    return ("Solve the following math problem step by step. "
            "Put your final numerical answer in \\boxed{}.\n\n"
            f"Problem: {q}\n\nSolution:")

def is_correct(pred, gt):
    try:
        return abs(float(pred) - float(gt)) < 1e-5
    except (ValueError, TypeError):
        return pred == gt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="path to LoRA adapter dir (optional)")
    ap.add_argument("--n", type=int, default=100, help="number of test questions")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--label", default="base")
    args = ap.parse_args()

    device = args.device
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map=device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()  # fold LoRA into base for clean inference
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = min(args.n, len(ds))

    correct = 0
    t0 = time.time()
    for i in range(n):
        q = ds[i]["question"]
        gt = extract_gt(ds[i]["answer"])
        prompt = build_prompt(q)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,  # greedy — deterministic eval
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(resp)
        if is_correct(pred, gt):
            correct += 1
        if (i + 1) % 20 == 0:
            print(f"  [{args.label}] {i+1}/{n}  running acc={correct/(i+1):.3f}", flush=True)

    acc = correct / n
    elapsed = time.time() - t0
    print(f"\n[{args.label}] GSM8K test accuracy: {correct}/{n} = {acc:.4f}  ({elapsed:.0f}s, greedy)")
    return acc

if __name__ == "__main__":
    main()
