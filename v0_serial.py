#!/usr/bin/env python3
"""
V0 Serial GRPO - Single process, single GPU baseline.
Usage: python v0_serial.py
       NUM_TRAIN_STEPS=3 python v0_serial.py   # smoke test
"""

import os
import re
import time
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
MODEL_PATH = "/data/models/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda:0"
NUM_TRAIN_STEPS = int(os.environ.get("NUM_TRAIN_STEPS", 200))
BATCH_SIZE = 4          # prompts per step (small for V0, single GPU)
GROUP_SIZE = 4           # responses per prompt (N in GRPO)
MAX_PROMPT_LEN = 256
MAX_RESPONSE_LEN = 256  # 512 → 256: halves gen time; most GSM8K solutions fit in 256
LEARNING_RATE = 5e-5
KL_COEFF = 0.04         # beta for KL penalty
CLIP_EPS = 0.2          # PPO-style clipping epsilon
LORA_RANK = 64
LORA_ALPHA = 128
LOG_EVERY = 5
SAVE_DIR = "/data/project/v0_results"

# ══════════════════════════════════════════════════════════════
# GSM8K Reward Function
# ══════════════════════════════════════════════════════════════

def extract_gsm8k_answer(text: str) -> str:
    """Extract the final numerical answer from GSM8K format.
    GSM8K ground truth format: '#### <number>'
    Model output format: we look for \\boxed{<number>} or #### <number>
    """
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip().replace(",", "")

    hash_match = re.findall(r'####\s*(.+)', text)
    if hash_match:
        return hash_match[-1].strip().replace(",", "")

    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    return ""

def extract_ground_truth(answer_text: str) -> str:
    """Extract ground truth from GSM8K answer field (format: reasoning\n#### number)."""
    match = re.findall(r'####\s*(.+)', answer_text)
    if match:
        return match[-1].strip().replace(",", "")
    return ""

def compute_reward(response: str, ground_truth: str) -> float:
    """Rule-based reward for GSM8K.
    - Exact match on numerical answer: +1.0
    - Format bonus (uses \\boxed{}): +0.2
    """
    pred = extract_gsm8k_answer(response)
    gt = extract_ground_truth(ground_truth)

    reward = 0.0

    try:
        if pred and gt and abs(float(pred) - float(gt)) < 1e-5:
            reward += 1.0
    except ValueError:
        if pred == gt:
            reward += 1.0

    if '\\boxed{' in response:
        reward += 0.2

    return reward

# ══════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════

def build_prompt(question: str) -> str:
    """Build a chat-style prompt for GSM8K."""
    return (
        "Solve the following math problem step by step. "
        "Put your final numerical answer in \\boxed{}.\n\n"
        f"Problem: {question}\n\n"
        "Solution:"
    )

def load_gsm8k_prompts():
    """Load GSM8K and return list of (prompt, ground_truth_answer)."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    data = []
    for item in ds:
        prompt = build_prompt(item["question"])
        gt = item["answer"]
        data.append((prompt, gt))
    return data

# ══════════════════════════════════════════════════════════════
# GRPO Core Logic
# ══════════════════════════════════════════════════════════════

def generate_responses(model, tokenizer, prompts, group_size, max_len):
    """Generate group_size responses for each prompt.
    Returns:
        all_response_texts: list of list of str, shape [batch, group_size]
        all_response_ids: list of list of tensor, token ids for each response
        all_prompt_ids: list of tensor, token ids for each prompt
    """
    model.eval()
    all_response_texts = []
    all_response_ids = []
    all_prompt_ids = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=MAX_PROMPT_LEN).to(DEVICE)
        prompt_ids = inputs["input_ids"][0]
        all_prompt_ids.append(prompt_ids)

        responses = []
        response_ids_list = []

        for _ in range(group_size):
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_len,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            resp_ids = output[0][len(prompt_ids):]
            resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
            responses.append(resp_text)
            response_ids_list.append(resp_ids)

        all_response_texts.append(responses)
        all_response_ids.append(response_ids_list)

    return all_response_texts, all_response_ids, all_prompt_ids

def compute_grpo_advantages(rewards_per_prompt):
    """Compute GRPO advantages via group normalization."""
    advantages = []
    for group_rewards in rewards_per_prompt:
        t = torch.tensor(group_rewards, dtype=torch.float32)
        mean = t.mean()
        std = t.std()
        if std < 1e-8:
            adv = torch.zeros_like(t)
        else:
            adv = (t - mean) / (std + 1e-8)
        advantages.append(adv.tolist())
    return advantages

def compute_log_probs(model, prompt_ids, response_ids, requires_grad: bool):
    """Compute per-token log probabilities of response given prompt.
    Returns tensor of shape [resp_len].
    """
    full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(DEVICE)

    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        # use_cache=False: we want logits over the full sequence, not autoregressive
        # generation. The KV cache would just waste memory here.
        outputs = model(full_ids, use_cache=False)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    prompt_len = len(prompt_ids)
    resp_len = len(response_ids)

    # logits[t] predicts token[t+1], so for the t-th response token
    # we use the logit at position (prompt_len-1 + t).
    pred_logits = logits[0, prompt_len-1 : prompt_len+resp_len-1, :]  # [resp_len, vocab]
    target_ids = response_ids.to(DEVICE)

    log_probs = F.log_softmax(pred_logits.float(), dim=-1)  # cast for numerical stability
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    return token_log_probs

def grpo_step(policy_model, ref_model, tokenizer, optimizer, prompts, ground_truths, group_size):
    """Execute one GRPO training step.

    Returns dict with metrics.
    """
    # === Phase 1: Generate responses ===
    t_gen_start = time.time()
    response_texts, response_ids, prompt_ids = generate_responses(
        policy_model, tokenizer, prompts, group_size, MAX_RESPONSE_LEN
    )
    # Release any KV-cache fragments lingering from generation before we start
    # the training phase, which is memory-hungry.
    torch.cuda.empty_cache()
    t_gen = time.time() - t_gen_start

    # === Phase 2: Score responses (rule-based, CPU) ===
    t_score_start = time.time()
    rewards = []
    for i, (resps, gt) in enumerate(zip(response_texts, ground_truths)):
        group_rewards = [compute_reward(r, gt) for r in resps]
        rewards.append(group_rewards)
    t_score = time.time() - t_score_start

    # === Phase 3: Compute advantages ===
    advantages = compute_grpo_advantages(rewards)

    # === Phase 4: Policy update ===
    # Per-sample backward + grad accumulation: we divide each loss by total
    # sample count and call .backward() immediately, so activations are freed
    # right after each backward instead of being held for all 16 samples.
    # Without this, a 1.5B policy + 1.5B ref + 16 stacked activation graphs
    # OOMs on a 16 GB V100.
    t_train_start = time.time()
    policy_model.train()

    total_loss_value = 0.0
    total_kl = 0.0
    num_samples_planned = sum(
        1 for i in range(len(prompts)) for j in range(group_size)
        if len(response_ids[i][j]) > 0
    )
    if num_samples_planned == 0:
        t_train = time.time() - t_train_start
        flat_rewards = [r for group in rewards for r in group]
        mean_reward = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0.0
        return {
            "mean_reward": mean_reward,
            "accuracy": 0.0,
            "mean_kl": 0.0,
            "loss": 0.0,
            "t_gen": t_gen,
            "t_score": t_score,
            "t_train": t_train,
            "t_total": t_gen + t_score + t_train,
            "num_samples": 0,
        }

    optimizer.zero_grad()
    num_samples = 0

    for i in range(len(prompts)):
        for j in range(group_size):
            adv = advantages[i][j]
            resp_ids = response_ids[i][j]
            p_ids = prompt_ids[i]

            if len(resp_ids) == 0:
                continue

            # Policy log probs (with grad)
            policy_log_probs = compute_log_probs(policy_model, p_ids, resp_ids, requires_grad=True)

            # Reference model log probs (no grad)
            ref_log_probs = compute_log_probs(ref_model, p_ids, resp_ids, requires_grad=False)

            # KL proxy: per-token (policy - ref) — sign matters for the penalty direction
            kl_per_token = policy_log_probs - ref_log_probs.detach()
            kl = kl_per_token.mean()

            # GRPO loss: -advantage * sum(log_probs) + beta * sum(KL)
            policy_loss = -(adv * policy_log_probs.sum()) + KL_COEFF * kl_per_token.sum()

            # Scale by total sample count and backward immediately.
            # Equivalent to mean-loss backward, but frees activations per-sample.
            (policy_loss / num_samples_planned).backward()

            total_loss_value += policy_loss.item()
            total_kl += kl.item()
            num_samples += 1

    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()

    avg_loss_value = total_loss_value / max(num_samples, 1)
    t_train = time.time() - t_train_start

    # === Compute metrics ===
    flat_rewards = [r for group in rewards for r in group]
    mean_reward = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0.0
    accuracy = sum(1 for r in flat_rewards if r >= 1.0) / len(flat_rewards) if flat_rewards else 0.0

    return {
        "mean_reward": mean_reward,
        "accuracy": accuracy,
        "mean_kl": total_kl / max(num_samples, 1),
        "loss": avg_loss_value,
        "t_gen": t_gen,
        "t_score": t_score,
        "t_train": t_train,
        "t_total": t_gen + t_score + t_train,
        "num_samples": num_samples,
    }

# ══════════════════════════════════════════════════════════════
# Main Training Loop
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("V0 Serial GRPO - Single Process Baseline")
    print("=" * 60)

    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] Loading policy model with LoRA...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=DEVICE
    )
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy_model = get_peft_model(policy_model, lora_config)
    policy_model.print_trainable_parameters()

    print("[3/5] Loading reference model (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map=DEVICE
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
    )

    print("[4/5] Loading GSM8K dataset...")
    all_data = load_gsm8k_prompts()
    print(f"  Loaded {len(all_data)} training problems")

    print(f"\n[5/5] Starting training for {NUM_TRAIN_STEPS} steps...")
    print(f"  Batch size: {BATCH_SIZE} prompts × {GROUP_SIZE} responses = {BATCH_SIZE * GROUP_SIZE} samples/step")
    print(f"  Device: {DEVICE}")
    print()

    metrics_log = []
    data_idx = 0

    for step in range(1, NUM_TRAIN_STEPS + 1):
        batch_data = []
        for _ in range(BATCH_SIZE):
            batch_data.append(all_data[data_idx % len(all_data)])
            data_idx += 1

        prompts = [d[0] for d in batch_data]
        ground_truths = [d[1] for d in batch_data]

        metrics = grpo_step(
            policy_model, ref_model, tokenizer, optimizer,
            prompts, ground_truths, GROUP_SIZE
        )
        metrics["step"] = step
        metrics_log.append(metrics)

        if step % LOG_EVERY == 0 or step == 1:
            print(
                f"Step {step:4d} | "
                f"reward={metrics['mean_reward']:.3f} | "
                f"acc={metrics['accuracy']:.3f} | "
                f"loss={metrics['loss']:.4f} | "
                f"kl={metrics['mean_kl']:.4f} | "
                f"gen={metrics['t_gen']:.1f}s score={metrics['t_score']:.3f}s "
                f"train={metrics['t_train']:.1f}s total={metrics['t_total']:.1f}s",
                flush=True,
            )

    metrics_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    early = metrics_log[:min(20, len(metrics_log))]
    late = metrics_log[-min(20, len(metrics_log)):]
    print(f"  Early (steps 1-{len(early)}):  mean_reward={sum(m['mean_reward'] for m in early)/len(early):.3f}, "
          f"accuracy={sum(m['accuracy'] for m in early)/len(early):.3f}")
    print(f"  Late  (last {len(late)}): mean_reward={sum(m['mean_reward'] for m in late)/len(late):.3f}, "
          f"accuracy={sum(m['accuracy'] for m in late)/len(late):.3f}")

    avg_timing = {
        "gen": sum(m["t_gen"] for m in metrics_log) / len(metrics_log),
        "score": sum(m["t_score"] for m in metrics_log) / len(metrics_log),
        "train": sum(m["t_train"] for m in metrics_log) / len(metrics_log),
    }
    total = sum(avg_timing.values())
    print(f"\n  Avg timing per step:")
    print(f"    Generation: {avg_timing['gen']:.1f}s ({avg_timing['gen']/total*100:.0f}%)")
    print(f"    Scoring:    {avg_timing['score']:.3f}s ({avg_timing['score']/total*100:.0f}%)")
    print(f"    Training:   {avg_timing['train']:.1f}s ({avg_timing['train']/total*100:.0f}%)")
    print(f"    Total:      {total:.1f}s")
    print(f"    Throughput:  {BATCH_SIZE * GROUP_SIZE / total:.2f} samples/sec")

if __name__ == "__main__":
    main()
