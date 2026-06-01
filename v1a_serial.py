#!/usr/bin/env python3
"""
V1a Serial GRPO - Fixed algorithm + batched generation.
Single process, single GPU. Baseline for multiprocess comparison.

Key fixes vs V0:
  1. Proper GRPO loss with importance ratio + clipping
  2. Batched generation (GROUP_SIZE responses per generate call)
  3. Larger batch/group size for lower variance

Usage: python v1a_serial.py
"""

import os
import re
import time
import json
import signal
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
MODEL_PATH = "/data/models/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda:0"
NUM_TRAIN_STEPS = 30
BATCH_SIZE = 8           # prompts per step
GROUP_SIZE = 8            # responses per prompt (N in GRPO)
MAX_PROMPT_LEN = 256
MAX_RESPONSE_LEN = 256
LEARNING_RATE = 5e-6     # lower LR for stability (was 5e-5)
KL_COEFF = 0.1           # stronger KL anchor (was 0.04)
CLIP_EPS = 0.2           # PPO-style clipping
LORA_RANK = 64
LORA_ALPHA = 128
LOG_EVERY = 5
SAVE_DIR = "/data/project/v1a_results"

# Graceful shutdown on SIGTERM
_shutdown = False
def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True
signal.signal(signal.SIGTERM, _handle_sigterm)

# ══════════════════════════════════════════════════════════════
# GSM8K Reward Function (same as V0)
# ══════════════════════════════════════════════════════════════

def extract_gsm8k_answer(text: str) -> str:
    """Extract the final numerical answer from model output."""
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
    """Extract ground truth from GSM8K answer field."""
    match = re.findall(r'####\s*(.+)', answer_text)
    if match:
        return match[-1].strip().replace(",", "")
    return ""

def compute_reward(response: str, ground_truth: str) -> float:
    """Rule-based reward for GSM8K."""
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
# Data Loading (same as V0)
# ══════════════════════════════════════════════════════════════

def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Put your final numerical answer in \\boxed{}.\n\n"
        f"Problem: {question}\n\n"
        "Solution:"
    )

def load_gsm8k_prompts():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    data = []
    for item in ds:
        prompt = build_prompt(item["question"])
        gt = item["answer"]
        data.append((prompt, gt))
    return data

# ══════════════════════════════════════════════════════════════
# Batched Generation
# ══════════════════════════════════════════════════════════════

def generate_responses_batched(model, tokenizer, prompts, group_size, max_new_tokens):
    """Generate group_size responses for each prompt using batched generation.

    For each prompt, we call generate() once with num_return_sequences=group_size.
    This is much faster than calling generate() group_size times.

    Returns:
        all_response_texts: list[list[str]], shape [batch, group_size]
        all_old_log_probs: list[list[Tensor]], per-token log probs at generation time
        all_response_ids: list[list[Tensor]], token ids for each response
        all_prompt_lengths: list[int], prompt lengths for each item
    """
    model.eval()
    all_response_texts = []
    all_old_log_probs = []
    all_response_ids = []
    all_prompt_lengths = []

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=MAX_PROMPT_LEN).to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]
        all_prompt_lengths.append(prompt_len)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=group_size,
                pad_token_id=pad_token_id,
                return_dict_in_generate=True,
            )

        sequences = outputs.sequences  # [group_size, prompt_len + resp_len]

        group_texts = []
        group_log_probs = []
        group_resp_ids = []

        for g in range(group_size):
            resp_ids = sequences[g, prompt_len:]
            non_pad = (resp_ids != pad_token_id)
            if non_pad.any():
                last_non_pad = non_pad.nonzero()[-1].item() + 1
                resp_ids = resp_ids[:last_non_pad]

            resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
            group_texts.append(resp_text)
            group_resp_ids.append(resp_ids)

        # Compute old log probs for all responses in this group at once
        with torch.no_grad():
            full_logits = model(sequences, use_cache=False).logits  # [group_size, seq_len, vocab]

        for g in range(group_size):
            resp_ids = group_resp_ids[g]
            resp_len = len(resp_ids)
            if resp_len == 0:
                group_log_probs.append(torch.tensor([], device=DEVICE))
                continue

            pred_logits = full_logits[g, prompt_len-1 : prompt_len+resp_len-1, :]
            log_probs = F.log_softmax(pred_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(
                1, resp_ids[:resp_len].unsqueeze(1)
            ).squeeze(1)
            group_log_probs.append(token_log_probs)

        all_response_texts.append(group_texts)
        all_old_log_probs.append(group_log_probs)
        all_response_ids.append(group_resp_ids)

        del full_logits, outputs, sequences
    torch.cuda.empty_cache()

    return all_response_texts, all_old_log_probs, all_response_ids, all_prompt_lengths

# ══════════════════════════════════════════════════════════════
# GRPO Core Logic (fixed version)
# ══════════════════════════════════════════════════════════════

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

def compute_current_log_probs(model, full_ids, prompt_len, resp_len):
    """Compute per-token log probs for response portion of full_ids.

    Args:
        model: policy model (in train mode)
        full_ids: [1, prompt_len + resp_len] tensor
        prompt_len: int
        resp_len: int
    Returns:
        token_log_probs: [resp_len] tensor with grad
    """
    outputs = model(full_ids, use_cache=False)
    logits = outputs.logits[0]  # [seq_len, vocab]

    pred_logits = logits[prompt_len-1 : prompt_len+resp_len-1, :]  # [resp_len, vocab]
    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
    target_ids = full_ids[0, prompt_len : prompt_len+resp_len]  # [resp_len]
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    return token_log_probs

def grpo_step(policy_model, ref_model, tokenizer, optimizer, prompts, ground_truths, group_size):
    """Execute one GRPO training step with proper importance ratio + clipping."""

    # === Phase 1: Generate responses (batched) ===
    t_gen_start = time.time()
    response_texts, old_log_probs_list, response_ids, prompt_lengths = \
        generate_responses_batched(policy_model, tokenizer, prompts, group_size, MAX_RESPONSE_LEN)
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

    # === Phase 4: Policy update with importance ratio + clipping ===
    # Per-sample backward + grad accumulation (same memory pattern as V0 fix).
    t_train_start = time.time()
    policy_model.train()

    num_valid_planned = sum(
        1 for i in range(len(prompts)) for j in range(group_size)
        if len(response_ids[i][j]) > 0 and len(old_log_probs_list[i][j]) > 0
    )

    total_loss_value = 0.0
    total_kl = 0.0
    num_valid = 0

    if num_valid_planned > 0:
        optimizer.zero_grad()

    for i in range(len(prompts)):
        prompt_ids_full = tokenizer(
            prompts[i], return_tensors="pt", truncation=True,
            max_length=MAX_PROMPT_LEN
        )["input_ids"][0].to(DEVICE)
        actual_prompt_len = len(prompt_ids_full)

        for j in range(group_size):
            adv = advantages[i][j]
            resp_ids = response_ids[i][j]
            old_lp = old_log_probs_list[i][j]

            if len(resp_ids) == 0 or len(old_lp) == 0:
                continue

            actual_resp_len = len(resp_ids)
            full_ids = torch.cat([prompt_ids_full, resp_ids]).unsqueeze(0)

            # Current policy log probs (with gradient)
            new_lp = compute_current_log_probs(
                policy_model, full_ids, actual_prompt_len, actual_resp_len
            )

            min_len = min(len(new_lp), len(old_lp))
            new_lp = new_lp[:min_len]
            old_lp_aligned = old_lp[:min_len].detach()

            # Reference model log probs (for KL)
            with torch.no_grad():
                ref_lp = compute_current_log_probs(
                    ref_model, full_ids, actual_prompt_len, actual_resp_len
                )[:min_len]

            # Importance ratio (per-token)
            log_ratio = new_lp - old_lp_aligned
            ratio = torch.exp(log_ratio)

            # Clipped surrogate loss
            adv_tensor = torch.tensor(adv, device=DEVICE, dtype=torch.float32)
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_tensor
            policy_loss = -torch.min(surr1, surr2).mean()  # per-token mean

            # KL penalty
            kl_per_token = new_lp - ref_lp
            kl_loss = KL_COEFF * kl_per_token.mean()

            sample_loss = policy_loss + kl_loss
            (sample_loss / num_valid_planned).backward()

            total_loss_value += sample_loss.item()
            total_kl += kl_per_token.mean().item()
            num_valid += 1

    if num_valid > 0:
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()
        avg_loss = total_loss_value / num_valid
    else:
        avg_loss = 0.0

    t_train = time.time() - t_train_start

    # === Metrics ===
    flat_rewards = [r for group in rewards for r in group]
    mean_reward = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0.0
    accuracy = sum(1 for r in flat_rewards if r >= 1.0) / len(flat_rewards) if flat_rewards else 0.0

    return {
        "mean_reward": mean_reward,
        "accuracy": accuracy,
        "mean_kl": total_kl / max(num_valid, 1),
        "loss": avg_loss,
        "t_gen": t_gen,
        "t_score": t_score,
        "t_train": t_train,
        "t_total": t_gen + t_score + t_train,
        "num_samples": num_valid,
    }

# ══════════════════════════════════════════════════════════════
# Main Training Loop
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("V1a Serial GRPO - Fixed Algorithm + Batched Generation")
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
    print(f"  LR={LEARNING_RATE}, KL_COEFF={KL_COEFF}, CLIP_EPS={CLIP_EPS}")
    print(f"  Device: {DEVICE}")
    print(flush=True)

    metrics_log = []
    data_idx = 0

    for step in range(1, NUM_TRAIN_STEPS + 1):
        if _shutdown:
            print(f"\n[SIGTERM] Graceful shutdown at step {step-1}")
            break

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
                flush=True
            )

        if step % 20 == 0:
            ckpt_path = os.path.join(SAVE_DIR, "metrics_checkpoint.json")
            with open(ckpt_path, "w") as f:
                json.dump(metrics_log, f, indent=2)

    metrics_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Save final LoRA adapter for post-hoc evaluation
    adapter_path = os.path.join(SAVE_DIR, "adapter")
    policy_model.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    n = len(metrics_log)
    if n >= 20:
        early = metrics_log[:20]
        late = metrics_log[-20:]
        print(f"  Early (steps 1-20):  mean_reward={sum(m['mean_reward'] for m in early)/len(early):.3f}, "
              f"accuracy={sum(m['accuracy'] for m in early)/len(early):.3f}")
        print(f"  Late  (last 20):     mean_reward={sum(m['mean_reward'] for m in late)/len(late):.3f}, "
              f"accuracy={sum(m['accuracy'] for m in late)/len(late):.3f}")

    avg_timing = {
        "gen": sum(m["t_gen"] for m in metrics_log) / n,
        "score": sum(m["t_score"] for m in metrics_log) / n,
        "train": sum(m["t_train"] for m in metrics_log) / n,
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
