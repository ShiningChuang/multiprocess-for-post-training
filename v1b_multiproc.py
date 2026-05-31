#!/usr/bin/env python3
"""
V1b Multiprocess GRPO - Two processes sharing one GPU.
Generator process and Learner process on the same cuda:0.
Measures GPU contention overhead vs V1a single-process baseline.

Usage: python v1b_multiproc.py
"""

import os
import re
import sys
import time
import json
import signal
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from queue import Empty
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# ══════════════════════════════════════════════════════════════
# Config (same as V1a for fair comparison)
# ══════════════════════════════════════════════════════════════
MODEL_PATH = "/data/models/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda:0"
NUM_TRAIN_STEPS = 30
BATCH_SIZE = 4   # downsized from 8 for 2-process single-GPU memory budget
GROUP_SIZE = 4   # downsized from 8; same 16 samples/step as V0 for fair throughput comparison
MAX_PROMPT_LEN = 256
MAX_RESPONSE_LEN = 128   # downsized from 256 after 1st OOM (Learner forward activations)
LEARNING_RATE = 5e-6
KL_COEFF = 0.1
CLIP_EPS = 0.2
LORA_RANK = 64
LORA_ALPHA = 128
LOG_EVERY = 5
SAVE_DIR = "/data/project/v1b_results"

# Queue config
QUEUE_MAXSIZE = 2         # small buffer: we want to measure contention, not hide it
WEIGHT_SYNC_EVERY = 1     # sync weights after every learner step

# ══════════════════════════════════════════════════════════════
# Shared utilities (reward, data loading - same as V1a)
# ══════════════════════════════════════════════════════════════

def extract_gsm8k_answer(text: str) -> str:
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
    match = re.findall(r'####\s*(.+)', answer_text)
    if match:
        return match[-1].strip().replace(",", "")
    return ""

def compute_reward(response: str, ground_truth: str) -> float:
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
# Generator Process
# ══════════════════════════════════════════════════════════════

def generator_process(
    data_queue: mp.Queue,       # output: send rollout data to learner
    weight_queue: mp.Queue,     # input: receive updated weights from learner
    all_data: list,
    num_steps: int,
    batch_size: int,
    group_size: int,
):
    """Generator process: generates responses, scores them, sends to learner."""
    try:
        print(f"[Generator] Starting on {DEVICE}, PID={os.getpid()}", flush=True)

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Attempt 3 downsize: Generator uses base model only (no LoRA).
        # Weight sync becomes ineffective — Generator stays at base policy
        # for the whole run. We accept this for the memory measurement;
        # it means V1b measures throughput/contention but not on-policy learning.
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map=DEVICE
        )
        model.eval()

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        data_idx = 0

        for step in range(1, num_steps + 1):
            t_step_start = time.time()

            # Generator has no LoRA in attempt-3 config; drain and discard any
            # weight updates from learner to keep the queue from filling.
            try:
                while not weight_queue.empty():
                    _ = weight_queue.get_nowait()
            except Empty:
                pass

            # Sample batch
            batch_data = []
            for _ in range(batch_size):
                batch_data.append(all_data[data_idx % len(all_data)])
                data_idx += 1
            prompts = [d[0] for d in batch_data]
            ground_truths = [d[1] for d in batch_data]

            # Generate responses (batched, same as V1a)
            t_gen_start = time.time()
            all_response_texts = []
            all_old_log_probs = []
            all_response_ids = []
            all_prompt_lengths = []

            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                  max_length=MAX_PROMPT_LEN).to(DEVICE)
                prompt_len = inputs["input_ids"].shape[1]
                all_prompt_lengths.append(prompt_len)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_RESPONSE_LEN,
                        do_sample=True, temperature=0.7, top_p=0.9,
                        num_return_sequences=group_size,
                        pad_token_id=pad_token_id,
                        return_dict_in_generate=True,
                    )

                sequences = outputs.sequences
                group_texts = []
                group_resp_ids = []

                for g in range(group_size):
                    resp_ids = sequences[g, prompt_len:]
                    non_pad = (resp_ids != pad_token_id)
                    if non_pad.any():
                        last_non_pad = non_pad.nonzero()[-1].item() + 1
                        resp_ids = resp_ids[:last_non_pad]
                    resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
                    group_texts.append(resp_text)
                    group_resp_ids.append(resp_ids.cpu())  # move to CPU for queue transfer

                # Compute old log probs
                with torch.no_grad():
                    full_logits = model(sequences, use_cache=False).logits

                group_old_lps = []
                for g in range(group_size):
                    resp_ids = group_resp_ids[g]
                    resp_len = len(resp_ids)
                    if resp_len == 0:
                        group_old_lps.append(torch.tensor([]))
                        continue
                    pred_logits = full_logits[g, prompt_len-1:prompt_len+resp_len-1, :]
                    log_probs = F.log_softmax(pred_logits, dim=-1)
                    token_lps = log_probs.gather(
                        1, resp_ids.to(DEVICE)[:resp_len].unsqueeze(1)
                    ).squeeze(1).cpu()  # move to CPU for queue
                    group_old_lps.append(token_lps)

                all_response_texts.append(group_texts)
                all_old_log_probs.append(group_old_lps)
                all_response_ids.append(group_resp_ids)

            t_gen = time.time() - t_gen_start

            # Score responses (CPU)
            t_score_start = time.time()
            rewards = []
            for resps, gt in zip(all_response_texts, ground_truths):
                group_rewards = [compute_reward(r, gt) for r in resps]
                rewards.append(group_rewards)
            t_score = time.time() - t_score_start

            # Compute advantages
            advantages = []
            for group_rewards in rewards:
                t = torch.tensor(group_rewards, dtype=torch.float32)
                mean, std = t.mean(), t.std()
                adv = torch.zeros_like(t) if std < 1e-8 else (t - mean) / (std + 1e-8)
                advantages.append(adv.tolist())

            # Package data and send to learner
            rollout_data = {
                "step": step,
                "prompts": prompts,
                "response_ids": all_response_ids,       # list[list[Tensor]] on CPU
                "old_log_probs": all_old_log_probs,      # list[list[Tensor]] on CPU
                "advantages": advantages,                 # list[list[float]]
                "rewards": rewards,                       # list[list[float]]
                "t_gen": t_gen,
                "t_score": t_score,
            }

            data_queue.put(rollout_data)
            t_total = time.time() - t_step_start
            print(f"[Generator] Step {step} done: gen={t_gen:.1f}s score={t_score:.3f}s total={t_total:.1f}s", flush=True)

        # Signal done
        data_queue.put(None)
        print("[Generator] Finished all steps, sent stop signal.", flush=True)

    except Exception as e:
        print(f"[Generator] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        data_queue.put(None)

# ══════════════════════════════════════════════════════════════
# Learner Process
# ══════════════════════════════════════════════════════════════

def learner_process(
    data_queue: mp.Queue,       # input: receive rollout data from generator
    weight_queue: mp.Queue,     # output: send updated weights to generator
    metrics_list: list,         # shared list for metrics (managed by mp.Manager)
):
    """Learner process: receives rollout data, computes GRPO loss, updates policy."""
    try:
        print(f"[Learner] Starting on {DEVICE}, PID={os.getpid()}", flush=True)

        # Load models
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Policy model with LoRA
        policy_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map=DEVICE
        )
        lora_config = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(policy_model, lora_config)

        # Gradient checkpointing — trades ~25% compute for ~1 GB activation memory.
        # Needed to fit Learner alongside Generator on a single 16 GB V100.
        policy_model.gradient_checkpointing_enable()
        policy_model.enable_input_require_grads()  # needed for grad through LoRA with checkpointing

        # Reference model (frozen)
        ref_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map=DEVICE
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            [p for p in policy_model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
        )

        step_count = 0

        while True:
            # Wait for data from generator
            t_wait_start = time.time()
            rollout_data = data_queue.get()
            t_wait = time.time() - t_wait_start

            if rollout_data is None:
                print("[Learner] Received stop signal. Exiting.", flush=True)
                break

            step_count += 1
            step = rollout_data["step"]
            prompts = rollout_data["prompts"]
            response_ids = rollout_data["response_ids"]
            old_log_probs_list = rollout_data["old_log_probs"]
            advantages = rollout_data["advantages"]
            rewards = rollout_data["rewards"]

            # === Training step ===
            t_train_start = time.time()
            policy_model.train()

            total_policy_loss = torch.tensor(0.0, device=DEVICE)
            total_kl = 0.0
            num_valid = 0

            for i in range(len(prompts)):
                for j in range(GROUP_SIZE):
                    adv = advantages[i][j]
                    resp_ids = response_ids[i][j].to(DEVICE)
                    old_lp = old_log_probs_list[i][j].to(DEVICE)

                    if len(resp_ids) == 0 or len(old_lp) == 0:
                        continue

                    # Build full sequence
                    prompt_ids = tokenizer(
                        prompts[i], return_tensors="pt", truncation=True,
                        max_length=MAX_PROMPT_LEN
                    )["input_ids"][0].to(DEVICE)
                    full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)
                    prompt_len = len(prompt_ids)
                    resp_len = len(resp_ids)

                    # New log probs (with gradient)
                    outputs = policy_model(full_ids, use_cache=False)
                    logits = outputs.logits[0]
                    pred_logits = logits[prompt_len-1:prompt_len+resp_len-1, :]
                    log_probs = F.log_softmax(pred_logits, dim=-1)
                    new_lp = log_probs.gather(
                        1, resp_ids[:resp_len].unsqueeze(1)
                    ).squeeze(1)

                    # Ref log probs
                    with torch.no_grad():
                        ref_out = ref_model(full_ids, use_cache=False)
                        ref_logits = ref_out.logits[0]
                        ref_pred = ref_logits[prompt_len-1:prompt_len+resp_len-1, :]
                        ref_lps = F.log_softmax(ref_pred, dim=-1)
                        ref_lp = ref_lps.gather(
                            1, resp_ids[:resp_len].unsqueeze(1)
                        ).squeeze(1)

                    # Align lengths
                    min_len = min(len(new_lp), len(old_lp), len(ref_lp))
                    new_lp = new_lp[:min_len]
                    old_lp_a = old_lp[:min_len].detach()
                    ref_lp = ref_lp[:min_len]

                    # Importance ratio + clipping
                    ratio = torch.exp(new_lp - old_lp_a)
                    adv_t = torch.tensor(adv, device=DEVICE, dtype=torch.float32)
                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # KL
                    kl_per_token = new_lp - ref_lp
                    kl_loss = KL_COEFF * kl_per_token.mean()

                    total_policy_loss = total_policy_loss + policy_loss + kl_loss
                    total_kl += kl_per_token.mean().item()
                    num_valid += 1

            if num_valid > 0:
                avg_loss = total_policy_loss / num_valid
                optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                optimizer.step()

            t_train = time.time() - t_train_start

            # === Send updated weights to generator ===
            t_sync_start = time.time()
            if step_count % WEIGHT_SYNC_EVERY == 0:
                # Only send LoRA parameters (much smaller than full model)
                lora_state = {
                    k: v.cpu() for k, v in policy_model.state_dict().items()
                    if "lora_" in k
                }
                weight_queue.put(lora_state)
            t_sync = time.time() - t_sync_start

            # === Metrics ===
            flat_rewards = [r for group in rewards for r in group]
            mean_reward = sum(flat_rewards) / len(flat_rewards)
            accuracy = sum(1 for r in flat_rewards if r >= 1.0) / len(flat_rewards)

            metrics = {
                "step": step,
                "mean_reward": mean_reward,
                "accuracy": accuracy,
                "mean_kl": total_kl / max(num_valid, 1),
                "loss": avg_loss.item() if num_valid > 0 else 0.0,
                "t_gen": rollout_data["t_gen"],
                "t_score": rollout_data["t_score"],
                "t_train": t_train,
                "t_wait": t_wait,
                "t_sync": t_sync,
                "t_total": rollout_data["t_gen"] + rollout_data["t_score"] + t_train + t_wait + t_sync,
                "num_samples": num_valid,
            }
            metrics_list.append(metrics)

            if step % LOG_EVERY == 0 or step == 1:
                print(
                    f"[Learner] Step {step:4d} | "
                    f"reward={mean_reward:.3f} | acc={accuracy:.3f} | "
                    f"loss={avg_loss.item() if num_valid > 0 else 0:.4f} | kl={total_kl/max(num_valid,1):.4f} | "
                    f"gen={rollout_data['t_gen']:.1f}s train={t_train:.1f}s "
                    f"wait={t_wait:.1f}s sync={t_sync:.2f}s",
                    flush=True
                )

            # Periodic checkpoint
            if step % 10 == 0:
                ckpt_path = os.path.join(SAVE_DIR, "metrics_checkpoint.json")
                with open(ckpt_path, "w") as f:
                    json.dump(list(metrics_list), f, indent=2)

        # Final save
        metrics_path = os.path.join(SAVE_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(list(metrics_list), f, indent=2)
        print(f"[Learner] Metrics saved to {metrics_path}", flush=True)

    except Exception as e:
        print(f"[Learner] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

# ══════════════════════════════════════════════════════════════
# Main: Launch both processes
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    mp.set_start_method("spawn", force=True)

    print("=" * 60)
    print("V1b Multiprocess GRPO - Two Processes, One GPU")
    print("=" * 60)
    print(f"  Generator + Learner both on {DEVICE}")
    print(f"  Batch: {BATCH_SIZE} × {GROUP_SIZE} = {BATCH_SIZE * GROUP_SIZE} samples/step")
    print(f"  Steps: {NUM_TRAIN_STEPS}")
    print(f"  Queue max size: {QUEUE_MAXSIZE}")
    print(f"  Weight sync every: {WEIGHT_SYNC_EVERY} steps")
    print(flush=True)

    # Load data in main process
    print("\nLoading GSM8K dataset...", flush=True)
    all_data = load_gsm8k_prompts()
    print(f"  Loaded {len(all_data)} training problems", flush=True)

    # Create communication queues
    data_queue = mp.Queue(maxsize=QUEUE_MAXSIZE)
    weight_queue = mp.Queue(maxsize=2)

    # Shared metrics list
    manager = mp.Manager()
    metrics_list = manager.list()

    # Launch processes
    print("\nLaunching Generator and Learner processes...", flush=True)

    gen_proc = mp.Process(
        target=generator_process,
        args=(data_queue, weight_queue, all_data, NUM_TRAIN_STEPS, BATCH_SIZE, GROUP_SIZE),
        name="Generator"
    )
    learn_proc = mp.Process(
        target=learner_process,
        args=(data_queue, weight_queue, metrics_list),
        name="Learner"
    )

    # Start Generator first, let its CUDA context allocate, then start Learner.
    # Avoids the OOM race where both processes try to allocate the model
    # simultaneously before either has finished init.
    gen_proc.start()
    print(f"  Generator PID: {gen_proc.pid} (sleeping 15s for CUDA init)", flush=True)
    time.sleep(15)
    learn_proc.start()
    print(f"  Learner PID: {learn_proc.pid}", flush=True)

    # Wait for both to finish
    gen_proc.join()
    print("[Main] Generator process finished.", flush=True)
    learn_proc.join()
    print("[Main] Learner process finished.", flush=True)

    # Print summary
    metrics = list(metrics_list)
    if metrics:
        print("\n" + "=" * 60)
        print("Training Summary (V1b - Multiprocess Single GPU)")
        print("=" * 60)

        n = len(metrics)
        if n >= 10:
            early = metrics[:10]
            late = metrics[-10:]
            print(f"  Early (steps 1-10):  mean_reward={sum(m['mean_reward'] for m in early)/len(early):.3f}, "
                  f"accuracy={sum(m['accuracy'] for m in early)/len(early):.3f}")
            print(f"  Late  (last 10):     mean_reward={sum(m['mean_reward'] for m in late)/len(late):.3f}, "
                  f"accuracy={sum(m['accuracy'] for m in late)/len(late):.3f}")

        avg = lambda key: sum(m[key] for m in metrics) / n
        total = avg("t_gen") + avg("t_score") + avg("t_train") + avg("t_wait") + avg("t_sync")
        print(f"\n  Avg timing per step:")
        print(f"    Generation:  {avg('t_gen'):.1f}s ({avg('t_gen')/total*100:.0f}%)")
        print(f"    Scoring:     {avg('t_score'):.3f}s ({avg('t_score')/total*100:.0f}%)")
        print(f"    Training:    {avg('t_train'):.1f}s ({avg('t_train')/total*100:.0f}%)")
        print(f"    Queue wait:  {avg('t_wait'):.1f}s ({avg('t_wait')/total*100:.0f}%)")
        print(f"    Weight sync: {avg('t_sync'):.2f}s ({avg('t_sync')/total*100:.0f}%)")
        print(f"    Total:       {total:.1f}s")
        print(f"    Throughput:   {BATCH_SIZE * GROUP_SIZE / total:.2f} samples/sec")

    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
