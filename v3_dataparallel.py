#!/usr/bin/env python3
"""
V3 Data-Parallel GRPO - 4 symmetric actor-learners.

Each GPU runs a FULL GRPO pipeline on its OWN data shard:
  generate -> score -> local gradients (per-sample backward) -> all-reduce gradients -> step
All 4 GPUs stay in exact sync (DDP-style). No fixed roles, no idle GPU.

vs V2 (functional split: 1 learner + 3 generators):
  - V2: at any instant only 1/4 GPUs busy
        (gen phase: 3 gens busy, learner idle 155s; train phase: learner busy, gens idle).
        Throughput 0.94 samples/sec, Learner idle 76%.
  - V3: at any instant 4/4 GPUs busy (all generate, then all train).
        Phases still SEPARATED in time -> no gen+train CPU contention (avoids the V2i trap).
        NCCL all-reduce works (synchronous collective, all ranks present).
        Effective batch 4x larger (256 vs 64 samples/update) -> lower gradient variance.

Round structure (10 rounds):
  1. each rank generates its own batch (BATCH x GROUP = 64 samples) with current synced policy
  2. each rank scores + computes group-normalized advantages locally
  3. each rank computes gradients locally (per-sample backward + grad accumulation)
  4. all-reduce gradients across 4 ranks (average) -> identical gradient on every rank
  5. clip + optimizer.step() -> all 4 ranks stay bit-identical
Fully on-policy (staleness 0): each rank trains on data it just generated with the synced weights.

Usage: python v3_dataparallel.py
"""

import os
import re
import time
import json
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# ══════════════════════════════════════════════════════════════
# Config (same hyperparams as V2 for direct comparison)
# ══════════════════════════════════════════════════════════════
MODEL_PATH = "/data/models/Qwen2.5-1.5B-Instruct"
WORLD_SIZE = 4
NUM_ROUNDS = 10          # each round = all 4 GPUs generate+train once = 1 synced update
BATCH_SIZE = 8
GROUP_SIZE = 8
MAX_PROMPT_LEN = 256
MAX_RESPONSE_LEN = 256
LEARNING_RATE = 5e-6
KL_COEFF = 0.1
CLIP_EPS = 0.2
LORA_RANK = 64
LORA_ALPHA = 128
SAVE_DIR = "/data/project/v3_results"

# ══════════════════════════════════════════════════════════════
# Reward / data (same as V2)
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
    m = re.findall(r'####\s*(.+)', answer_text)
    return m[-1].strip().replace(",", "") if m else ""

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
    return [(build_prompt(item["question"]), item["answer"]) for item in ds]

def build_lora_config():
    return LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

# ══════════════════════════════════════════════════════════════
# Distributed helpers
# ══════════════════════════════════════════════════════════════

def setup_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def all_reduce_gradients(model, world_size):
    """DDP-style: sum gradients across ranks, divide by world_size -> mean gradient.
    Called ONCE after the local per-sample backward loop (not per backward call),
    so the per-sample grad-accumulation pattern is preserved and we sync only once."""
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size

# ══════════════════════════════════════════════════════════════
# Generation (per-rank, on its own GPU) - same logic as V2
# ══════════════════════════════════════════════════════════════

def generate_batch(model, tokenizer, prompts, device):
    model.eval()
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    all_texts, all_old_lps, all_resp_ids, all_plens = [], [], [], []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=MAX_PROMPT_LEN).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        all_plens.append(prompt_len)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_RESPONSE_LEN,
                do_sample=True, temperature=0.7, top_p=0.9,
                num_return_sequences=GROUP_SIZE,
                pad_token_id=pad_token_id,
                return_dict_in_generate=True,
            )
        sequences = outputs.sequences

        group_texts, group_resp_ids = [], []
        for g in range(GROUP_SIZE):
            resp_ids = sequences[g, prompt_len:]
            non_pad = (resp_ids != pad_token_id)
            if non_pad.any():
                last = non_pad.nonzero()[-1].item() + 1
                resp_ids = resp_ids[:last]
            group_texts.append(tokenizer.decode(resp_ids, skip_special_tokens=True))
            group_resp_ids.append(resp_ids)

        with torch.no_grad():
            full_logits = model(sequences, use_cache=False).logits

        group_old_lps = []
        for g in range(GROUP_SIZE):
            resp_ids = group_resp_ids[g]
            resp_len = len(resp_ids)
            if resp_len == 0:
                group_old_lps.append(torch.tensor([], device=device))
                continue
            pred_logits = full_logits[g, prompt_len-1:prompt_len+resp_len-1, :]
            lp = F.log_softmax(pred_logits.float(), dim=-1)
            tlp = lp.gather(1, resp_ids[:resp_len].unsqueeze(1)).squeeze(1)
            group_old_lps.append(tlp)

        all_texts.append(group_texts)
        all_old_lps.append(group_old_lps)
        all_resp_ids.append(group_resp_ids)
        del full_logits, outputs, sequences

    torch.cuda.empty_cache()
    return all_texts, all_old_lps, all_resp_ids, all_plens

def compute_grpo_advantages(rewards_per_prompt):
    advantages = []
    for gr in rewards_per_prompt:
        t = torch.tensor(gr, dtype=torch.float32)
        mean, std = t.mean(), t.std()
        adv = torch.zeros_like(t) if std < 1e-8 else (t - mean) / (std + 1e-8)
        advantages.append(adv.tolist())
    return advantages

# ══════════════════════════════════════════════════════════════
# Local gradient computation (per-sample backward, NO step)
# ══════════════════════════════════════════════════════════════

def compute_local_gradients(policy_model, ref_model, tokenizer, optimizer,
                            prompts, response_ids, old_lps, advantages, device):
    """Per-sample backward + grad accumulation into .grad. Does optimizer.zero_grad()
    but NOT step() -- the worker steps after the cross-rank all-reduce."""
    policy_model.train()

    num_valid_planned = sum(
        1 for i in range(len(prompts)) for j in range(GROUP_SIZE)
        if len(response_ids[i][j]) > 0 and len(old_lps[i][j]) > 0
    )

    total_loss = 0.0
    total_kl = 0.0
    num_valid = 0

    optimizer.zero_grad()
    if num_valid_planned == 0:
        return 0.0, 0.0, 0

    for i in range(len(prompts)):
        prompt_ids = tokenizer(
            prompts[i], return_tensors="pt", truncation=True,
            max_length=MAX_PROMPT_LEN
        )["input_ids"][0].to(device)
        prompt_len = len(prompt_ids)

        for j in range(GROUP_SIZE):
            adv = advantages[i][j]
            resp_ids = response_ids[i][j]
            old_lp = old_lps[i][j]
            if len(resp_ids) == 0 or len(old_lp) == 0:
                continue

            resp_len = len(resp_ids)
            full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)

            logits = policy_model(full_ids, use_cache=False).logits[0]
            pred = logits[prompt_len-1:prompt_len+resp_len-1, :]
            new_lp = F.log_softmax(pred.float(), dim=-1).gather(
                1, resp_ids[:resp_len].unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                rlogits = ref_model(full_ids, use_cache=False).logits[0]
                rpred = rlogits[prompt_len-1:prompt_len+resp_len-1, :]
                ref_lp = F.log_softmax(rpred.float(), dim=-1).gather(
                    1, resp_ids[:resp_len].unsqueeze(1)).squeeze(1)

            min_len = min(len(new_lp), len(old_lp), len(ref_lp))
            new_lp = new_lp[:min_len]
            old_lp_a = old_lp[:min_len].detach()
            ref_lp = ref_lp[:min_len]

            ratio = torch.exp(new_lp - old_lp_a)
            adv_t = torch.tensor(adv, device=device, dtype=torch.float32)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            kl = new_lp - ref_lp
            loss = policy_loss + KL_COEFF * kl.mean()
            (loss / num_valid_planned).backward()

            total_loss += loss.item()
            total_kl += kl.mean().item()
            num_valid += 1

    return total_loss / max(num_valid, 1), total_kl / max(num_valid, 1), num_valid

# ══════════════════════════════════════════════════════════════
# Symmetric worker (every rank runs the FULL pipeline)
# ══════════════════════════════════════════════════════════════

def worker(rank, world_size, all_data):
    try:
        setup_distributed(rank, world_size)
        device = f"cuda:{rank}"
        is_main = (rank == 0)
        if is_main:
            print(f"[rank0] V3 data-parallel: {world_size} symmetric actor-learners", flush=True)
        print(f"[rank{rank}] start on {device}, PID={os.getpid()}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        policy_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map=device)
        policy_model = get_peft_model(policy_model, build_lora_config())

        ref_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map=device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            [p for p in policy_model.parameters() if p.requires_grad], lr=LEARNING_RATE)

        # Each rank reads a disjoint slice of prompts; advance by world_size*BATCH each round
        data_idx = rank * BATCH_SIZE
        metrics_log = []

        for round_num in range(NUM_ROUNDS):
            t_round = time.time()

            # --- own batch ---
            batch = [all_data[(data_idx + i) % len(all_data)] for i in range(BATCH_SIZE)]
            data_idx += world_size * BATCH_SIZE
            prompts = [b[0] for b in batch]
            gts = [b[1] for b in batch]

            # --- generate ---
            t0 = time.time()
            resp_texts, old_lps, resp_ids, _ = generate_batch(policy_model, tokenizer, prompts, device)
            t_gen = time.time() - t0

            # --- score + advantages ---
            rewards = [[compute_reward(r, gt) for r in resps]
                       for resps, gt in zip(resp_texts, gts)]
            advantages = compute_grpo_advantages(rewards)

            # --- local gradients ---
            t0 = time.time()
            avg_loss, avg_kl, num_valid = compute_local_gradients(
                policy_model, ref_model, tokenizer, optimizer,
                prompts, resp_ids, old_lps, advantages, device)

            # --- all-reduce gradients (cross-rank sync) ---
            t_ar0 = time.time()
            all_reduce_gradients(policy_model, world_size)
            t_allreduce = time.time() - t_ar0

            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()
            t_train = time.time() - t0

            # --- global reward stats via all-reduce ---
            flat = [r for group in rewards for r in group]
            local_sum = sum(flat)
            local_cnt = len(flat)
            local_correct = sum(1 for r in flat if r >= 1.0)
            stats = torch.tensor([local_sum, local_cnt, local_correct],
                                 dtype=torch.float32, device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_reward = (stats[0] / stats[1]).item()
            global_acc = (stats[2] / stats[1]).item()

            dist.barrier()
            t_round_total = time.time() - t_round
            samples_this_round = world_size * BATCH_SIZE * GROUP_SIZE  # 256

            if is_main:
                m = {
                    "round": round_num + 1,
                    "mean_reward": global_reward,
                    "accuracy": global_acc,
                    "loss": avg_loss,
                    "mean_kl": avg_kl,
                    "t_gen": t_gen,
                    "t_train": t_train,
                    "t_allreduce": t_allreduce,
                    "t_round_total": t_round_total,
                    "samples_this_round": samples_this_round,
                    "throughput_samples_per_sec": samples_this_round / t_round_total,
                }
                metrics_log.append(m)
                print(f"[rank0] round {round_num+1:2d}/{NUM_ROUNDS} | "
                      f"reward={global_reward:.3f} acc={global_acc:.3f} "
                      f"loss={avg_loss:.4f} kl={avg_kl:.4f} | "
                      f"t_gen={t_gen:.1f}s t_train={t_train:.1f}s "
                      f"t_allreduce={t_allreduce:.3f}s t_round={t_round_total:.1f}s | "
                      f"{samples_this_round/t_round_total:.2f} samples/s", flush=True)

                if (round_num + 1) % 2 == 0 or round_num == NUM_ROUNDS - 1:
                    with open(os.path.join(SAVE_DIR, "metrics_checkpoint.json"), "w") as f:
                        json.dump(metrics_log, f, indent=2)

        if is_main:
            with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f:
                json.dump(metrics_log, f, indent=2)
            print(f"[rank0] saved metrics to {os.path.join(SAVE_DIR, 'metrics.json')}", flush=True)
            # Save final LoRA adapter (all ranks are bit-identical; rank 0 saves) for eval
            adapter_path = os.path.join(SAVE_DIR, "adapter")
            policy_model.save_pretrained(adapter_path)
            print(f"[rank0] adapter saved to {adapter_path}", flush=True)

        dist.destroy_process_group()
    except Exception as e:
        print(f"[rank{rank}] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            dist.destroy_process_group()
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    mp.set_start_method("spawn", force=True)

    print("=" * 60)
    print("V3 Data-Parallel GRPO - 4 symmetric actor-learners")
    print("=" * 60)
    print(f"  Every GPU: generate own batch -> train -> all-reduce grads -> step")
    print(f"  Batch {BATCH_SIZE} x Group {GROUP_SIZE} = {BATCH_SIZE*GROUP_SIZE} samples/rank/round")
    print(f"  Effective batch/update: {WORLD_SIZE*BATCH_SIZE*GROUP_SIZE} samples")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(flush=True)

    all_data = load_gsm8k_prompts()
    print(f"  Loaded {len(all_data)} training problems", flush=True)

    procs = []
    for rank in range(WORLD_SIZE):
        p = mp.Process(target=worker, args=(rank, WORLD_SIZE, all_data), name=f"rank{rank}")
        p.start()
        procs.append(p)
        print(f"  rank{rank} PID: {p.pid}", flush=True)

    for p in procs:
        p.join()

    print("[Main] all workers finished.", flush=True)

    metrics_path = os.path.join(SAVE_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        n = len(metrics)
        if n >= 4:
            early = metrics[:max(1, n // 3)]
            late = metrics[-max(1, n // 3):]
            print("\n" + "=" * 60)
            print(f"Training Summary (V3 - {n} rounds)")
            print("=" * 60)
            print(f"  Early reward={sum(x['mean_reward'] for x in early)/len(early):.3f} "
                  f"acc={sum(x['accuracy'] for x in early)/len(early):.3f}")
            print(f"  Late  reward={sum(x['mean_reward'] for x in late)/len(late):.3f} "
                  f"acc={sum(x['accuracy'] for x in late)/len(late):.3f}")
            avg = lambda k: sum(x[k] for x in metrics) / n
            print(f"\n  Avg per round:")
            print(f"    t_gen:        {avg('t_gen'):.1f}s")
            print(f"    t_train:      {avg('t_train'):.1f}s")
            print(f"    t_allreduce:  {avg('t_allreduce'):.3f}s")
            print(f"    t_round:      {avg('t_round_total'):.1f}s")
            print(f"    Throughput:   {avg('throughput_samples_per_sec'):.2f} samples/sec")
            print(f"\n  vs V2 baseline 0.94 samples/sec -> "
                  f"{avg('throughput_samples_per_sec')/0.94:.2f}x")

if __name__ == "__main__":
    main()
