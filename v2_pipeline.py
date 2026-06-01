#!/usr/bin/env python3
"""
V2 Pipeline GRPO - 4-GPU pipeline.
  rank 0 = Learner on cuda:0 (policy + ref + optimizer)
  rank 1-3 = Generators on cuda:1, cuda:2, cuda:3 (each has policy + LoRA)

Communication:
  - Rollout data: mp.Queue (CPU tensors), Generators push, Learner pulls
  - Weight sync: NCCL dist.broadcast on LoRA tensors, src=0

Round structure:
  Each round = 3 parallel Generator rollouts + 3 sequential Learner steps + 1 broadcast.
  NUM_TRAIN_STEPS / NUM_GENERATORS rounds total (e.g., 30/3 = 10 rounds).

Hyperparams match V1a (not V1b's downsized config). V1b was forced to
downsize because of single-GPU memory contention; V2 has 16 GB per process.

Usage: python v2_pipeline.py
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
# Config
# ══════════════════════════════════════════════════════════════
MODEL_PATH = "/data/models/Qwen2.5-1.5B-Instruct"
WORLD_SIZE = 4
NUM_GENERATORS = 3
NUM_TRAIN_STEPS = 30
NUM_ROUNDS = NUM_TRAIN_STEPS // NUM_GENERATORS   # 10
BATCH_SIZE = 8
GROUP_SIZE = 8
MAX_PROMPT_LEN = 256
MAX_RESPONSE_LEN = 256
LEARNING_RATE = 5e-6
KL_COEFF = 0.1
CLIP_EPS = 0.2
LORA_RANK = 64
LORA_ALPHA = 128
LOG_EVERY = 1     # log every learner step
SAVE_DIR = "/data/project/v2_results"

# ══════════════════════════════════════════════════════════════
# GSM8K Reward Function (same as V1a/V1b)
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
    return [(build_prompt(item["question"]), item["answer"]) for item in ds]

# ══════════════════════════════════════════════════════════════
# Distributed setup
# ══════════════════════════════════════════════════════════════

def setup_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def build_lora_config():
    return LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

def broadcast_lora(model, sorted_lora_keys, device):
    """Broadcast LoRA tensors from src=0 to all ranks.
    Both sender and receivers call this; sender's tensors are sent in-place.
    """
    state = model.state_dict()
    for k in sorted_lora_keys:
        t = state[k].to(device).contiguous()
        dist.broadcast(t, src=0)
        # On receivers, copy back into the model
        if dist.get_rank() != 0:
            state[k].copy_(t)
    if dist.get_rank() != 0:
        model.load_state_dict(state, strict=False)

# ══════════════════════════════════════════════════════════════
# Generator worker (rank 1-3)
# ══════════════════════════════════════════════════════════════

def generator_worker(rank, world_size, data_queue, all_data):
    try:
        setup_distributed(rank, world_size)
        device = f"cuda:{rank}"
        print(f"[Generator-{rank}] start on {device}, PID={os.getpid()}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map=device
        )
        model = get_peft_model(model, build_lora_config())
        model.eval()

        sorted_lora_keys = sorted(k for k in model.state_dict().keys() if "lora_" in k)
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        # Stagger which prompts each Generator sees so they don't all generate the same batch.
        # Rank 1 starts at offset 0, rank 2 at BATCH_SIZE, rank 3 at 2*BATCH_SIZE.
        data_idx = (rank - 1) * BATCH_SIZE

        for round_num in range(NUM_ROUNDS):
            t_round = time.time()

            # === Generate batch ===
            batch_data = []
            for _ in range(BATCH_SIZE):
                batch_data.append(all_data[data_idx % len(all_data)])
                data_idx += NUM_GENERATORS  # advance by num_generators to keep ranks disjoint
            prompts = [d[0] for d in batch_data]
            ground_truths = [d[1] for d in batch_data]

            t_gen_start = time.time()
            all_response_texts, all_old_log_probs, all_response_ids = [], [], []

            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                  max_length=MAX_PROMPT_LEN).to(device)
                prompt_len = inputs["input_ids"].shape[1]

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
                    group_resp_ids.append(resp_ids.cpu())

                with torch.no_grad():
                    full_logits = model(sequences, use_cache=False).logits

                group_old_lps = []
                for g in range(GROUP_SIZE):
                    resp_ids = group_resp_ids[g]
                    resp_len = len(resp_ids)
                    if resp_len == 0:
                        group_old_lps.append(torch.tensor([]))
                        continue
                    pred_logits = full_logits[g, prompt_len-1:prompt_len+resp_len-1, :]
                    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
                    token_lps = log_probs.gather(
                        1, resp_ids.to(device)[:resp_len].unsqueeze(1)
                    ).squeeze(1).cpu()
                    group_old_lps.append(token_lps)

                all_response_texts.append(group_texts)
                all_old_log_probs.append(group_old_lps)
                all_response_ids.append(group_resp_ids)

                del full_logits, outputs, sequences
            torch.cuda.empty_cache()
            t_gen = time.time() - t_gen_start

            # === Score (CPU) ===
            t_score_start = time.time()
            rewards = [[compute_reward(r, gt) for r in resps]
                       for resps, gt in zip(all_response_texts, ground_truths)]
            t_score = time.time() - t_score_start

            # === Advantages ===
            advantages = []
            for gr in rewards:
                t = torch.tensor(gr, dtype=torch.float32)
                mean, std = t.mean(), t.std()
                adv = torch.zeros_like(t) if std < 1e-8 else (t - mean) / (std + 1e-8)
                advantages.append(adv.tolist())

            rollout = {
                "round": round_num,
                "gen_rank": rank,
                "prompts": prompts,
                "response_ids": all_response_ids,
                "old_log_probs": all_old_log_probs,
                "advantages": advantages,
                "rewards": rewards,
                "t_gen": t_gen,
                "t_score": t_score,
            }
            data_queue.put(rollout)
            t_put = time.time() - t_round - t_gen - t_score

            # === Wait for weight broadcast from Learner ===
            t_wait_start = time.time()
            broadcast_lora(model, sorted_lora_keys, device)
            t_sync = time.time() - t_wait_start

            dist.barrier()
            t_total = time.time() - t_round
            print(f"[Generator-{rank}] round {round_num+1}/{NUM_ROUNDS}: "
                  f"gen={t_gen:.1f}s sync_wait={t_sync:.1f}s total={t_total:.1f}s", flush=True)

        dist.destroy_process_group()
        print(f"[Generator-{rank}] done.", flush=True)
    except Exception as e:
        print(f"[Generator-{rank}] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            dist.destroy_process_group()
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════
# Learner worker (rank 0)
# ══════════════════════════════════════════════════════════════

def train_one_step(policy_model, ref_model, tokenizer, optimizer, rollout, device):
    """One GRPO optimizer step on a single rollout. Returns metrics dict."""
    prompts = rollout["prompts"]
    response_ids = rollout["response_ids"]
    old_log_probs_list = rollout["old_log_probs"]
    advantages = rollout["advantages"]
    rewards = rollout["rewards"]

    policy_model.train()

    num_valid_planned = sum(
        1 for i in range(len(prompts)) for j in range(GROUP_SIZE)
        if len(response_ids[i][j]) > 0 and len(old_log_probs_list[i][j]) > 0
    )

    total_loss_value = 0.0
    total_kl = 0.0
    num_valid = 0

    if num_valid_planned > 0:
        optimizer.zero_grad()

    for i in range(len(prompts)):
        prompt_ids = tokenizer(
            prompts[i], return_tensors="pt", truncation=True,
            max_length=MAX_PROMPT_LEN
        )["input_ids"][0].to(device)
        prompt_len = len(prompt_ids)

        for j in range(GROUP_SIZE):
            adv = advantages[i][j]
            resp_ids = response_ids[i][j].to(device)
            old_lp = old_log_probs_list[i][j].to(device)

            if len(resp_ids) == 0 or len(old_lp) == 0:
                continue

            resp_len = len(resp_ids)
            full_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)

            # policy fwd (grad)
            outputs = policy_model(full_ids, use_cache=False)
            logits = outputs.logits[0]
            pred_logits = logits[prompt_len-1:prompt_len+resp_len-1, :]
            log_probs = F.log_softmax(pred_logits.float(), dim=-1)
            new_lp = log_probs.gather(1, resp_ids[:resp_len].unsqueeze(1)).squeeze(1)

            # ref fwd (no grad)
            with torch.no_grad():
                ref_out = ref_model(full_ids, use_cache=False)
                ref_logits = ref_out.logits[0]
                ref_pred = ref_logits[prompt_len-1:prompt_len+resp_len-1, :]
                ref_lps = F.log_softmax(ref_pred.float(), dim=-1)
                ref_lp = ref_lps.gather(1, resp_ids[:resp_len].unsqueeze(1)).squeeze(1)

            min_len = min(len(new_lp), len(old_lp), len(ref_lp))
            new_lp = new_lp[:min_len]
            old_lp_a = old_lp[:min_len].detach()
            ref_lp = ref_lp[:min_len]

            ratio = torch.exp(new_lp - old_lp_a)
            adv_t = torch.tensor(adv, device=device, dtype=torch.float32)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

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

    flat_rewards = [r for group in rewards for r in group]
    mean_reward = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0.0
    accuracy = sum(1 for r in flat_rewards if r >= 1.0) / len(flat_rewards) if flat_rewards else 0.0

    return {
        "mean_reward": mean_reward,
        "accuracy": accuracy,
        "mean_kl": total_kl / max(num_valid, 1),
        "loss": avg_loss,
        "num_samples": num_valid,
    }

def learner_worker(rank, world_size, data_queue):
    try:
        setup_distributed(rank, world_size)
        device = f"cuda:{rank}"
        print(f"[Learner] start on {device}, PID={os.getpid()}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        policy_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map=device
        )
        policy_model = get_peft_model(policy_model, build_lora_config())

        ref_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map=device
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            [p for p in policy_model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
        )

        sorted_lora_keys = sorted(k for k in policy_model.state_dict().keys() if "lora_" in k)

        metrics_log = []
        global_step = 0

        for round_num in range(NUM_ROUNDS):
            t_round = time.time()

            # === Consume NUM_GENERATORS rollouts ===
            rollouts = []
            t_wait_start = time.time()
            for _ in range(NUM_GENERATORS):
                rollouts.append(data_queue.get())
            t_wait = time.time() - t_wait_start

            # === Train sequentially on each rollout ===
            t_train_start = time.time()
            per_step_metrics = []
            for rollout in rollouts:
                m = train_one_step(policy_model, ref_model, tokenizer, optimizer, rollout, device)
                global_step += 1
                m["step"] = global_step
                m["round"] = round_num + 1
                m["gen_rank"] = rollout["gen_rank"]
                m["t_gen"] = rollout["t_gen"]
                m["t_score"] = rollout["t_score"]
                per_step_metrics.append(m)
            t_train = time.time() - t_train_start

            # === Broadcast new LoRA weights to all Generators ===
            t_sync_start = time.time()
            broadcast_lora(policy_model, sorted_lora_keys, device)
            t_sync = time.time() - t_sync_start

            t_round_total = time.time() - t_round

            for m in per_step_metrics:
                m["t_wait"] = t_wait / NUM_GENERATORS
                m["t_train"] = t_train / NUM_GENERATORS
                m["t_sync"] = t_sync / NUM_GENERATORS
                m["t_round_total"] = t_round_total
                m["t_step_amortized"] = t_round_total / NUM_GENERATORS
                metrics_log.append(m)
                if m["step"] % LOG_EVERY == 0 or m["step"] == 1:
                    print(f"[Learner] step {m['step']:3d} (round {m['round']:2d}, gen {m['gen_rank']}) | "
                          f"reward={m['mean_reward']:.3f} acc={m['accuracy']:.3f} "
                          f"loss={m['loss']:.4f} kl={m['mean_kl']:.4f} | "
                          f"t_gen={m['t_gen']:.1f}s "
                          f"t_train_amort={m['t_train']:.1f}s "
                          f"t_wait_amort={m['t_wait']:.1f}s "
                          f"t_sync_amort={m['t_sync']:.2f}s",
                          flush=True)

            dist.barrier()

            # Periodic checkpoint
            if (round_num + 1) % 2 == 0 or round_num == NUM_ROUNDS - 1:
                ckpt = os.path.join(SAVE_DIR, "metrics_checkpoint.json")
                with open(ckpt, "w") as f:
                    json.dump(metrics_log, f, indent=2)

        # Final save
        metrics_path = os.path.join(SAVE_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_log, f, indent=2)
        print(f"[Learner] saved metrics to {metrics_path}", flush=True)

        # Save final LoRA adapter (rank 0 / learner) for post-hoc evaluation
        adapter_path = os.path.join(SAVE_DIR, "adapter")
        policy_model.save_pretrained(adapter_path)
        print(f"[Learner] adapter saved to {adapter_path}", flush=True)

        dist.destroy_process_group()
    except Exception as e:
        print(f"[Learner] ERROR: {e}", flush=True)
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
    print("V2 Pipeline GRPO - 4-GPU")
    print("=" * 60)
    print(f"  Learner: cuda:0  |  Generators: cuda:1,2,3")
    print(f"  Batch: {BATCH_SIZE} x {GROUP_SIZE} = {BATCH_SIZE * GROUP_SIZE} samples per gen step")
    print(f"  Rounds: {NUM_ROUNDS} (3 learner steps per round = {NUM_TRAIN_STEPS} total)")
    print(flush=True)

    all_data = load_gsm8k_prompts()
    print(f"  Loaded {len(all_data)} training problems", flush=True)

    # Use Manager queue so it survives across spawned processes cleanly.
    manager = mp.Manager()
    data_queue = manager.Queue(maxsize=NUM_GENERATORS * 2)

    processes = []
    # Learner first
    p0 = mp.Process(target=learner_worker, args=(0, WORLD_SIZE, data_queue), name="Learner")
    p0.start()
    print(f"  Learner PID: {p0.pid}", flush=True)
    processes.append(p0)

    # Stagger generators slightly so they don't all hit init_process_group simultaneously
    time.sleep(2)
    for rank in range(1, WORLD_SIZE):
        p = mp.Process(target=generator_worker, args=(rank, WORLD_SIZE, data_queue, all_data),
                       name=f"Generator-{rank}")
        p.start()
        print(f"  Generator-{rank} PID: {p.pid}", flush=True)
        processes.append(p)

    for p in processes:
        p.join()

    print("[Main] all workers finished.", flush=True)

    # Summary
    metrics_path = os.path.join(SAVE_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        n = len(metrics)
        if n >= 10:
            early = metrics[:10]
            late = metrics[-10:]
            print("\n" + "=" * 60)
            print("Training Summary (V2 - 4-GPU Pipeline)")
            print("=" * 60)
            print(f"  Steps recorded: {n}")
            print(f"  Early (1-10):  mean_reward={sum(m['mean_reward'] for m in early)/len(early):.3f}, "
                  f"acc={sum(m['accuracy'] for m in early)/len(early):.3f}")
            print(f"  Late  (last 10): mean_reward={sum(m['mean_reward'] for m in late)/len(late):.3f}, "
                  f"acc={sum(m['accuracy'] for m in late)/len(late):.3f}")

            avg = lambda key: sum(m[key] for m in metrics) / n
            t_step_amortized = avg("t_step_amortized")
            print(f"\n  Avg amortized per learner step:")
            print(f"    Generation (per-gen wall): {avg('t_gen'):.1f}s")
            print(f"    Queue wait (amortized):    {avg('t_wait'):.1f}s")
            print(f"    Training  (amortized):     {avg('t_train'):.1f}s")
            print(f"    Sync     (amortized):      {avg('t_sync'):.2f}s")
            print(f"    Round total / 3:           {t_step_amortized:.1f}s")
            print(f"    Throughput:                {BATCH_SIZE * GROUP_SIZE / t_step_amortized:.2f} samples/sec")

if __name__ == "__main__":
    main()
