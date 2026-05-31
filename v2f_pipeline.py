#!/usr/bin/env python3
"""
V2f Fixed Async Pipeline GRPO - same async architecture as V2i,
with the two IPC layers swapped out:

  rollout transport: mp.Manager().Queue()     →  torch.multiprocessing.Queue (raw)
  weight sync:       mp.Queue with pickle     →  /dev/shm file + atomic rename + mp.Value version counter

Everything else (free-running Generators, streaming Learner, sync every 3 steps,
hyperparams, GRPO loss) is identical to V2i.

The point: the V2i throughput regression came from the IPC layer, not from
async-ness. V2f tests whether fixing those two layers recovers the expected
throughput while keeping V2i's overlap behavior.
"""

import os
import re
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
# Config (identical to V2/V2i)
# ══════════════════════════════════════════════════════════════
MODEL_PATH = "/data/models/Qwen2.5-1.5B-Instruct"
NUM_GENERATORS = 3
NUM_TRAIN_STEPS = 30
GEN_STEPS_PER_WORKER = NUM_TRAIN_STEPS // NUM_GENERATORS  # 10
BATCH_SIZE = 8
GROUP_SIZE = 8
MAX_PROMPT_LEN = 256
MAX_RESPONSE_LEN = 256
LEARNING_RATE = 5e-6
KL_COEFF = 0.1
CLIP_EPS = 0.2
LORA_RANK = 64
LORA_ALPHA = 128
LOG_EVERY = 1
SYNC_EVERY_N_STEPS = 3
SAVE_DIR = "/data/project/v2f_results"

# Shared-memory weight transport
SHM_DIR = "/dev/shm"
WEIGHT_FILE = os.path.join(SHM_DIR, "v2f_lora_weights.pt")
WEIGHT_FILE_TMP = os.path.join(SHM_DIR, "v2f_lora_weights.tmp.pt")

# ══════════════════════════════════════════════════════════════
# Reward / data (same as V2/V2i)
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
# Weight sync via /dev/shm
# ══════════════════════════════════════════════════════════════

def save_weights_to_shm(policy_model, version_counter):
    """Learner-side. Atomic via tmp + os.replace."""
    t0 = time.time()
    lora_state = {
        k: v.cpu() for k, v in policy_model.state_dict().items()
        if "lora_" in k
    }
    torch.save(lora_state, WEIGHT_FILE_TMP)
    os.replace(WEIGHT_FILE_TMP, WEIGHT_FILE)
    with version_counter.get_lock():
        version_counter.value += 1
    return time.time() - t0

def maybe_load_weights_from_shm(model, version_counter, current_version, device):
    """Generator-side. Returns (new_version, t_load_seconds_if_loaded)."""
    new_version = version_counter.value
    if new_version <= current_version:
        return current_version, 0.0
    t0 = time.time()
    try:
        new_state = torch.load(WEIGHT_FILE, map_location=device, weights_only=True)
    except Exception as e:
        # File might be mid-rename or unreadable; skip this update
        print(f"[Generator] weight load skipped: {e}", flush=True)
        return current_version, 0.0
    model.load_state_dict(new_state, strict=False)
    return new_version, time.time() - t0

# ══════════════════════════════════════════════════════════════
# Generator worker
# ══════════════════════════════════════════════════════════════

def generator_worker(gen_id, device_idx, data_queue, version_counter, all_data):
    try:
        device = f"cuda:{device_idx}"
        torch.cuda.set_device(device_idx)
        print(f"[Generator-{gen_id}] start on {device}, PID={os.getpid()}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map=device
        )
        model = get_peft_model(model, build_lora_config())
        model.eval()

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        data_idx = (gen_id - 1) * BATCH_SIZE
        current_weight_version = 0

        for local_step in range(GEN_STEPS_PER_WORKER):
            # Non-blocking weight check
            current_weight_version, t_wload = maybe_load_weights_from_shm(
                model, version_counter, current_weight_version, device
            )

            # === Generate batch ===
            batch_data = []
            for _ in range(BATCH_SIZE):
                batch_data.append(all_data[data_idx % len(all_data)])
                data_idx += NUM_GENERATORS
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
                "gen_id": gen_id,
                "gen_local_step": local_step + 1,
                "weight_version": current_weight_version,
                "prompts": prompts,
                "response_ids": all_response_ids,
                "old_log_probs": all_old_log_probs,
                "advantages": advantages,
                "rewards": rewards,
                "t_gen": t_gen,
                "t_score": t_score,
                "t_wload": t_wload,
            }
            data_queue.put(rollout)
            print(f"[Generator-{gen_id}] step {local_step+1}/{GEN_STEPS_PER_WORKER}: "
                  f"gen={t_gen:.1f}s wver={current_weight_version} wload={t_wload:.2f}s",
                  flush=True)

        data_queue.put(None)
        print(f"[Generator-{gen_id}] done.", flush=True)
    except Exception as e:
        print(f"[Generator-{gen_id}] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            data_queue.put(None)
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════
# Learner worker
# ══════════════════════════════════════════════════════════════

def train_one_step(policy_model, ref_model, tokenizer, optimizer, rollout, device):
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

            outputs = policy_model(full_ids, use_cache=False)
            logits = outputs.logits[0]
            pred_logits = logits[prompt_len-1:prompt_len+resp_len-1, :]
            log_probs = F.log_softmax(pred_logits.float(), dim=-1)
            new_lp = log_probs.gather(1, resp_ids[:resp_len].unsqueeze(1)).squeeze(1)

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

def learner_worker(data_queue, version_counter):
    try:
        device = "cuda:0"
        torch.cuda.set_device(0)
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

        metrics_log = []
        global_step = 0
        nones_received = 0

        while True:
            t_overlap_start = time.time()
            rollout = data_queue.get()
            t_overlap = time.time() - t_overlap_start

            if rollout is None:
                nones_received += 1
                print(f"[Learner] stop signal {nones_received}/{NUM_GENERATORS}", flush=True)
                if nones_received >= NUM_GENERATORS:
                    break
                continue

            global_step += 1
            staleness = version_counter.value - rollout["weight_version"]

            t_train_start = time.time()
            m = train_one_step(policy_model, ref_model, tokenizer, optimizer, rollout, device)
            t_train = time.time() - t_train_start

            t_sync = 0.0
            if global_step % SYNC_EVERY_N_STEPS == 0:
                t_sync = save_weights_to_shm(policy_model, version_counter)

            t_total = t_overlap + t_train + t_sync
            m.update({
                "step": global_step,
                "gen_id": rollout["gen_id"],
                "weight_version_used": rollout["weight_version"],
                "learner_weight_version": version_counter.value,
                "staleness": staleness,
                "t_gen": rollout["t_gen"],
                "t_score": rollout["t_score"],
                "t_wload_on_gen": rollout.get("t_wload", 0.0),
                "t_train": t_train,
                "t_overlap": t_overlap,
                "t_sync": t_sync,
                "t_total": t_total,
            })
            metrics_log.append(m)

            if global_step % LOG_EVERY == 0 or global_step == 1:
                print(f"[Learner] step {global_step:3d} (gen {rollout['gen_id']}) | "
                      f"reward={m['mean_reward']:.3f} acc={m['accuracy']:.3f} "
                      f"loss={m['loss']:.4f} kl={m['mean_kl']:.4f} | "
                      f"t_overlap={t_overlap:.1f}s t_train={t_train:.1f}s t_sync={t_sync:.2f}s | "
                      f"stale={staleness} (lver={version_counter.value})",
                      flush=True)

            if global_step % 10 == 0:
                ckpt = os.path.join(SAVE_DIR, "metrics_checkpoint.json")
                with open(ckpt, "w") as f:
                    json.dump(metrics_log, f, indent=2)

        metrics_path = os.path.join(SAVE_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_log, f, indent=2)
        print(f"[Learner] saved metrics to {metrics_path}", flush=True)
    except Exception as e:
        print(f"[Learner] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

_child_procs = []
def _sigterm_handler(signum, frame):
    print(f"[Main] signal {signum}, terminating children...", flush=True)
    for p in _child_procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:
            pass

def cleanup_shm():
    for f in (WEIGHT_FILE, WEIGHT_FILE_TMP):
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    cleanup_shm()
    mp.set_start_method("spawn", force=True)
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    print("=" * 60)
    print("V2f Fixed Async Pipeline GRPO - 4-GPU")
    print("=" * 60)
    print(f"  Rollout transport: raw mp.Queue (no Manager)")
    print(f"  Weight sync:       /dev/shm + atomic rename + mp.Value version")
    print(f"  Learner cuda:0 | Generators cuda:1,2,3")
    print(f"  Batch {BATCH_SIZE} x Group {GROUP_SIZE} = {BATCH_SIZE*GROUP_SIZE} samples/step")
    print(f"  Total steps: {NUM_TRAIN_STEPS} (each generator: {GEN_STEPS_PER_WORKER})")
    print(f"  Sync every {SYNC_EVERY_N_STEPS} learner steps")
    print(flush=True)

    all_data = load_gsm8k_prompts()
    print(f"  Loaded {len(all_data)} training problems", flush=True)

    # Raw mp.Queue (NO manager). 3 generators × max 2 each.
    data_queue = mp.Queue(maxsize=NUM_GENERATORS * 2)
    version_counter = mp.Value('i', 0)

    learner_proc = mp.Process(target=learner_worker,
                              args=(data_queue, version_counter),
                              name="Learner")
    learner_proc.start()
    _child_procs.append(learner_proc)
    print(f"  Learner PID: {learner_proc.pid}", flush=True)

    time.sleep(2)

    for gid in range(1, NUM_GENERATORS + 1):
        p = mp.Process(target=generator_worker,
                       args=(gid, gid, data_queue, version_counter, all_data),
                       name=f"Generator-{gid}")
        p.start()
        _child_procs.append(p)
        print(f"  Generator-{gid} PID: {p.pid}", flush=True)

    for p in _child_procs:
        p.join()

    print("[Main] all workers finished.", flush=True)
    cleanup_shm()

    metrics_path = os.path.join(SAVE_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        n = len(metrics)
        print("\n" + "=" * 60)
        print(f"Training Summary (V2f — {n} steps)")
        print("=" * 60)
        if n >= 10:
            early = metrics[:10]
            late = metrics[-10:]
            print(f"  Early (1-10):    reward={sum(m['mean_reward'] for m in early)/10:.3f}  "
                  f"acc={sum(m['accuracy'] for m in early)/10:.3f}")
            print(f"  Late  (last 10): reward={sum(m['mean_reward'] for m in late)/10:.3f}  "
                  f"acc={sum(m['accuracy'] for m in late)/10:.3f}")
        avg = lambda k: sum(m[k] for m in metrics) / n
        total = avg("t_total")
        print(f"\n  Avg per learner step:")
        print(f"    t_overlap (queue wait): {avg('t_overlap'):.2f}s ({avg('t_overlap')/total*100:.0f}%)")
        print(f"    t_train:                {avg('t_train'):.2f}s ({avg('t_train')/total*100:.0f}%)")
        print(f"    t_sync (shm save):      {avg('t_sync'):.3f}s ({avg('t_sync')/total*100:.0f}%)")
        print(f"    t_total:                {total:.2f}s")
        print(f"    Throughput:             {BATCH_SIZE * GROUP_SIZE / total:.2f} samples/sec")
        print(f"\n  Avg staleness: {avg('staleness'):.2f} rounds")
        print(f"  Avg t_gen (per gen wall): {avg('t_gen'):.1f}s")
        print(f"  Avg t_wload_on_gen (shm read): {avg('t_wload_on_gen'):.3f}s")

if __name__ == "__main__":
    main()
