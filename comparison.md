# GRPO V0 → V1a → V1b → V2 → (V2i/V2f) → V3 — Comparison

All runs train Qwen2.5-1.5B-Instruct + LoRA (r=64) on GSM8K with the same rule-based reward function (`+1.0` for correct numerical answer, `+0.2` for `\boxed{}` format). What changes is the *system architecture* and (in V0 only) the loss formulation.

## What each version is

| | Architecture | Algorithm | Hyperparams | Purpose |
|--|-------------|-----------|-------------|---------|
| **V0** | 1 process, 1 GPU, serial gen | Sum-then-backward (BROKEN) | b=4, g=4, M=256 | Establish a baseline + expose the loss bug |
| **V1a** | 1 process, 1 GPU, batched gen | Proper clipped-surrogate + per-token mean | b=8, g=8, M=256 | Fix algorithm + batch generation in one step |
| **V1b** | 2 processes (Gen + Learner), same GPU | Same as V1a | b=4, g=4, M=128 (downsized) | Measure single-GPU multi-process contention |
| **V2** | 4 proc (1 Learner + 3 Gen), 4 GPUs, **sync** rounds, NCCL broadcast | Same as V1a | b=8, g=8, M=256 | Pipeline parallelism across GPUs |
| **V2i** | V2 made **async** (free-run gen + streaming learner), mp.Queue weights | Same as V1a | b=8, g=8, M=256 | Try to overlap gen+train, kill learner idle |
| **V2f** | V2i with fixed IPC (raw mp.Queue + /dev/shm weights) | Same as V1a | b=8, g=8, M=256 | Controlled test: is the regression IPC's fault? |
| **V3** | 4 proc, **data-parallel** (4 symmetric actor-learners), grad all-reduce | Same as V1a | b=8, g=8, M=256 | Every GPU does everything; no idle |

`b`=BATCH_SIZE (prompts/step), `g`=GROUP_SIZE (responses/prompt), `M`=MAX_RESPONSE_LEN.

## Headline numbers

| Metric | V0 | V1a | V1b† | V2 | V2i | V2f | **V3** |
|--------|----|-----|------|-----|-----|-----|--------|
| **Throughput (samples/sec)** | **0.064** | **0.33** | **0.37** | **0.94** | 0.64 | 0.66 | **1.41**§ |
| **Speedup vs V0** | 1× | 5.2× | 5.8× | 14.7× | 10× | 10.3× | **22×** |
| vs V2 | — | — | — | 1.00× | 0.68× | 0.70× | **1.50×** |
| GPUs used | 1 | 1 | 1 | 4 | 4 | 4 | 4 |
| GPU busy at any instant | 1/1 | 1/1 | 1/1 (shared) | 1/4 | ~all (but contended) | ~all (contended) | **4/4** |
| Peak GPU mem | ~13 GB | ~11 GB | ~16 GB (OOM) | ~6 GB | ~6 GB | ~6 GB | ~11 GB ×4 |
| Effective batch / update | 16 | 64 | 16 | 64 | 64 | 64 | **256** |
| Weight sync | — | — | mp.Queue | 0.02s NCCL | 1.26s mp.Q | 0.21s shm | **0.04s NCCL** |
| reward valid? | ❌ broken | ✅ | ❌ distorted | ✅ | ✅ | ✅ | ✅ |

† V1b numbers caveated (downsized config distorts reward) — see "V1b reward caveat".
§ V3 steady-state (rounds 6-10); 10-round average is 1.28. V2i/V2f are async attempts that *regressed* — see V2i/V2f section.

## Per-step timing breakdown

| Phase | V0 | V1a | V1b | V2 (amortized) |
|-------|----|-----|-----|----------------|
| Generation | 244 s (99%) | 175 s (90%) | 22 s (50%) | 152 s wall (parallel, 3×) |
| Scoring | 0.001 s (0%) | 0.001 s (0%) | 0.000 s (0%) | 0.001 s (0%) |
| Training (fwd+bwd+opt) | 4 s (1%) | 20 s (10%) | 7.5 s (17%) | 16.3 s (24%) |
| Queue wait (Learner idle) | n/a | n/a | 14 s (32%) | 51.7 s (76%) |
| Weight sync | n/a | n/a | 0.24 s (1%) | **0.02 s (~0%)** |

## GPU utilization (rough)

Estimated from architecture and per-phase timing (no nvprof traces collected; numbers are bracketed estimates).

| Run | cuda:0 | cuda:1 | cuda:2 | cuda:3 | Aggregate system busy |
|-----|--------|--------|--------|--------|------------------------|
| V0 | ~95–99% during gen, idle during the 1% train phase → **~95%** | idle | idle | idle | ~24% (1 of 4 GPUs near-saturated) |
| V1a | gen dominates (90%) + train (10%) busy → **~95%** | idle | idle | idle | ~24% |
| V1b | shared by 2 processes; both active most of the round → **~95% on cuda:0** | idle | idle | idle | ~24% |
| V2 | training busy ~24% of round time; idle 76% waiting → **~24%** | gen busy 152s of 204s round → **~75%** | **~75%** | **~75%** | **~62%** (3 of 4 GPUs near-saturated) |

So V2 didn't max out the cluster (~62% aggregate vs V1a's ~24%), but it converted single-GPU saturation into multi-GPU parallel work, and the *throughput* gain (2.85× vs V1a) maps directly to the parallel generators.

## Why V2 isn't 4× V1a (only 2.85×)

The Learner pulls 3 rollouts before training, so it sits at `data_queue.get()` for 52 s/step (76 % of amortized step time) waiting for Generators to finish. The round structure is:

```
t=0    : 3 generators start generating  (parallel on cuda:1/2/3)
t=152s : first rollout arrives → learner can start training
t=152s : learner trains rollout 1 (~16s)
t=168s : learner trains rollout 2 (~16s)
t=184s : learner trains rollout 3 (~16s)
t=200s : NCCL broadcast (~0.02s) → round end
```

Learner is idle from t=0 to t=152s waiting for the first rollout. Reclaiming that idle time would require **interleaving**: train on rollout 1 while generators are still producing rollouts 2 and 3. That moves V2 closer to a 4× theoretical limit.

## Reward / accuracy trajectories

| | Early (first ~10 steps) | Late (last ~10 steps) | Trend |
|--|------------------------|----------------------|-------|
| V0 (steps 1-20 vs 30-50) | reward 0.529, acc 0.53 | reward 0.342, acc 0.34 | **Monotone decline** — broken loss eroded the strong base |
| V1a (steps 1-20 vs last 20) | reward 0.705, acc 0.541 | reward 0.695, acc 0.517 | **Flat** — KL anchor at 0.1 held, but no signal to climb |
| V1b† | reward 0.227, acc 0.188 | reward 0.180, acc 0.150 | **Flat at low baseline** (see caveat) |
| V2 (1-10 vs last 10) | reward 0.703, acc 0.561 | reward 0.656, acc 0.478 | **Flat with slight late dip** — within batch=8 noise |

### V1b reward caveat — IMPORTANT

V1b's reward/accuracy numbers are **not directly comparable** to the others. To fit two processes on a single 16 GB V100, V1b's attempt-3 config:
- Dropped LoRA from the Generator → Generator stays at the base model for the whole run; Learner's updates never make it back to the rollouts.
- Cut `MAX_RESPONSE_LEN` 256→128 → most GSM8K solutions don't fit, get truncated before `\boxed{}`, lose the +1.0 correctness bonus.

So V1b's reward floor sits at 0.18–0.23 because of truncation + frozen Generator, not because the algorithm is failing. V1b's *throughput and contention* numbers (the timing table above) are the actual measurement and ARE meaningful.

## Key findings

1. **Algorithm matters more than parallelism on the wrong baseline.** V0's broken loss (sum-then-backward, no importance ratio, KL too soft) made the policy degrade no matter how many GPUs we threw at it. V1a fixed it at zero extra hardware cost and got 5× speedup + stable training.

2. **Single-GPU multi-process is dominated by memory contention.** On a 16 GB V100, fitting both a generator-process and a backward-pass learner-process with two full 1.5B model copies needs aggressive compromises: gradient checkpointing on the Learner, no LoRA on the Generator (which breaks the on-policy loop), and a 50% cut in response length. Net: V1b runs but you sacrifice the learning loop to do it.

3. **NCCL broadcast is essentially free.** Across 10 rounds of broadcasting ~73 M LoRA params on 4 V100s with NVLink-style P2P, the average sync took **20 ms** — three orders of magnitude smaller than any other phase. Weight synchronization is **not** the bottleneck for pipeline GRPO.

4. **V2's bottleneck is Learner idle time, not gen or sync.** 76 % of V2's amortized step time is Learner waiting for Generators. With async/interleaved training (train on rollout-k while rollout-k+1 generates), the same hardware could approach 4× theoretical speedup over V1a.

5. **Reward signal is dominated by sample noise across all runs.** With batch=8 and a strong starting model (~70 % GSM8K cold), the σ per step is ~0.15–0.20 and the expected per-step Δ is ~0.005. None of the runs ran long enough or large enough to see real lift; that's a sample-budget problem, not a system problem.

## What to do next

- **Interleave gen and train in V2** to reclaim the 52 s/step Learner idle. Expected: 1.5–2× over current V2.
- **Bigger total sample budget** to push past the noise floor — either more steps or substantially larger batches.
- **K-epoch updates** to make the importance ratio + clipping actually do something (with K=1 the ratio is always ≈1.0 and the clip is a no-op).
- **vLLM for generation** in a V3 — Generators currently run HF `model.generate` which is the slowest part by far.

## File pointers

| Run | Code | Metrics | Log |
|-----|------|---------|-----|
| V0 | `/data/project/v0_serial.py` | `/data/project/v0_results/metrics.json` (incomplete; data in log) | `/data/project/v0_results/run.log` |
| V1a | `/data/project/v1a_serial.py` | `/data/project/v1a_results/metrics.json` | `/data/project/v1a_results/run.log` |
| V1b | `/data/project/v1b_multiproc.py` | `/data/project/v1b_results/metrics.json` (20 steps; checkpoint copy) | `/data/project/v1b_results/run.log` |
| V2 | `/data/project/v2_pipeline.py` | `/data/project/v2_results/metrics.json` (30 steps) | `/data/project/v2_results/run.log` |
