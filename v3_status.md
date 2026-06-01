# V3 Data-Parallel GRPO — Status Log

## Goal

Beat V2's throughput (0.94 samples/sec) by fixing the root inefficiency: **in V2, only 1/4 GPUs work at any instant** (gen phase: 3 generators busy, learner idle 155s; train phase: learner busy, 3 generators idle). Idea (user's): make every GPU a full self-contained GRPO worker — generate + train on its own data shard — then integrate via gradient all-reduce. No fixed roles, no idle.

## Design vs V2

| | V2 (functional split) | V3 (data-parallel) |
|--|----------------------|--------------------|
| Decomposition | by **role** (1 learner + 3 generators) | by **data** (4 identical actor-learners) |
| Each GPU does | one fixed role | full pipeline: generate → score → local grads → all-reduce → step |
| GPU busy at any instant | 1/4 | **4/4** |
| Weight integration | NCCL broadcast (learner → generators) | **NCCL all-reduce** of gradients (DDP-style) |
| Effective batch / update | 64 samples | **256 samples** (4× lower variance) |
| Staleness | up to 1 round | **0** (fully on-policy — each rank trains on what it just generated) |

**Why it avoids the V2i trap:** generation and training stay **separated in time** (all 4 generate → barrier → all 4 train → barrier). No gen+train temporal overlap → no cross-workload CPU contention. Within each phase it's same-workload concurrency, which is mild (see below). And because it's synchronous, **NCCL collectives work again** (all ranks present at the all-reduce) — unlike async V2i/V2f which were forced off NCCL.

Hyperparams identical to V2: BATCH=8, GROUP=8, MAX_RESPONSE_LEN=256, lr=5e-6, KL=0.1, CLIP=0.2, LORA r=64. 10 rounds.

## Result: +50% steady-state, +37% average — goal met

| Metric | V2 | V3 (avg) | V3 (steady, rounds 6-10) |
|--------|-----|----------|--------------------------|
| **Throughput** | **0.94** | **1.28** | **1.41 samples/sec** |
| **vs V2** | 1.00× | **1.37× (+37%)** | **1.50× (+50%)** |
| t_round | 204s | 201s | 181s |
| t_gen | 152s (3 gen) | — | 160s (4 gen) |
| t_train | 49s (1 learner, serial) | — | **17s (4-way DDP, parallel)** |
| weight sync | 0.02s (broadcast) | — | 0.042s (all-reduce) |
| GPU busy | 1/4 | — | **4/4** |
| effective batch | 64 | — | **256** |
| reward (valid?) | ~0.70 ✅ | 0.54–0.70 ✅ | ✅ |
| peak mem / GPU | ~6 GB | — | **11 GB / 16 GB** (full footprint ×4, fits) |

### Where the +50% comes from (decomposition)

| phase | V2 | V3 | Δ |
|-------|-----|-----|---|
| Generation | 152s (3 gen) | 160s (4 gen) | **+8s** — mild 4-way contention (only +6%, NOT the 2× of V2i) |
| Training | 49s (1 learner, 3 sequential updates) | 17s (4 GPUs DDP-parallel) | **−32s** — the main win |
| Samples / round | 192 | 256 | **+33% more data** |

Net: training parallelism (49→17s) + 33% more data, partially offset by mild gen contention → **+50%**.

### Two findings worth highlighting

1. **Same-workload concurrency is cheap; cross-workload concurrency is what killed V2i.** 4 concurrent generators run at 160s vs V2's 3 at 152s — only +6%. This confirms the thesis: V2i's 2× slowdown came specifically from gen+train *different* workloads fighting for CPU. Putting 4 *same* workloads side by side barely contends. So **spatial filling (V3) works where temporal overlap (V2i) failed.**

2. **NCCL all-reduce is essentially free in a synchronous design — 0.042s.** Same as V2's broadcast. The async versions were forced onto slow mp.Queue/shm (1.26s / 0.21s) *only because* async can't satisfy NCCL's "all ranks present" requirement. Going back to synchronous data-parallel recovers NCCL's speed.

### Warmup vs steady-state

Rounds 1–5 averaged ~1.16 samples/s (t_round ~220s); rounds 6–10 jumped to ~1.41 (t_round ~181s). The ~40s/round difference is barrier-wait from cross-rank straggling in early rounds (response lengths vary across ranks → slowest rank gates the barrier); as the policy settles, ranks balance and the barrier wait vanishes. Round 1 additionally paid a 3.3s all-reduce warmup (NCCL channel setup) + inflated t_gen (210s, CUDA kernel compile). Steady-state 1.41 is the representative number.

## Correctness notes

- **Fully on-policy, staleness 0**: each rank generates with the synced weights, then trains on that same data before the step. ratio ≈ 1 (modulo LoRA dropout), consistent with V1a/V2.
- **GRPO-compatible**: advantages are group-normalized *within* each rank's own groups (8 responses/prompt) — no cross-rank dependency. Each rank has complete groups.
- **All 4 ranks stay bit-identical**: gradient all-reduce (sum/world_size) + same optimizer.step() → no model divergence.
- **Reward valid and comparable to V2** (~0.6, not failing) — unlike V1b/V2i where downsizing/staleness distorted it.

## Hard ceiling

V3 reaches ~1.41; the wall is generation's 160s/round (still serial-phased, not overlapped with training). Theoretical ceiling if training were free: 256/160 = **1.60 samples/s**. To break past that you must attack generation itself (more GPUs for more generators, or vLLM to make generation faster + CPU-light). Within 4×V100 + HF generate, ~1.4–1.6 is the ceiling, and V3 is close to it.

## Files

- Script: `/data/project/v3_dataparallel.py`
- Log: `/data/project/v3_results/run.log`
- Metrics: `/data/project/v3_results/metrics.json` (10 rounds)
- This status: `/data/project/v3_status.md`

## Bottom line

> V3 makes all 4 GPUs symmetric full GRPO workers (generate + train own shard, integrate via gradient all-reduce). It solves V2's "1/4 GPUs busy" problem → 4/4 busy, hits **1.41 samples/sec steady (1.50× V2)**, keeps reward valid, and confirms two theses: (1) same-workload concurrency is cheap so spatial filling beats V2i's temporal overlap, (2) synchronous design recovers NCCL's near-free sync. This is the cleanest, highest-throughput design in the series without changing the generation engine.
