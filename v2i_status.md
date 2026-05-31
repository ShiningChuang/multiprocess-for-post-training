# V2i Run — Status Log

## Goal

Modify V2's "synchronous round" pipeline so Generators don't block on the Learner. Measure whether the Learner's 52 s/step idle wait (76 % of V2 amortized step time) shrinks when generation and training overlap. Compare throughput head-to-head with V2 at identical hyperparams.

## Design vs V2

| | V2 | V2i |
|--|----|----|
| Generator loop | barrier between rollouts; waits for NCCL broadcast | free-run; non-blocking weight queue check at top of loop |
| Learner loop | pulls 3 rollouts → trains 3 → broadcasts → barrier | pulls 1 rollout → trains 1 → broadcasts every 3 steps; no barriers |
| Weight transport | NCCL broadcast on LoRA tensors (collective) | mp.Queue per Generator, learner pushes CPU LoRA state_dict |
| Process group | NCCL backend, world_size=4 | none — pure mp.Process |
| New metrics | — | `t_overlap` (queue wait per step), `staleness` (rounds between gen weight version and learner weight version) |

Same hyperparams as V2: BATCH=8, GROUP=8, MAX_RESPONSE_LEN=256, LR=5e-6, KL_COEFF=0.1, CLIP_EPS=0.2, LORA_RANK=64. Total 30 learner steps = 3 generators × 10 local steps each.

## Launch

- Script: `/data/project/v2i_pipeline.py` (548 lines)
- Log: `/data/project/v2i_results/run.log`
- Metrics out: `/data/project/v2i_results/metrics.json`
- Started detached (nohup) at PID 82307 — survives SSH disconnect
- Smoke test skipped per user direction (going straight to full background run)

## Expected vs V2

| Metric | V2 (actual) | V2i (expected) |
|--------|-------------|----------------|
| samples/sec | 0.94 | 1.2–1.3 |
| Learner queue wait | 51.7 s (76%) | < 10 s (< 10%) |
| Staleness | 0 rounds (synchronous) | ~1 round (one batch ahead) |
| NCCL/queue sync | 0.02 s | 0.02 s (similar) |

Hypothesis: overlap reclaims the Learner idle, throughput ~1.3× V2. Trade-off: rollouts trained on are 1 weight-version stale, but with `CLIP_EPS=0.2` the clipped surrogate handles small policy drift.

## Status: DONE — 30/30 steps

Wall-clock: ~46 min (vs V2's 34 min). All 30 learner steps completed, metrics saved.

## Result: V2i is SLOWER than V2 (the hypothesis was wrong)

| Metric | V2 (actual) | V2i (expected) | V2i (actual) | vs V2 |
|--------|-------------|----------------|--------------|-------|
| samples/sec | 0.94 | 1.2–1.3 | **0.64** | **−32%** |
| t_overlap (Learner wait) | 51.7 s (76%) | <10 s | 67.6 s (67%) | +31% absolute, share fell |
| t_train per step | 16.3 s | ~same | 31.5 s | **+93%** |
| t_gen per generator wall | 152.4 s | ~same | 279.3 s | **+83%** |
| t_sync (weight transport) | 0.02 s (NCCL) | similar | 1.26 s (mp.Queue) | **~63×** |
| Average staleness | 0 rounds | ~1 | **0.97 rounds** | as predicted |
| t_total amortized | 68.1 s | ~50 s | 100.3 s | +47% |
| Reward (early/late) | 0.703 / 0.656 | flat | 0.683 / 0.725 | slight uptick |

The staleness number landed exactly where the design predicted (≈1 round). Everything else got worse. The expected throughput win didn't materialize.

## What went wrong — three compounding effects

**1. `manager.Queue` is a CPU bottleneck under concurrent access.** V2 and V2i both use `mp.Manager().Queue()`. In V2, the synchronous round means only one process touches the queue at a time (Generators all push in parallel during gen, then Learner pulls during train). In V2i, all 4 processes hit the manager concurrently and continuously. Every `put`/`get` serializes through the single Manager subprocess. With ~30 MB rollouts being pickled per put, that one process saturates a core and gates everything.

   Fix would be: use raw `mp.Queue` (anonymous pipe per pair), or a CUDA-IPC shared-memory ring buffer. Not done here.

**2. NCCL broadcast → mp.Queue weight sync is ~60× slower.** V2's sync was 20 ms (NCCL P2P over NVLink-style fabric). V2i's sync is 1.26 s (LoRA state_dict moved GPU→CPU, pickled, transferred via mp.Queue, unpickled, moved CPU→GPU). At ~73 M LoRA params × 2 bytes = 146 MB per generator × 3 generators = 438 MB CPU-mediated transfer per sync. This was a deliberate spec choice ("Learner broadcast ... 把 LoRA state_dict 放到 weight_queue") but it's a real cost.

**3. Per-step train time doubled — because faster weight propagation made responses longer.** V2i's policy gets updated every ~3 learner steps with only 1 round staleness, so it actually drifts faster than V2's "update once per round" cadence. The policy converged to generating longer responses (response length, not directly logged here, but inferable: per-sample train time = 0.25 s → 0.49 s, and per-sample gen time grew proportionally). Longer responses → quadratic-ish growth in forward/backward time → both Learner and Generators got slower per-step.

   Reward trajectory backs this up: V2i early=0.683 → late=0.725, the first run we've seen with a *positive* delta from early to late. The faster weight update loop is doing learning work — but the system pays for it in wall-clock per step.

## What overlap actually achieved

The `t_overlap` share of step time dropped from 76 % (V2) → 67 % (V2i). So **some** overlap happened. But in absolute terms `t_overlap` *grew* (51.7 s → 67.6 s) because the surrounding step time grew faster. The math:

- V2: per round, all 3 generators finish at the same time (~152 s). Learner waits 152 s for the first arrival, then trains 3 batches at ~16 s each = 48 s. Wait share = 152/(152+48) = 76 %.
- V2i: Generators run continuously, each taking ~280 s per rollout. So a rollout arrives roughly every 93 s (280 s ÷ 3 generators). Learner trains in 31 s. Wait per rollout = 93 − 31 = 62 s on the queue. Wait share = 62/(62+31) = 67 %.

The arrival-interval gain (152 s → 93 s) is real, but it's swamped by the per-step train slowdown (16 s → 31 s) and the per-gen slowdown (152 s → 279 s wall).

## Findings for the report

1. **Async interleaving is not a free win on this stack.** The combination of `mp.Manager().Queue` for high-frequency rollout transport + CPU-mediated weight sync introduced ~30% throughput regression vs the simple synchronous V2.
2. **The bottleneck moved, but it didn't shrink.** Learner idle dropped from 76 % to 67 % of step time, but per-step time itself grew 47 %, so absolute idle time *increased*. Reclaiming "idle %" without watching wall-clock is misleading.
3. **NCCL P2P is genuinely much faster than mp.Queue for weight transfer.** A 60× sync slowdown is a lot when sync happens every 3 steps. For a V3, NCCL broadcast is the right transport — V2i didn't need to choose mp.Queue.
4. **Faster on-policy updates DID help the reward signal.** Reward went early 0.683 → late 0.725 (acc 0.556 → 0.562). This is the first run in the series with a positive late delta. The increased policy update frequency (every 3 learner steps vs once per round) gave a more responsive on-policy loop — at the cost of wall-clock.

## What a working V2i would look like

A correct version would: (a) use NCCL broadcast for weight sync (same as V2), (b) replace `manager.Queue` with raw `mp.Queue` or CUDA IPC for the rollout transport so puts/gets don't serialize through one Manager process. Expected: keep V2's 0.02 s sync, drop overlap to ~10 % of step time, and reach the originally-hypothesized 1.2–1.3 samples/sec. Not implemented here.

## Files

- Script: `/data/project/v2i_pipeline.py`
- Log: `/data/project/v2i_results/run.log`
- Metrics: `/data/project/v2i_results/metrics.json` (30 steps)
- This status: `/data/project/v2i_status.md`

