# V2f Run — Status Log

## Goal

Test the hypothesis from V2i: "异步架构本身是有效的，瓶颈在 IPC 实现." V2f keeps V2i's async pipeline architecture **verbatim** but swaps the two IPC layers:

| Component | V2i | V2f |
|-----------|-----|-----|
| Rollout transport | `mp.Manager().Queue()` | raw `mp.Queue()` (no Manager subprocess) |
| Weight sync | `mp.Queue` (CPU pickle round-trip) | `/dev/shm` file + atomic rename + `mp.Value` version counter |
| Everything else | — | identical to V2i |

Hyperparams identical to V2/V2i.

## Result: hypothesis was wrong — IPC was not the main culprit

26 / 30 steps completed before the Learner hit the same `resource_sharer FileNotFoundError` we saw in V1b (raw `mp.Queue` tensor-FD cleanup race when a producer process exits before the consumer drains its rollouts). Metrics for the 26 completed steps reconstructed from the log.

### Three-way comparison (the goal of this experiment)

| Metric | V2 (sync) | V2i (async, bad IPC) | V2f (async, fixed IPC) |
|--------|-----------|----------------------|-----------------------|
| **Throughput (samples/sec)** | **0.94** | **0.64** | **0.66** |
| samples/sec vs V2 | 1.00× | 0.68× | 0.70× |
| t_total per step | 68.1 s | 100.3 s | 96.5 s |
| t_overlap (Learner queue wait) | 51.7 s (76%) | 67.6 s (67%) | 63.4 s (66%) |
| t_train per step | 16.3 s | 31.5 s | 32.8 s |
| t_gen per Generator wall | 152.4 s | 279.3 s | 278.3 s |
| **t_sync (weight transport)** | **0.02 s (NCCL)** | **1.26 s (mp.Queue)** | **0.21 s (shm)** |
| t_wload (Generator side, shm read) | — | (counted in t_sync) | 0.33 s |
| Avg staleness | 0 rounds | 0.97 | 1.15 |
| Reward early → late | 0.703 → 0.656 | 0.683 → 0.725 | 0.688 → 0.653 |

### What V2f confirms

**The IPC fixes worked at the layer they targeted:**
- t_sync dropped 6× from V2i (1.26 s → 0.21 s) by going CPU↔/dev/shm instead of through `mp.Queue` pickle. Generator-side load was 0.33 s (146 MB LoRA from tmpfs).
- Raw `mp.Queue` didn't crash on contention — the 4 processes can all push/pull without serializing through a Manager.

**But the throughput problem stayed:**
- t_overlap fell only marginally (67.6 s → 63.4 s, 6 % improvement). The "fix the Manager queue" hypothesis predicted this would drop dramatically; it didn't.
- **t_train and t_gen both stayed at ~2× V2's values** (16 → 33 s training, 152 → 278 s generation). This is the real cost, and it's NOT from the IPC layer.

### Where the lost time actually went

V2f and V2i have the same per-step train and gen times despite completely different IPC stacks. So the IPC explanation for V2i's regression was wrong. What's left:

**The async architecture itself adds CPU contention.** In V2, when Generators are at the NCCL barrier waiting for the Learner, they're parked in a kernel wait — no CPU load. When the Learner is at NCCL waiting for Generators, same. CPU contention is bounded by phase. In V2f/V2i, all 4 Python processes are continuously active: tokenizing, building tensors, decoding, pickling rollouts, applying state_dicts. Each of those ops carries Python/PyTorch dispatch overhead, and with 4 processes pinning 4+ cores, latency grows.

This shows up as:
- Learner's `train_one_step` has 64 sequential forward/backward calls (per-sample backward + grad accumulation). Each call has Python overhead. With CPU contested, that overhead inflates ~2×.
- Generators' `model.generate` similarly relies on per-token Python loops for sampling, scoring, KV-cache management. Same inflation.

Confirming hint: per-step train wall doubled (16 → 33 s) even though Learner's GPU (cuda:0) is exclusive to it — only CPU is shared.

**Other possible contributors I can't rule out without per-process traces:**
- PCIe bandwidth contention (CPU↔GPU transfers happen on a shared PCIe root complex).
- Thermal throttling — 4× V100-SXM2 at sustained 300 W TDP = ~1.2 kW continuous, possible chassis-cooling limit. Synchronous V2 has 25 % duty cycle on Learner so total thermal load is lower.

### Reward trajectory

V2f went 0.688 → 0.653 over 26 steps. Same shape as V2 (slight decline within batch=8 noise). V2i's positive 0.683 → 0.725 trajectory didn't reproduce here — confirms that V2i's "improvement" was likely sample noise on a short run, not signal from faster weight propagation.

## Bug — `resource_sharer` FileNotFoundError

Learner crashed at step 27 with the same multiprocessing error V1b hit:

```
File "torch/multiprocessing/reductions.py", line 541, in rebuild_storage_fd
  fd = df.detach()
File "multiprocessing/resource_sharer.py", line 86, in get_connection
  ...
FileNotFoundError: [Errno 2] No such file or directory
```

This is the known raw-`mp.Queue` + `torch.multiprocessing` issue: producer process exits, its shared-memory file descriptors get reaped by `resource_tracker`, but the queue still holds a pickled reference to them. When the consumer tries to unpickle, the FD is gone.

**The 26 steps captured are intact**; nothing got corrupted, the queue just couldn't be drained further. Fixes for a V3 would be one of:
- `mp.set_sharing_strategy('file_system')` before any tensor moves — uses tmpfs files instead of FDs, more durable.
- Producers `.clone()` tensors to detach from shared memory before putting in queue.
- Producers don't exit until consumer ACKs all rollouts received.

Not implemented for V2f — out of scope and the 26-step data is enough to settle the IPC-vs-architecture question.

## Final answer to the experiment

> Was V2i's regression caused by the IPC implementation (Manager Queue + mp.Queue weight sync)?

**No.** V2f fixed both IPC layers cleanly (raw queue + shm sync) and got a 6× reduction in t_sync, but throughput only recovered from 0.64 → 0.66 samples/sec — still 30 % below V2's 0.94. **The async architecture itself imposes a per-step cost on this stack** (CPU contention from 4 concurrently active Python processes is the most likely root cause), and that cost wipes out the wait-time savings.

The "right" V3 would need to address the CPU side: either fewer Python overheads (cuda graphs, batched forward over the entire group instead of per-sample backward), or fewer processes (one process driving multiple CUDA devices via threads), or proper async at the framework level (vLLM-style sampling server). Not a "swap the queue" job.

## Files

- Script: `/data/project/v2f_pipeline.py`
- Log: `/data/project/v2f_results/run.log` (26 steps)
- Metrics (reconstructed from log): `/data/project/v2f_results/metrics.json`
- This status: `/data/project/v2f_status.md`
