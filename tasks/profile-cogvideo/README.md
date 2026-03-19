# CogVideoX-2B Profiling Task

Profiles the CogVideoX-2B text-to-video inference pipeline on Intel XPU (Arc Pro series) to measure GPU memory consumption and operator-level performance.

## What It Profiles

The script runs three phases:

| Phase | What it does |
|-------|-------------|
| **1 — Peak Memory** | Measures peak GPU memory (allocated & reserved) after each pipeline stage: model load → `pipe.to("xpu")` → inference (denoise + VAE decode) |
| **2 — Profiler Trace** | Captures a `torch.profiler` trace with CPU + XPU activities, exports to Chrome trace format, and prints top-20 operators by XPU time |
| **3 — Summary** | Prints overall peak memory, total wall-clock time, and output file paths |

Uses **4 inference steps** (vs. 50 in production) to keep profiling fast and traces compact.

## How to Run

```bash
cd tasks/profile-cogvideo/
uv sync
uv run python profile_run.py
```

## Output Files

| File | Description |
|------|-------------|
| `profile_trace.json` | Chrome trace — operator-level timeline of the denoising loop |
| `profile_output.mp4` | Video from the Phase 1 inference run (4 steps, for sanity check) |

## Profiling Results

Pre-computed profiling traces are available on Hugging Face:
👉 **https://huggingface.co/Yi30/sageattn-xpu-profiling**

Download `profile_trace.json.gz` and open it in `chrome://tracing` (Chrome handles `.json.gz` natively).

## Viewing the Trace

1. Open Google Chrome
2. Navigate to `chrome://tracing`
3. Click **Load** and select `profile_trace.json` (or `.json.gz`)
4. Use WASD keys to navigate the timeline

The trace shows CPU and XPU operator execution, memory allocations, and kernel launches. Look for:
- **Long XPU kernels** — potential optimization targets
- **CPU-XPU gaps** — data transfer or synchronization overhead
- **Memory spikes** — stages where peak memory occurs

## Expected Output (stdout)

```
========================================================================
Device Diagnostics
========================================================================
  PyTorch version : 2.10.0+xpu
  XPU available   : True
  Device count    : 2
  Device name     : Intel(R) Arc(TM) Pro ...

========================================================================
Phase 1: Peak Memory per Pipeline Stage
========================================================================
  Stage                           Peak Allocated     Peak Reserved
  ------------------------------  ------------------  ----------------
  model_load_cpu                         X.X MB allocated       X.X MB reserved
  to_xpu                              XXXX.X MB allocated    XXXX.X MB reserved
  inference                            XXXX.X MB allocated    XXXX.X MB reserved

  Wall-clock time:
    model_load_cpu                      XX.XX s
    to_xpu                               X.XX s
    inference                           XX.XX s

========================================================================
Phase 2: Torch Profiler Trace (Denoising Loop)
========================================================================
  Trace exported to: profile_trace.json
  ...top 20 operators table...

========================================================================
Phase 3: Summary
========================================================================
  Overall peak memory allocated :     XXXX.X MB
  Overall peak memory reserved  :     XXXX.X MB
  Total wall-clock time         :       XX.XX s
  Inference steps               :          4
  Frame count                   :         49
  Trace file                    : profile_trace.json
```

## Configuration

Edit constants at the top of `profile_run.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_INFERENCE_STEPS` | 4 | Number of denoising steps (increase for production-like profiling) |
| `NUM_FRAMES` | 49 | Number of video frames |
| `GUIDANCE_SCALE` | 6 | Classifier-free guidance scale |
| `DTYPE` | `torch.float16` | Model precision |
| `MODEL_PATH` | `/mnt/data4/yiliu/zai-org/CogVideoX-2b` | Path to pretrained model |
| `SEED` | 42 | Random seed for reproducibility |
