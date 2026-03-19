"""
CogVideoX-2B Profiling Script for Intel XPU

Profiles the full text-to-video pipeline across three phases:
  1. Peak GPU memory per pipeline stage
  2. Torch profiler trace of the denoising loop
  3. Wall-clock timing summary

Outputs:
  - Memory table (printed to stdout)
  - profile_trace.json (Chrome trace format — open in chrome://tracing)
  - Timing summary (printed to stdout)
"""

import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_INFERENCE_STEPS = 4       # keep profiling fast
NUM_FRAMES = 49               # same frame count as production
GUIDANCE_SCALE = 6
DTYPE = torch.float16
SEED = 42
MODEL_PATH = "/mnt/data4/yiliu/zai-org/CogVideoX-2b"
TRACE_OUTPUT = "profile_trace.json"

PROMPT = (
    "A panda, dressed in a small, red jacket and a tiny hat, "
    "sits on a wooden stool in a serene bamboo forest. "
    "The panda gently sways as a breeze rustles through the tall bamboo stalks."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mb(b: int) -> float:
    """Convert bytes to megabytes."""
    return b / (1024 * 1024)


def print_memory(label: str, stats: dict):
    """Print one row of the memory table."""
    print(
        f"  {label:<30s}  "
        f"{stats['peak_allocated']:>10.1f} MB allocated  "
        f"{stats['peak_reserved']:>10.1f} MB reserved"
    )


def snapshot_peak() -> dict:
    """Return current peak memory stats and reset counters."""
    torch.xpu.synchronize()
    stats = {
        "peak_allocated": mb(torch.xpu.max_memory_allocated()),
        "peak_reserved": mb(torch.xpu.max_memory_reserved()),
    }
    torch.xpu.reset_peak_memory_stats()
    return stats


# ---------------------------------------------------------------------------
# Device diagnostics
# ---------------------------------------------------------------------------
def print_device_info():
    print("=" * 72)
    print("Device Diagnostics")
    print("=" * 72)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  XPU available   : {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"  Device count    : {torch.xpu.device_count()}")
        print(f"  Device name     : {torch.xpu.get_device_name(0)}")
    print()


# ===================================================================== #
# Phase 1 — Peak memory per pipeline stage                              #
# ===================================================================== #
def phase1_memory_profile():
    print("=" * 72)
    print("Phase 1: Peak Memory per Pipeline Stage")
    print("=" * 72)

    timings = {}
    memory_stats = {}

    # --- Model load (CPU) ---
    torch.xpu.reset_peak_memory_stats()
    t0 = time.perf_counter()

    with record_function("model_load"):
        pipe = CogVideoXPipeline.from_pretrained(MODEL_PATH, torch_dtype=DTYPE)

    timings["model_load_cpu"] = time.perf_counter() - t0
    memory_stats["model_load_cpu"] = snapshot_peak()

    # --- Move to XPU ---
    t0 = time.perf_counter()

    with record_function("to_xpu"):
        pipe = pipe.to("xpu")

    timings["to_xpu"] = time.perf_counter() - t0
    memory_stats["to_xpu"] = snapshot_peak()

    # --- Enable VAE optimizations ---
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # --- Inference (denoising + VAE decode) ---
    t0 = time.perf_counter()

    with record_function("inference"):
        video = pipe(
            prompt=PROMPT,
            num_videos_per_prompt=1,
            num_inference_steps=NUM_INFERENCE_STEPS,
            num_frames=NUM_FRAMES,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator(device="xpu").manual_seed(SEED),
        ).frames[0]

    torch.xpu.synchronize()
    timings["inference"] = time.perf_counter() - t0
    memory_stats["inference"] = snapshot_peak()

    # --- Print memory table ---
    print()
    print(f"  {'Stage':<30s}  {'Peak Allocated':>18s}  {'Peak Reserved':>16s}")
    print(f"  {'-'*30}  {'-'*18}  {'-'*16}")
    for stage, stats in memory_stats.items():
        print_memory(stage, stats)
    print()

    # --- Print timing ---
    print("  Wall-clock time:")
    for stage, t in timings.items():
        print(f"    {stage:<30s}  {t:>8.2f} s")
    print()

    return pipe, video


# ===================================================================== #
# Phase 2 — Torch profiler trace (denoising loop)                       #
# ===================================================================== #
def phase2_profiler_trace(pipe):
    print("=" * 72)
    print("Phase 2: Torch Profiler Trace (Denoising Loop)")
    print("=" * 72)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        with record_function("profiled_inference"):
            pipe(
                prompt=PROMPT,
                num_videos_per_prompt=1,
                num_inference_steps=NUM_INFERENCE_STEPS,
                num_frames=NUM_FRAMES,
                guidance_scale=GUIDANCE_SCALE,
                generator=torch.Generator(device="xpu").manual_seed(SEED),
            )

    torch.xpu.synchronize()

    # Export Chrome trace
    prof.export_chrome_trace(TRACE_OUTPUT)
    print(f"\n  Trace exported to: {TRACE_OUTPUT}")
    print(f"  Open in chrome://tracing to view operator-level breakdown")

    # Print top operators by XPU time
    print(f"\n  Top 20 operators by XPU time:")
    print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=20))
    print()


# ===================================================================== #
# Phase 3 — Summary                                                     #
# ===================================================================== #
def phase3_summary(t_total: float):
    print("=" * 72)
    print("Phase 3: Summary")
    print("=" * 72)

    overall_peak_alloc = mb(torch.xpu.max_memory_allocated())
    overall_peak_res = mb(torch.xpu.max_memory_reserved())

    print(f"  Overall peak memory allocated : {overall_peak_alloc:>10.1f} MB")
    print(f"  Overall peak memory reserved  : {overall_peak_res:>10.1f} MB")
    print(f"  Total wall-clock time         : {t_total:>10.2f} s")
    print(f"  Inference steps               : {NUM_INFERENCE_STEPS}")
    print(f"  Frame count                   : {NUM_FRAMES}")
    print(f"  Trace file                    : {TRACE_OUTPUT}")
    print()


# ===================================================================== #
# Main                                                                   #
# ===================================================================== #
def main():
    print_device_info()

    t_start = time.perf_counter()

    # Phase 1: Memory profiling (returns the loaded pipeline & first video)
    pipe, video = phase1_memory_profile()

    # Save the video from Phase 1 for verification
    export_to_video(video, "profile_output.mp4", fps=8)
    print(f"  Video saved to profile_output.mp4\n")

    # Phase 2: Profiler trace (re-runs inference under the profiler)
    phase2_profiler_trace(pipe)

    t_total = time.perf_counter() - t_start

    # Phase 3: Summary
    phase3_summary(t_total)


if __name__ == "__main__":
    main()
