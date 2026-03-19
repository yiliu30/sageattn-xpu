"""
CogVideoX-2B FlexAttention Profiler Trace — Intel XPU

Profiles the FlexAttention path under torch.profiler to identify exactly
where the 2.7× regression vs SDPA comes from. Runs both baseline (SDPA)
and FlexAttention under the profiler for direct comparison.

Outputs:
  - profile_flex_baseline.json.gz  — baseline SDPA trace
  - profile_flex_attention.json.gz — FlexAttention trace
  - Side-by-side top-20 operator tables (stdout)

Usage:
    cd tasks/profile-cogvideo
    uv run python profile_flex.py
"""

import gzip
import shutil
import time

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from torch.profiler import ProfilerActivity, profile, record_function

from diffusers import CogVideoXPipeline
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_INFERENCE_STEPS = 4
NUM_FRAMES = 49
GUIDANCE_SCALE = 6
DTYPE = torch.float16
SEED = 42
MODEL_PATH = "/mnt/data4/yiliu/zai-org/CogVideoX-2b"

PROMPT = (
    "A panda, dressed in a small, red jacket and a tiny hat, "
    "sits on a wooden stool in a serene bamboo forest. "
    "The panda gently sways as a breeze rustles through the tall bamboo stalks."
)


# ---------------------------------------------------------------------------
# FlexAttention Processor (same as bench_optimizations.py)
# ---------------------------------------------------------------------------
_compiled_flex_attention = torch.compile(flex_attention)


class CogVideoXFlexAttnProcessor:
    """
    Drop-in replacement for CogVideoXAttnProcessor2_0 using FlexAttention.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        hidden_states = _compiled_flex_attention(query, key, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mb(b: int) -> float:
    return b / (1024 * 1024)


def run_inference(pipe):
    """Single inference pass."""
    pipe(
        prompt=PROMPT,
        num_videos_per_prompt=1,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_frames=NUM_FRAMES,
        guidance_scale=GUIDANCE_SCALE,
        generator=torch.Generator(device="xpu").manual_seed(SEED),
    )
    torch.xpu.synchronize()


def compress_trace(json_path: str):
    """Compress a .json trace to .json.gz and remove the original."""
    gz_path = json_path + ".gz"
    with open(json_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    import os
    os.remove(json_path)
    return gz_path


def profile_config(pipe, label: str, trace_file: str):
    """Warmup + profiled run for one configuration."""
    print(f"\n  [{label}] warmup run...")
    t0 = time.perf_counter()
    run_inference(pipe)
    print(f"  [{label}] warmup done in {time.perf_counter() - t0:.1f}s")

    print(f"  [{label}] profiled run...")
    torch.xpu.reset_peak_memory_stats()
    t0 = time.perf_counter()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        with record_function(f"profiled_inference_{label}"):
            run_inference(pipe)

    elapsed = time.perf_counter() - t0
    peak_mem = mb(torch.xpu.max_memory_allocated())

    # Export trace
    json_path = trace_file.replace(".json.gz", ".json")
    prof.export_chrome_trace(json_path)
    gz_path = compress_trace(json_path)
    print(f"  [{label}] profiled run done in {elapsed:.1f}s, peak mem={peak_mem:.0f} MB")
    print(f"  [{label}] trace saved to {gz_path}")

    return prof, elapsed, peak_mem


def print_top_ops(prof, label: str, n: int = 30):
    """Print top operators by XPU time."""
    print(f"\n  Top {n} operators by XPU time ({label}):")
    print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=n))


def compare_tables(baseline_prof, flex_prof):
    """Side-by-side comparison of key operator categories."""
    print("\n" + "=" * 72)
    print("Operator Category Comparison: Baseline vs FlexAttention")
    print("=" * 72)

    categories = {
        "GEMM (gemm_kernel)": "gemm_kernel",
        "aten::addmm": "aten::addmm",
        "aten::mm": "aten::mm",
        "aten::bmm": "aten::bmm",
        "aten::mul": "aten::mul",
        "aten::add": "aten::add",
        "aten::cat": "aten::cat",
        "aten::copy_": "aten::copy_",
        "aten::gelu": "aten::gelu",
        "aten::softmax": "aten::_softmax",
        "aten::layer_norm": "aten::layer_norm",
        "aten::matmul": "aten::matmul",
        "aten::scaled_dot_product_attention": "aten::scaled_dot_product_attention",
    }

    def get_stats(prof, key_substr):
        total_xpu = 0.0
        total_cpu = 0.0
        count = 0
        for evt in prof.key_averages():
            if key_substr in evt.key:
                total_xpu += evt.self_device_time_total / 1000  # μs → ms
                total_cpu += evt.self_cpu_time_total / 1000
                count += evt.count
        return total_xpu, total_cpu, count

    header = (
        f"  {'Operator':<38s}  "
        f"{'Baseline XPU':>12s}  {'Flex XPU':>12s}  {'Delta':>8s}  "
        f"{'Base #':>7s}  {'Flex #':>7s}"
    )
    print(header)
    print(f"  {'-'*38}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*7}  {'-'*7}")

    for label, key in categories.items():
        b_xpu, b_cpu, b_count = get_stats(baseline_prof, key)
        f_xpu, f_cpu, f_count = get_stats(flex_prof, key)

        if b_xpu == 0 and f_xpu == 0:
            continue

        if b_xpu > 0:
            delta = f"{(f_xpu - b_xpu) / b_xpu * 100:+.0f}%"
        elif f_xpu > 0:
            delta = "new"
        else:
            delta = "—"

        print(
            f"  {label:<38s}  "
            f"{b_xpu:>10.0f}ms  {f_xpu:>10.0f}ms  {delta:>8s}  "
            f"{b_count:>7d}  {f_count:>7d}"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("CogVideoX-2B FlexAttention Profiling")
    print(f"  {NUM_INFERENCE_STEPS} steps, {NUM_FRAMES} frames, {DTYPE}")
    print(f"  Device: {torch.xpu.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 72)

    # Load pipeline
    print("\nLoading pipeline...")
    pipe = CogVideoXPipeline.from_pretrained(MODEL_PATH, torch_dtype=DTYPE)
    pipe = pipe.to("xpu")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # ------------------------------------------------------------------
    # 1. Profile Baseline (SDPA)
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("Profile 1/2: Baseline (SDPA)")
    print("-" * 72)
    baseline_prof, baseline_time, baseline_mem = profile_config(
        pipe, "baseline", "profile_flex_baseline.json.gz"
    )
    print_top_ops(baseline_prof, "baseline")

    # ------------------------------------------------------------------
    # 2. Profile FlexAttention
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("Profile 2/2: FlexAttention")
    print("-" * 72)
    pipe.transformer.set_attn_processor(CogVideoXFlexAttnProcessor())
    flex_prof, flex_time, flex_mem = profile_config(
        pipe, "flex_attention", "profile_flex_attention.json.gz"
    )
    print_top_ops(flex_prof, "flex_attention")

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    compare_tables(baseline_prof, flex_prof)

    # Summary
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Baseline wall-clock:      {baseline_time:>8.1f}s  peak_mem={baseline_mem:.0f} MB")
    print(f"  FlexAttention wall-clock:  {flex_time:>8.1f}s  peak_mem={flex_mem:.0f} MB")
    print(f"  Ratio:                     {flex_time / baseline_time:>8.2f}x")
    print()
    print("  Trace files:")
    print("    profile_flex_baseline.json.gz   — open in chrome://tracing")
    print("    profile_flex_attention.json.gz  — open in chrome://tracing")
    print()


if __name__ == "__main__":
    main()
