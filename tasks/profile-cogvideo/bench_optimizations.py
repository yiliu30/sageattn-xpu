"""
CogVideoX-2B Optimization Benchmark — Intel XPU

Compares 4 configurations on the same pipeline:
  1. baseline             — stock CogVideoXAttnProcessor2_0 (SDPA)
  2. flex_attention        — FlexAttention processor (compiled fused kernel)
  3. torch_compile         — torch.compile(transformer) with SDPA
  4. flex_attn + compile   — both combined

Each config: 1 warmup run (JIT) + 3 timed runs → mean ± std, peak memory, speedup.

Usage:
    cd tasks/profile-cogvideo
    uv run python bench_optimizations.py
"""

import gc
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

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

WARMUP_RUNS = 1
TIMED_RUNS = 3

PROMPT = (
    "A panda, dressed in a small, red jacket and a tiny hat, "
    "sits on a wooden stool in a serene bamboo forest. "
    "The panda gently sways as a breeze rustles through the tall bamboo stalks."
)


# ---------------------------------------------------------------------------
# FlexAttention Processor — drop-in replacement via diffusers plugin API
# ---------------------------------------------------------------------------
# Pre-compile flex_attention once (triggers Triton codegen on first call)
_compiled_flex_attention = torch.compile(flex_attention)


class CogVideoXFlexAttnProcessor:
    """
    Same as CogVideoXAttnProcessor2_0 but replaces F.scaled_dot_product_attention
    with torch.compile(flex_attention) for fused attention on XPU.
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

        # Q / K / V projections
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

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        # >>> FlexAttention replaces F.scaled_dot_product_attention <<<
        # CogVideoX never passes attention_mask in its transformer blocks,
        # so we use flex_attention without score_mod or block_mask.
        hidden_states = _compiled_flex_attention(query, key, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
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
    """Run one inference pass and return wall-clock seconds."""
    torch.xpu.synchronize()
    torch.xpu.reset_peak_memory_stats()

    t0 = time.perf_counter()
    pipe(
        prompt=PROMPT,
        num_videos_per_prompt=1,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_frames=NUM_FRAMES,
        guidance_scale=GUIDANCE_SCALE,
        generator=torch.Generator(device="xpu").manual_seed(SEED),
    )
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mem = mb(torch.xpu.max_memory_allocated())
    return elapsed, peak_mem


@dataclass
class BenchResult:
    name: str
    times: list[float] = field(default_factory=list)
    peak_mems: list[float] = field(default_factory=list)

    @property
    def mean_time(self) -> float:
        return sum(self.times) / len(self.times)

    @property
    def std_time(self) -> float:
        m = self.mean_time
        return (sum((t - m) ** 2 for t in self.times) / len(self.times)) ** 0.5

    @property
    def max_peak_mem(self) -> float:
        return max(self.peak_mems)


def bench_config(pipe, name: str) -> BenchResult:
    """Warmup + timed runs for a single configuration."""
    result = BenchResult(name=name)

    print(f"\n  [{name}] warmup ({WARMUP_RUNS} run)...")
    for _ in range(WARMUP_RUNS):
        run_inference(pipe)

    print(f"  [{name}] timing ({TIMED_RUNS} runs)...")
    for i in range(TIMED_RUNS):
        elapsed, peak_mem = run_inference(pipe)
        result.times.append(elapsed)
        result.peak_mems.append(peak_mem)
        print(f"    run {i+1}: {elapsed:.2f}s  peak_mem={peak_mem:.0f} MB")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print(f"CogVideoX-2B Optimization Benchmark")
    print(f"  {NUM_INFERENCE_STEPS} steps, {NUM_FRAMES} frames, {DTYPE}")
    print(f"  warmup={WARMUP_RUNS}, timed={TIMED_RUNS}")
    print(f"  Device: {torch.xpu.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Load pipeline once
    # ------------------------------------------------------------------
    print("\nLoading pipeline...")
    pipe = CogVideoXPipeline.from_pretrained(MODEL_PATH, torch_dtype=DTYPE)
    pipe = pipe.to("xpu")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    results: list[BenchResult] = []

    # ------------------------------------------------------------------
    # 1. Baseline — SDPA (uncompiled)
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("Config 1/4: baseline (SDPA, uncompiled)")
    print("-" * 72)
    results.append(bench_config(pipe, "baseline"))

    # ------------------------------------------------------------------
    # 2. FlexAttention only (uncompiled transformer)
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("Config 2/4: flex_attention")
    print("-" * 72)
    pipe.transformer.set_attn_processor(CogVideoXFlexAttnProcessor())
    results.append(bench_config(pipe, "flex_attention"))

    # Restore baseline processor before compile step
    pipe.transformer.set_attn_processor(CogVideoXAttnProcessor2_0())

    # ------------------------------------------------------------------
    # 3. torch.compile only — SDPA + compiled transformer
    #    NOTE: compile is irreversible, so this must come after configs 1-2
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("Config 3/4: torch_compile (SDPA + compiled transformer)")
    print("-" * 72)
    pipe.transformer = torch.compile(pipe.transformer, backend="inductor")
    results.append(bench_config(pipe, "torch_compile"))

    # ------------------------------------------------------------------
    # 4. FlexAttention + compile — swap processor on compiled transformer
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("Config 4/4: flex_attention + torch_compile")
    print("-" * 72)
    # set_attn_processor works on the underlying module inside the compiled wrapper
    pipe.transformer.set_attn_processor(CogVideoXFlexAttnProcessor())
    results.append(bench_config(pipe, "flex_attn + compile"))

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    baseline_mean = results[0].mean_time

    print("\n")
    print("=" * 72)
    print(f"CogVideoX-2B Optimization Benchmark Results")
    print(f"  ({NUM_INFERENCE_STEPS} steps, {NUM_FRAMES} frames, {DTYPE})")
    print("=" * 72)
    header = (
        f"  {'Config':<24s}  {'Mean (s)':>9s}  {'Std (s)':>8s}  "
        f"{'Peak Mem (MB)':>14s}  {'Speedup':>8s}"
    )
    print(header)
    print(f"  {'-'*24}  {'-'*9}  {'-'*8}  {'-'*14}  {'-'*8}")

    for r in results:
        speedup = baseline_mean / r.mean_time
        print(
            f"  {r.name:<24s}  {r.mean_time:>9.2f}  {r.std_time:>8.2f}  "
            f"{r.max_peak_mem:>14.0f}  {speedup:>7.2f}x"
        )

    print()
    print("Notes:")
    print("  - Speedup is relative to the baseline (SDPA, uncompiled)")
    print("  - Peak Mem is the max across timed runs (excludes warmup)")
    print("  - flex_attention uses torch.compile(flex_attention) for fused kernels")
    print("  - torch_compile uses torch.compile(transformer, backend='inductor')")
    print()


if __name__ == "__main__":
    main()
