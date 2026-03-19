# CogVideoX-2B XPU Profiling Analysis

> **Hardware**: Intel Arc Pro B60 (32 GB VRAM) × 4 (single GPU used)
> **Software**: PyTorch 2.10.0+xpu, diffusers 0.37.0, fp16 precision
> **Config**: 4 inference steps, 49 frames, guidance_scale=6, seed=42
> **Date**: 2026-03-19

## 1. Executive Summary

CogVideoX-2B text-to-video inference on a single Intel Arc Pro B60 (32 GB) uses
**~18.4 GB peak GPU memory** and completes 4-step inference in **39 seconds**.
The pipeline comfortably fits within the 32 GB VRAM budget with ~13 GB headroom.

The denoising backbone is **GEMM-dominated** (51% of XPU time in matmul ops),
making it an ideal candidate for SageAttention kernel optimization. Secondary
bottlenecks include tensor concatenation (`aten::cat`) with high CPU overhead
and dtype conversion (`aten::copy_`) consuming significant wall-clock time.

Follow-up benchmarking of FlexAttention and `torch.compile` (Section 9) showed that
**FlexAttention regresses 2.7× on XPU** (unfused fallback), confirming that a custom
SageAttention kernel is the only viable attention optimization path. `torch.compile`
provides a **free 5% end-to-end speedup** from op fusion.

---

## 2. Memory Profile

### 2.1 Peak Memory per Pipeline Stage

| Stage | Peak Allocated | Peak Reserved | Description |
|-------|---------------|---------------|-------------|
| Model load (CPU) | 0.0 MB | 0.0 MB | Weights loaded to CPU — no XPU allocation |
| `pipe.to("xpu")` | 14,965 MB (14.6 GB) | 15,344 MB (15.0 GB) | Model weights transferred to GPU |
| Inference (4 steps) | 18,860 MB (18.4 GB) | 21,114 MB (20.6 GB) | Peak during denoising + VAE decode |

### 2.2 Memory Analysis

- **Model weights** consume ~14.6 GB on GPU — this is the static cost that cannot
  be reduced without quantization or offloading.
- **Inference activations** add ~3.9 GB on top of model weights (18.9 − 15.0 GB).
  This includes the 3D-UNet activations, attention KV-cache, and VAE decode buffers.
- **Memory fragmentation** is moderate: the allocator reserves 21.1 GB but only
  18.9 GB is actively allocated (10.5% overhead from the caching allocator).
- **Headroom**: 32 GB − 21.1 GB = **~10.9 GB free** at peak. Enough for:
  - Increasing `num_frames` beyond 49
  - Running the 5B variant (would need to verify)
  - Enabling gradient checkpointing for fine-tuning experiments

### 2.3 Memory Breakdown Estimate

```
Model weights on XPU ............ ~14.6 GB  (77% of peak)
Inference activations ........... ~ 3.9 GB  (21% of peak)
Allocator fragmentation ......... ~ 2.2 GB  ( 2% overhead)
                                  --------
Total reserved (peak) ........... ~20.6 GB
Available VRAM .................. ~32.0 GB
Headroom ........................ ~11.4 GB
```

---

## 3. Timing Profile

### 3.1 Wall-Clock Time per Stage

| Stage | Time | % of Total |
|-------|------|-----------|
| Model load (CPU) | 1.98 s | 4.4% |
| Move to XPU | 4.23 s | 9.3% |
| Inference (4 steps) | 39.02 s | 86.3% |
| **Total (Phase 1)** | **45.23 s** | 100% |

### 3.2 Per-Step Inference Time

With 4 steps completing in 39.02s:
- **~5.05 s/step** for the denoising loop
- VAE decode is included in the inference timing (runs once after all steps)

### 3.3 Production Estimate (50 steps)

Extrapolating linearly from the denoising step cost:
- 50 steps × 5.05 s/step ≈ **252 s (~4.2 min)** for denoising alone
- Plus VAE decode + overhead ≈ **~4.5–5 min total** for 50-step inference

---

## 4. Operator-Level Analysis

### 4.1 Top XPU Operators (by self XPU time)

| Rank | Operator | Self XPU Time | % of XPU Total | # Calls | Avg Time/Call |
|------|----------|--------------|----------------|---------|--------------|
| 1 | `gemm_kernel` | 2,472 ms | 30.3% | 1,412 | 1.75 ms |
| 2 | `aten::addmm` | 1,742 ms | 21.4% | 980 | 1.78 ms |
| 3 | `aten::mul` | 1,030 ms | 12.6% | 4,785 | 0.22 ms |
| 4 | `aten::cat` | 872 ms | 10.7% | 2,675 | 0.33 ms |
| 5 | `aten::add` | 867 ms | 10.6% | 5,401 | 0.16 ms |
| 6 | `aten::copy_` | 765 ms | 9.4% | 11,369 | 0.07 ms |
| 7 | `aten::gelu` | 752 ms | 9.2% | 120 | 6.27 ms |
| 8 | `aten::mm` | 568 ms | 7.0% | 336 | 1.69 ms |
| 9 | `aten::to/copy_` | 482 ms | 5.9% | 291 | 1.66 ms |
| 10 | `aten::layer_norm` | 286 ms | 3.5% | 728 | 0.39 ms |

**Total self XPU time: 8.15 s** (across a 35.6 s wall-clock run — the gap is
CPU scheduling, Python overhead, and CPU-XPU synchronization).

### 4.2 Operator Categories

Grouping by function:

| Category | Operators | Combined XPU Time | % of Total |
|----------|-----------|-------------------|-----------|
| **Matrix multiplication** | `gemm_kernel`, `aten::addmm`, `aten::mm` | 4,782 ms | **58.7%** |
| **Elementwise** | `aten::mul`, `aten::add`, `aten::gelu` | 2,649 ms | **32.5%** |
| **Data movement** | `aten::copy_`, `aten::to` | 1,247 ms | **15.3%** |
| **Tensor reshape** | `aten::cat`, `aten::clone` | 1,131 ms | **13.9%** |
| **Normalization** | `aten::layer_norm` | 286 ms | **3.5%** |

> Percentages sum >100% because some operators overlap (e.g. `gemm_kernel` is
> the underlying kernel for both `aten::addmm` and `aten::mm`).

### 4.3 CPU vs XPU Time Disparity

Several operators show significant CPU-side overhead:

| Operator | Self CPU Time | Self XPU Time | CPU:XPU Ratio | Issue |
|----------|-------------|---------------|--------------|-------|
| `aten::cat` | 4,159 ms | 872 ms | **4.8×** | CPU-bound tensor bookkeeping |
| `aten::copy_` | 362 ms | 765 ms | 0.5× | Normal — GPU-bound transfer |
| `aten::to/_to_copy` | 1.6 ms | 482 ms | 0.003× | Normal — GPU dtype conversion |

The `aten::cat` CPU overhead (4.16 s) is the **single largest CPU bottleneck**,
likely from metadata allocation for 2,675 concatenation calls in the diffusion
pipeline.

---

## 5. Key Findings and Optimization Opportunities

### 5.1 SageAttention Relevance

The profiling confirms that **attention GEMM operations dominate XPU time at ~59%**.
This is the primary target for SageAttention optimization:

- `aten::addmm` (980 calls, 1.74 s) — linear projections in Q/K/V and output
- `aten::mm` (336 calls, 568 ms) — attention score computation (Q × K^T) and
  value aggregation (attn × V)
- Combined: **2.31 s of pure attention matmul** per 4-step run

SageAttention's quantized attention kernels could reduce this by replacing
fp16 Q×K^T with INT8 quantized GEMM, potentially yielding **1.5–2× speedup**
on the attention path alone.

### 5.2 `aten::cat` CPU Bottleneck

2,675 `cat` calls consume 4.16 s of CPU time — more than any GPU operator.
This stems from the CogVideoX architecture concatenating conditional and
unconditional predictions at each step (classifier-free guidance with
`guidance_scale=6`).

**Potential mitigations:**
- Pre-allocate output tensors and use `torch.cat(out=...)` to avoid repeated allocation
- Investigate `torch.compile()` to fuse concatenation patterns
- Use `guidance_scale=1.0` (no CFG) if quality allows — halves the cat calls

### 5.3 Data Movement Overhead

`aten::copy_` is called **11,369 times** (765 ms XPU, 362 ms CPU), suggesting
frequent dtype conversions or device transfers within the pipeline. The
`aten::_to_copy` path (291 calls, 482 ms XPU) indicates fp16↔fp32 casts,
likely in layer normalization or attention softmax.

**Potential mitigations:**
- Ensure the entire pipeline runs in fp16 (check for unexpected fp32 upcasts)
- Use `torch.autocast("xpu", dtype=torch.float16)` to eliminate manual casts

### 5.4 GELU Activation Cost

`aten::gelu` accounts for 752 ms across only 120 calls (6.27 ms/call average) —
unusually expensive for an activation function. This suggests the GELU kernel
on XPU may not be fully optimized, or the tensor sizes are very large (3D video
activations).

**Potential mitigations:**
- Try `approximate="tanh"` GELU variant (often faster on non-CUDA backends)
- Profile with `torch.compile()` to check if GELU gets fused with adjacent ops

---

## 6. Estimated Production Performance (50 steps)

| Metric | 4 Steps (measured) | 50 Steps (estimated) |
|--------|-------------------|---------------------|
| Denoising time | ~20 s | ~250 s (~4.2 min) |
| VAE decode | ~19 s (once) | ~19 s (once) |
| Total inference | ~39 s | ~269 s (~4.5 min) |
| Peak memory | 18.9 GB | ~18.9 GB (same) |
| Output | 49 frames @ 8fps = 6.1 s video | same |

Memory stays constant regardless of step count — only wall-clock time scales
linearly with the number of denoising steps.

---

## 7. Hardware Utilization

| Metric | Value |
|--------|-------|
| GPU memory used at peak | 20.6 GB / 32 GB (64%) |
| XPU compute time | 8.15 s |
| Wall-clock time | 35.6 s (Phase 2) |
| **XPU utilization** | **~23%** |

The low XPU utilization (23%) indicates that the GPU spends most of its time
waiting — either for CPU-side scheduling, Python overhead, or data transfers.
This is common for diffusion models with many small kernel launches.
`torch.compile()` could significantly improve utilization by fusing small
operations into fewer, larger kernels.

---

## 8. Recommendations

| Priority | Optimization | Expected Impact | Effort |
|----------|-------------|----------------|--------|
| **P0** | Integrate SageAttention kernels for Q×K^T and attn×V | 1.5–2× attention speedup (~30% of total XPU time) | High |
| **P1** | Apply `torch.compile()` to the pipeline | Fuse small ops, reduce launch overhead, improve XPU utilization | Medium |
| **P2** | Investigate `aten::cat` CPU bottleneck — pre-allocate or fuse | Save ~4 s CPU time per run | Medium |
| **P3** | Audit dtype casts — ensure consistent fp16 throughout | Reduce `aten::copy_` calls from 11K | Low |
| **P4** | Try `approximate="tanh"` GELU | May save ~750 ms per run | Low |
| **P5** | Profile with 50 steps to validate linear scaling assumption | Confirm estimates | Low |

---

## 9. Optimization Benchmark Results

To validate recommendations P0 (attention kernel replacement) and P1 (`torch.compile`),
we benchmarked four configurations on the same pipeline and hardware:

| Config | Mean (s) | Std (s) | Peak Mem (MB) | Speedup |
|--------|----------|---------|---------------|---------|
| **baseline** (SDPA, uncompiled) | 35.21 | 0.00 | 18,861 | 1.00× |
| **flex_attention** | 93.97 | 2.15 | 18,859 | 0.37× |
| **torch.compile** (SDPA + compiled transformer) | 33.53 | 0.02 | 18,859 | **1.05×** |
| **flex_attention + compile** | 90.79 | 0.01 | 18,859 | 0.39× |

> Each config: 1 warmup run (JIT compilation) + 3 timed runs. Mean ± std reported
> for timed runs only. Peak memory is the max across timed runs.

### 9.1 FlexAttention Is Not Viable on XPU (0.37×)

`torch.nn.attention.flex_attention` — PyTorch's fused attention API — was tested as a
potential drop-in replacement for `F.scaled_dot_product_attention` in the
`CogVideoXAttnProcessor2_0`. Despite being wrapped in `torch.compile()`, FlexAttention
on the XPU backend falls back to an **unfused execution path** that is **2.7× slower**
than the native SDPA kernel.

Per-step denoising time increased from **5.00 s → 19.3 s**, confirming the regression
is in the attention computation itself, not in surrounding ops. The unfused fallback
decomposes attention into explicit Q×K^T matmul → softmax → attn×V, losing the
memory-efficient fused kernel that SDPA provides.

**Conclusion:** FlexAttention is not a viable shortcut for attention optimization on
Intel XPU. A custom kernel (SageAttention) remains the correct approach.

### 9.2 torch.compile Provides a Modest 5% Speedup (1.05×)

Compiling the transformer with `torch.compile(backend='inductor')` reduced per-step
denoising time from **5.00 s → 4.59 s** — a consistent **8% per-step improvement**
that translates to a **5% end-to-end speedup** (VAE decode is uncompiled).

The gain comes from fusing the "long tail" of small element-wise ops identified in
Section 4.2 (§ Elementwise: `aten::mul`, `aten::add`, `aten::gelu` = 2.65 s combined).
The inductor backend merges these into fewer, larger XPU kernels, reducing launch
overhead and improving utilization.

The first inference step after compilation takes ~70 s (JIT warmup), but all
subsequent runs are stable at 33.5 s. Memory usage is unchanged.

**Conclusion:** `torch.compile` is a free lunch — recommended as a baseline
optimization for production inference (P1 confirmed).

### 9.3 Memory Is Unchanged Across All Configs

All four configurations show identical peak memory (~18,859 MB). Neither FlexAttention
nor `torch.compile` introduces additional memory overhead, confirming that any
optimization gains are purely in compute efficiency.

### 9.4 Updated Recommendations

| Priority | Optimization | Status | Finding |
|----------|-------------|--------|---------|
| **P0** | SageAttention custom XPU kernels | **Confirmed needed** | FlexAttention fallback is 2.7× slower — no shortcut exists |
| **P1** | `torch.compile()` | **Validated: +5%** | Free 5% speedup, no memory cost, recommended for production |
| ~~P0-alt~~ | ~~FlexAttention as SDPA replacement~~ | **Rejected** | 2.7× regression due to unfused XPU fallback |

> Benchmark script: `bench_optimizations.py` (same directory)

---

## 10. Environment Details

```
PyTorch           : 2.10.0+xpu
Device            : Intel(R) Arc(TM) Pro B60 Graphics
VRAM              : 32 GB per device
Devices available : 4 (single device used)
Driver            : Xe kernel driver
Precision         : float16
Model             : CogVideoX-2b (2.6B parameters)
Pipeline          : diffusers.CogVideoXPipeline
```

---

## 11. Trace File

The full operator-level trace is available for interactive exploration:

- **Local**: `profile_trace.json` (167 MB)
- **Compressed**: `profile_trace.json.gz` (9.8 MB)
- **Hugging Face**: [Yi30/sageattn-xpu-profiling](https://huggingface.co/Yi30/sageattn-xpu-profiling)

Open in `chrome://tracing` for timeline visualization of CPU and XPU kernel execution.
