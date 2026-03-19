# Run CogVideoX-2b on Intel XPU

Run the CogVideoX-2b text-to-video model on Intel Data Center GPUs using PyTorch XPU.

- Model card: https://huggingface.co/zai-org/CogVideoX-2b
- Local model path: `/mnt/data4/yiliu/zai-org/CogVideoX-2b`

## Quick Start

```bash
# Install dependencies (uv handles the venv automatically)
uv sync

# Run inference
uv run python run.py
# => output.mp4 (~6s video, 49 frames at 8fps)
```

## Environment

| Component | Version / Detail |
|-----------|-----------------|
| Python | 3.12 |
| PyTorch | 2.10.0+xpu |
| Diffusers | 0.37.0 |
| Transformers | 5.3.0 |
| GPU | 4x Intel Arc Pro B60 (32 GB VRAM each) |

## Key Setup Notes

### PyTorch XPU index in uv

PyTorch XPU wheels live on a separate index (`https://download.pytorch.org/whl/xpu`).
In `pyproject.toml` the index is marked `explicit = true` so only the packages we
explicitly map in `[tool.uv.sources]` are pulled from it — everything else resolves
from PyPI as usual.

Critically, `torch` depends on `pytorch-triton-xpu` and `triton-xpu` which also live
on the XPU index. Because they are transitive deps, they must be added as **direct
dependencies** with their own source mapping, otherwise uv cannot find them:

```toml
[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-xpu" }
pytorch-triton-xpu = { index = "pytorch-xpu" }
triton-xpu = { index = "pytorch-xpu" }
```

### No IPEX needed

`torch 2.10.0+xpu` has native XPU support built in. There is no need to install
`intel_extension_for_pytorch` — `pipe.to("xpu")` and `torch.xpu.*` work out of the box.

### Extra dependencies beyond diffusers

- `protobuf` — required by the T5 tokenizer's SentencePiece extractor at load time.
- `imageio` + `imageio-ffmpeg` — required by `diffusers.utils.export_to_video` for
  writing mp4 files (without these it falls back to OpenCV, which is not installed).

## Inference Parameters

| Parameter | Value |
|-----------|-------|
| dtype | float16 |
| Inference steps | 50 |
| Frames | 49 |
| Guidance scale | 6 |
| Seed | 42 |
| VAE slicing | enabled |
| VAE tiling | enabled |

## Benchmark Results

| Metric | Value |
|--------|-------|
| Total inference time | ~4 min 10 sec |
| Per-step latency | ~5.0 sec/step |
| Output size | 406 KB |
| GPU memory | Single GPU, no OOM (32 GB headroom) |

## Potential Issues & Fallbacks

- **OOM**: If a single GPU runs out of memory, enable `pipe.enable_model_cpu_offload()`
  or reduce `num_frames` from 49 to 25.
- **FP16 issues**: If FP16 produces artifacts on XPU, try `torch.bfloat16` or `torch.float32`.
- **XPU not detected**: You may need to `source /opt/intel/oneapi/setvars.sh` first.
