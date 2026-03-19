import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

MODEL_PATH = "/mnt/data4/yiliu/zai-org/CogVideoX-2b"

prompt = (
    "A panda, dressed in a small, red jacket and a tiny hat, "
    "sits on a wooden stool in a serene bamboo forest. "
    "The panda gently sways as a breeze rustles through the tall bamboo stalks."
)

print(f"PyTorch version: {torch.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")
if torch.xpu.is_available():
    print(f"XPU device count: {torch.xpu.device_count()}")
    print(f"XPU device name: {torch.xpu.get_device_name(0)}")

print(f"\nLoading model from {MODEL_PATH}...")
pipe = CogVideoXPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
pipe = pipe.to("xpu")

# Enable memory optimizations
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

print("Running inference...")
video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="xpu").manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)
print("Video saved to output.mp4")
