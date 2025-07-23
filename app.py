import os
import torch
from uuid import uuid4
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse, JSONResponse
from diffusers import DiffusionPipeline

# -----------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------
app = FastAPI(
    title="SkyReels Text‑to‑Video API",
    description="Generate short clips with SkyReels‑V2 (1.3 B, 540 P)",
    version="1.0"
)

# -----------------------------------------------------------
# Model: use the 1.3 B Diffusers export so it fits 44 GB VRAM
# -----------------------------------------------------------
print("🔄 Loading SkyReels model…")
model_id = "tolgacangoz/SkyReels-V2-DF-1.3B-540P-Diffusers"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True,                   # allow custom code
    custom_pipeline="skyreels_v2_diffusion_forcing"  # load pipeline file in repo
).to("cuda" if torch.cuda.is_available() else "cpu")

# (Optional) memory helpers
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

print("✅ SkyReels model ready!")

# Folder to keep outputs
os.makedirs("outputs", exist_ok=True)

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "running", "message": "SkyReels API is live!"}


@app.post("/generate")
def generate_video(
    prompt: str = Body(..., embed=True),
    num_frames: int = Body(16, embed=True),
    fps: int = Body(8, embed=True)
):
    """
    POST JSON:
      {
        "prompt": "A cyberpunk city skyline at night",
        "num_frames": 200,
        "fps": 24
      }
    Returns: {"video_path": "outputs/<uuid>.gif"}
    """
    try:
        print(f"🎬 Generating {num_frames}‑frame clip for: {prompt}")
        result = pipe(prompt, num_frames=num_frames)
        frames = result.frames                       # list[PIL.Image]

        # Save as GIF (runs everywhere, no ffmpeg)
        outfile = f"outputs/{uuid4().hex}.gif"
        frames[0].save(
            outfile,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0
        )

        print(f"✅ Saved: {outfile}")
        return JSONResponse({"video_path": outfile})

    except Exception as e:
        print("❌ Generation failed:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/download/{filename}")
def download_video(filename: str):
    """Download a generated clip."""
    path = os.path.join("outputs", filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/gif")
    return JSONResponse({"error": "File not found"}, status_code=404)
