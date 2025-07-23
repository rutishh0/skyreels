import os
from uuid import uuid4

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse, JSONResponse

# -----------------------------------------------------------
# FastAPI application
# -----------------------------------------------------------
app = FastAPI(
    title="SkyReels Text‑to‑Video API",
    description="Generate short clips with SkyReels‑V2 (1.3 B, 540 P)",
    version="1.1",
)

# -----------------------------------------------------------
# Load the model (fits a 44 GB RTX‑4000)
# -----------------------------------------------------------
MODEL_ID = "tolgacangoz/SkyReels-V2-DF-1.3B-540P-Diffusers"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🔄 Loading SkyReels model …")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    trust_remote_code=True,  # picks up custom code if the repo provides it
).to(DEVICE)

# Optional memory helper
pipe.enable_model_cpu_offload()
# This pipeline doesn’t expose enable_vae_slicing(); remove or guard it:
# if hasattr(pipe, "enable_vae_slicing"):
#     pipe.enable_vae_slicing()

print("✅ Model ready!")

# Folder to keep generated clips
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
    fps: int = Body(8, embed=True),
):
    """
    Example request:
      {
        "prompt": "A cyberpunk city skyline at night",
        "num_frames": 24,
        "fps": 8
      }
    Response:
      { "video_path": "outputs/<uuid>.gif" }
    """
    try:
        print(f"🎬 Generating {num_frames}‑frame clip for prompt: {prompt!r}")
        result = pipe(prompt, num_frames=num_frames)
        frames = result.frames  # list[PIL.Image]

        outfile = f"outputs/{uuid4().hex}.gif"
        frames[0].save(
            outfile,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )

        print(f"✅ Saved: {outfile}")
        return JSONResponse({"video_path": outfile})

    except Exception as err:
        print("❌ Generation failed:", err)
        return JSONResponse({"error": str(err)}, status_code=500)


@app.get("/download/{filename}")
def download_video(filename: str):
    """Download a generated clip."""
    path = os.path.join("outputs", filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/gif")
    return JSONResponse({"error": "File not found"}, status_code=404)
