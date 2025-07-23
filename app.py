import os
from uuid import uuid4

import numpy as np
from PIL import Image
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
    version="1.2",
)

# -----------------------------------------------------------
# Load the model (fits a 44 GB RTX‑4000)
# -----------------------------------------------------------
model_id = "tolgacangoz/SkyReels-V2-DF-14B-540P-Diffusers"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🔄 Loading SkyReels model …")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to(DEVICE)
pipe.enable_model_cpu_offload()

print("✅ Model ready!")

# Folder to keep generated clips
os.makedirs("outputs", exist_ok=True)

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def to_pil(frames_np):
    """Convert a list of np.ndarray to PIL.Image."""
    return [Image.fromarray(f.astype(np.uint8)) for f in frames_np]

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
    format: str = Body("gif", embed=True),  # "gif" or "mp4"
):
    """
    POST JSON:
      {
        "prompt": "Glowing jellyfish in space",
        "num_frames": 12,
        "fps": 8,
        "format": "gif"
      }
    """
    try:
        # SkyReels needs (num_frames - 1) % 4 == 0
        if (num_frames - 1) % 4 != 0:
            num_frames = ((num_frames - 1) // 4) * 4 + 1
            print(f"[Info] Adjusted num_frames to {num_frames} to match model requirement.")

        print(f"🎬 Generating {num_frames}‑frame clip for prompt: {prompt!r}")
        result = pipe(prompt, num_frames=num_frames)
        frames = result.frames  # list[np.ndarray]

        # Convert to PIL so we can save easily
        frames_pil = to_pil(frames)

        uid = uuid4().hex
        if format.lower() == "mp4":
            # Requires imageio[ffmpeg] in requirements.txt
            import imageio.v3 as iio
            outfile = f"outputs/{uid}.mp4"
            iio.imwrite(outfile, frames, fps=fps, codec="libx264")
            media_type = "video/mp4"
        else:
            outfile = f"outputs/{uid}.gif"
            frames_pil[0].save(
                outfile,
                save_all=True,
                append_images=frames_pil[1:],
                duration=int(1000 / fps),
                loop=0,
            )
            media_type = "image/gif"

        print(f"✅ Saved: {outfile}")
        return JSONResponse({"video_path": outfile, "mime_type": media_type})

    except Exception as err:
        print("❌ Generation failed:", err)
        return JSONResponse({"error": str(err)}, status_code=500)


@app.get("/download/{filename}")
def download_video(filename: str):
    """Download a generated clip."""
    path = os.path.join("outputs", filename)
    if os.path.exists(path):
        # infer mime type from extension
        mime = "video/mp4" if filename.lower().endswith(".mp4") else "image/gif"
        return FileResponse(path, media_type=mime)
    return JSONResponse({"error": "File not found"}, status_code=404)
