import os
from uuid import uuid4

import numpy as np
from PIL import Image
import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse, JSONResponse

# ---------------------------------
# FastAPI app
# ---------------------------------
app = FastAPI(
    title="SkyReels Text‑to‑Video API",
    description="Generates clips with SkyReels‑V2",
    version="1.3",
)

# ---------------------------------
# Pick the model
# ---------------------------------
# 1.3 B 540 P  → fits 24‑44 GB cards
# MODEL_ID = "tolgacangoz/SkyReels-V2-DF-1.3B-540P-Diffusers"

# 14 B 540 P  → needs 51 GB+ VRAM (your A100 can handle it)
MODEL_ID = "tolgacangoz/SkyReels-V2-DF-14B-540P-Diffusers"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔄 Loading {MODEL_ID} …")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to(DEVICE)
pipe.enable_model_cpu_offload()
print("✅ Model ready!")

os.makedirs("outputs", exist_ok=True)


# ---------------------------------
# Helper to turn ndarray → PIL.Image
# ---------------------------------
def to_pil(frames_np):
    return [Image.fromarray(f.astype(np.uint8)) for f in frames_np]


# ---------------------------------
# Routes
# ---------------------------------
@app.get("/")
def health():
    return {"status": "running", "message": "SkyReels API is live!"}


@app.post("/generate")
def generate(
    prompt: str = Body(..., embed=True),
    num_frames: int = Body(29, embed=True),   # 29 meets (n‑1) % 4 == 0
    fps: int = Body(8, embed=True),
    fmt: str = Body("gif", embed=True),       # gif | mp4
):
    try:
        if (num_frames - 1) % 4 != 0:
            num_frames = ((num_frames - 1) // 4) * 4 + 1
            print(f"[Info] num_frames adjusted to {num_frames}")

        print(f"🎬 Generating {num_frames} frames for: {prompt!r}")
        out = pipe(prompt, num_frames=num_frames)
        frames_np = out.frames                  # list[np.ndarray]
        frames = to_pil(frames_np)

        uid = uuid4().hex
        if fmt.lower() == "mp4":
            import imageio.v3 as iio           # add imageio[ffmpeg] to deps
            path = f"outputs/{uid}.mp4"
            iio.imwrite(path, frames_np, fps=fps, codec="libx264")
            mime = "video/mp4"
        else:
            path = f"outputs/{uid}.gif"
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),
                loop=0,
            )
            mime = "image/gif"

        print(f"✅ Saved to {path}")
        return {"video_path": path, "mime_type": mime}

    except Exception as e:
        print("❌ Generation failed:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/download/{file}")
def download(file: str):
    real = os.path.join("outputs", file)
    if os.path.exists(real):
        mime = "video/mp4" if file.endswith(".mp4") else "image/gif"
        return FileResponse(real, media_type=mime)
    return JSONResponse({"error": "File not found"}, status_code=404)
