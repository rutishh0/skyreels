import os
from uuid import uuid4
from typing import Literal, List

import numpy as np
from PIL import Image
import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, Body, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse

# -----------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------
app = FastAPI(
    title="SkyReels Text‑to‑Video API",
    description="Async clip generator with job IDs (robust ndarray→PIL handling)",
    version="2.1",
)

# -----------------------------------------------------------
# Model choice
# -----------------------------------------------------------
MODEL_ID = "tolgacangoz/SkyReels-V2-DF-14B-540P-Diffusers"  # or 1.3 B path
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔄 Loading {MODEL_ID} …")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to(DEVICE)
pipe.enable_model_cpu_offload()
print("✅ Model ready!")

os.makedirs("outputs", exist_ok=True)

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def ndarray_to_pil(frames_any) -> List[Image.Image]:
    """Convert SkyReels output (list or ndarray) to list of PIL.Image."""
    # SkyReels may return a single 4‑D ndarray or a list of ndarrays
    if isinstance(frames_any, np.ndarray):
        frames_any = list(frames_any)

    images: List[Image.Image] = []
    for arr in frames_any:
        a = np.asarray(arr)

        # Squeeze leading singleton dimensions until ndim ≤ 3
        while a.ndim > 3 and a.shape[0] == 1:
            a = a.squeeze(0)

        # If channels‑first (3, H, W) move channels last
        if a.ndim == 3 and a.shape[0] in (1, 3):
            a = np.transpose(a, (1, 2, 0))  # (H, W, C)

        images.append(Image.fromarray(a.astype(np.uint8)))
    return images


def run_job(
    uid: str,
    prompt: str,
    num_frames: int,
    fps: int,
    fmt: Literal["gif", "mp4"],
    steps: int,
):
    # SkyReels stride rule
    if (num_frames - 1) % 4 != 0:
        num_frames = ((num_frames - 1) // 4) * 4 + 1

    print(f"[{uid}] Generating {num_frames} frames for {prompt!r}")
    out = pipe(prompt, num_frames=num_frames, num_inference_steps=steps)
    frames_np = out.frames  # could be list[np.ndarray] or single ndarray

    if fmt == "mp4":
        import imageio.v3 as iio  # ensure imageio[ffmpeg] in requirements.txt
        path = f"outputs/{uid}.mp4"
        if isinstance(frames_np, list):
            frames_to_write = np.stack(frames_np)
        else:
            frames_to_write = frames_np
        iio.imwrite(path, frames_to_write, fps=fps, codec="libx264")
    else:
        frames_pil = ndarray_to_pil(frames_np)
        path = f"outputs/{uid}.gif"
        frames_pil[0].save(
            path,
            save_all=True,
            append_images=frames_pil[1:],
            duration=int(1000 / fps),
            loop=0,
        )

    print(f"[{uid}] Saved to {path}")

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------
@app.get("/")
def health():
    return {"status": "running"}


@app.post("/generate", status_code=status.HTTP_202_ACCEPTED)
def generate(
    bg: BackgroundTasks,
    prompt: str = Body(..., embed=True),
    num_frames: int = Body(29, embed=True),
    fps: int = Body(8, embed=True),
    fmt: Literal["gif", "mp4"] = Body("gif", embed=True),
    num_inference_steps: int = Body(50, embed=True),
):
    uid = uuid4().hex
    bg.add_task(run_job, uid, prompt, num_frames, fps, fmt, num_inference_steps)
    return {"id": uid}


@app.get("/download/{filename}")
def download(filename: str):
    path = os.path.join("outputs", filename)
    if os.path.exists(path):
        mime = "video/mp4" if filename.endswith(".mp4") else "image/gif"
        return FileResponse(path, media_type=mime)
    return JSONResponse({"status": "processing"}, status_code=404)
