import os
import torch
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse, JSONResponse
from diffusers import DiffusionPipeline
from uuid import uuid4

# ✅ FastAPI app
app = FastAPI(
    title="SkyReels Text-to-Video API",
    description="Generate short videos using SkyReels-V2-DF-14B-720P",
    version="1.0"
)

# ✅ Load model once at startup
print("🔄 Loading SkyReels model... This might take a few minutes on first run.")
model_id = "Skywork/SkyReels-V2-DF-14B-720P"

pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda" if torch.cuda.is_available() else "cpu")

print("✅ SkyReels model loaded successfully!")

# ✅ Output folder for videos
os.makedirs("outputs", exist_ok=True)


@app.get("/")
def health_check():
    return {"status": "running", "message": "SkyReels API is live!"}


@app.post("/generate")
def generate_video(
    prompt: str = Body(..., embed=True),
    num_frames: int = Body(16, embed=True)
):
    """
    Generate a short video from a text prompt.
    - prompt: text describing the scene
    - num_frames: how many frames to generate (default 16)
    """

    try:
        # 🔄 Generate frames
        print(f"🎬 Generating video for prompt: {prompt}")
        video_frames = pipe(prompt=prompt, num_frames=num_frames).frames

        # ✅ Save video as mp4
        output_filename = f"outputs/{uuid4().hex}.mp4"
        video_frames[0].save(output_filename, format="MP4")

        print(f"✅ Video saved at {output_filename}")

        # Return the file path (you can also upload to S3 or another CDN)
        return JSONResponse({"video_path": output_filename})

    except Exception as e:
        print("❌ Error generating video:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/download/{filename}")
def download_video(filename: str):
    """
    Download a generated video by filename.
    """
    file_path = os.path.join("outputs", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4")
    else:
        return JSONResponse({"error": "File not found"}, status_code=404)
