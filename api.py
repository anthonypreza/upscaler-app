import torch
import RRDBNet_arch as arch

from fastapi import FastAPI
from upscale import upscale
from pydantic import BaseModel


class UpscaleRequest(BaseModel):
    img: str


device = torch.device("cpu")
model_path = (
    "models/RRDB_ESRGAN_x4.pth"  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
)
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

app = FastAPI()


@app.post("/upscale")
async def execute(request: UpscaleRequest):
    upscaled = upscale(model, bytes(request.img.encode("utf-8")))
    return {"upscaled": upscaled}