"""upscale.py
Module that provides a function for upscaling images with ESRGAN.
"""

import base64
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    import RRDBNet_arch as arch

device = torch.device("cpu")


def upscale(model: "arch.RRDBNet", image_bytes: bytes):
    original = base64.b64decode(image_bytes)
    img = np.frombuffer(original, dtype=np.uint8)
    img = cv2.imdecode(img, flags=1)

    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    success, encoded_img = cv2.imencode(".png", output)
    encoded_img = encoded_img.tobytes()
    encoded_img = base64.b64encode(encoded_img)
    return encoded_img


if __name__ == "__main__":
    import os
    import RRDBNet_arch as arch

    model_path = "models/RRDB_ESRGAN_x4.pth"  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    for img_file in os.listdir("LR"):
        with open(f"LR/{img_file}", "rb") as f:
            img = f.read()
            img_bytes = base64.b64encode(img)

            output = upscale(model, img_bytes)

            returned_img = base64.b64decode(output)
            upscaled = np.frombuffer(returned_img, dtype=np.uint8)
            upscaled = cv2.imdecode(upscaled, flags=1)

            cv2.imwrite(
                "results/{}_esrgan.png".format(img_file.split(".")[0]), upscaled
            )
