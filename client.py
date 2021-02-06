import base64
import os

import cv2
import requests
import numpy as np

for img_file in os.listdir("LR"):
    with open(f"LR/{img_file}", "rb") as f:
        img = f.read()
        img_bytes = base64.b64encode(img)

        res = requests.post(
            "http://localhost:8000/upscale", json={"img": img_bytes.decode("utf-8")}
        )
        data = res.json()
        output = data["upscaled"]

        returned_img = base64.b64decode(output)
        upscaled = np.frombuffer(returned_img, dtype=np.uint8)
        upscaled = cv2.imdecode(upscaled, flags=1)

        cv2.imwrite("results/{}_esrgan.png".format(img_file.split(".")[0]), upscaled)
