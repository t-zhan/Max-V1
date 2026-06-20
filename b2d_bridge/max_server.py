import argparse
import base64

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import Response

from models.max_v1.max_carla import Max

app = FastAPI()
model = None


@app.post("/max_predict")
async def predict(request: Request):
    body = await request.json()
    rgb = []
    for b64 in body["images"]:
        jpg = base64.b64decode(b64)
        bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    wp, _ = model.carla_generate(rgb, body["ego_speed"], body["command_idx"])
    payload = wp.detach().cpu().float().contiguous().numpy().tobytes()
    return Response(content=payload, media_type="application/octet-stream")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global model
    model = Max.from_pretrained(args.model_path).eval().cuda()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
