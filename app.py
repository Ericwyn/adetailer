import base64
import binascii
import io
import os
import time
import numpy as np
import requests
import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "adetailer/deepfashion2_yolov8s-seg.pt"),
)
MAX_IMAGE_SIZE_MB = float(os.environ.get("MAX_IMAGE_SIZE_MB", "10"))
MAX_IMAGE_SIZE_BYTES = int(MAX_IMAGE_SIZE_MB * 1024 * 1024)
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
NUM_THREADS = int(os.environ.get("NUM_THREADS", "1"))
MAX_DET = int(os.environ.get("MAX_DET", "50"))

# Base64 expands payloads by roughly 4/3; keep the decoded image limit unchanged.
app.config["MAX_CONTENT_LENGTH"] = int(MAX_IMAGE_SIZE_BYTES * 4 / 3) + 4096

torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(NUM_THREADS)

print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded.")

_warmup_img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
with torch.inference_mode():
    model(_warmup_img, imgsz=IMGSZ, conf=CONF_THRESHOLD, max_det=MAX_DET, retina_masks=False, verbose=False)
del _warmup_img
print("Model warmed up.")


def _prescale_image(img: Image.Image) -> tuple:
    """Downscale to at most IMGSZ on the longest side before passing to YOLO.
    Avoids allocating a large tensor when the source image far exceeds inference size."""
    w, h = img.size
    scale = min(IMGSZ / w, IMGSZ / h)
    if scale >= 1.0:
        return img, 1.0
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR), scale


def fetch_image(image_url):
    with requests.get(image_url, timeout=15, stream=True) as resp:
        resp.raise_for_status()
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(f"Image exceeds {MAX_IMAGE_SIZE_MB:g} MB limit")

        chunks = []
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > MAX_IMAGE_SIZE_BYTES:
                raise ValueError(f"Image exceeds {MAX_IMAGE_SIZE_MB:g} MB limit")
            chunks.append(chunk)

    return Image.open(io.BytesIO(b"".join(chunks))).convert("RGB")


def decode_base64_image(image_base64):
    if "," in image_base64 and image_base64.split(",", 1)[0].lower().endswith(";base64"):
        image_base64 = image_base64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("Invalid base64 image") from e

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(f"Image exceeds {MAX_IMAGE_SIZE_MB:g} MB limit")

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def load_image_from_request(data):
    image_url = data.get("image_url") or data.get("imageurl")
    image_base64 = data.get("image_base64")

    if image_url and image_base64:
        raise ValueError("Use either image_url/imageurl or image_base64, not both")
    if image_base64:
        return decode_base64_image(image_base64)
    if image_url:
        return fetch_image(image_url)

    raise ValueError("image_url, imageurl, or image_base64 is required")


@app.route("/regionPredict", methods=["POST"])
def region_predict():
    t_start = time.monotonic()
    data = request.get_json(force=True, silent=True) or {}
    try:
        img = load_image_from_request(data)
    except Exception as e:
        return jsonify({"error": f"Failed to load image: {str(e)}"}), 400
    t_load = time.monotonic()

    orig_w, orig_h = img.size
    scaled_img, scale = _prescale_image(img)

    with torch.inference_mode():
        results = model(scaled_img, imgsz=IMGSZ, conf=CONF_THRESHOLD, max_det=MAX_DET, retina_masks=False, verbose=False)
    t_infer = time.monotonic()

    regions = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = round(float(box.conf[0]), 3)
            cls = int(box.cls[0])
            label = model.names[cls]
            regions.append({
                "bbox": [
                    int(x1 / scale), int(y1 / scale),
                    int(x2 / scale), int(y2 / scale),
                ],
                "label": label,
                "confidence": conf,
            })

    t_end = time.monotonic()
    print(
        f"regionPredict | image={orig_w}x{orig_h} scale={scale:.2f} regions={len(regions)} | "
        f"load={1000*(t_load-t_start):.1f}ms infer={1000*(t_infer-t_load):.1f}ms "
        f"post={1000*(t_end-t_infer):.1f}ms total={1000*(t_end-t_start):.1f}ms"
    )

    return jsonify({
        "regions": regions,
        "image_size": {"width": orig_w, "height": orig_h},
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/")
def debug_page():
    return render_template("debug.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9000))
    app.run(host="0.0.0.0", port=port, debug=False)
