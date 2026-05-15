import base64
import binascii
import io
import os
import time
import torch
import requests
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO

# Default to all available cores locally; set CPU_CORES=2 on FC to match instance size
_cpu_count = int(os.environ.get("CPU_CORES", str(os.cpu_count() or 2)))
torch.set_num_threads(_cpu_count)
# Use direct assignment so CPU_CORES always takes effect before ONNX session is created
os.environ["OMP_NUM_THREADS"] = str(_cpu_count)
os.environ["OPENBLAS_NUM_THREADS"] = str(_cpu_count)
os.environ["ORT_NUM_THREADS"] = str(_cpu_count)

app = Flask(__name__)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "adetailer/deepfashion2_yolov8s-seg.pt"),
)
# Inference image size: smaller = faster. 416 is a good balance for fashion detection.
# Override via INFER_IMGSZ env var if needed (e.g. 320 for max speed, 640 for max accuracy).
INFER_IMGSZ = int(os.environ.get("INFER_IMGSZ", "416"))

MAX_IMAGE_SIZE_MB = float(os.environ.get("MAX_IMAGE_SIZE_MB", "10"))
MAX_IMAGE_SIZE_BYTES = int(MAX_IMAGE_SIZE_MB * 1024 * 1024)

# Base64 expands payloads by roughly 4/3; keep the decoded image limit unchanged.
app.config["MAX_CONTENT_LENGTH"] = int(MAX_IMAGE_SIZE_BYTES * 4 / 3) + 4096

# Prefer ONNX model for faster CPU inference; fall back to .pt if not exported yet
_onnx_path = os.path.splitext(MODEL_PATH)[0] + ".onnx"
if os.path.exists(_onnx_path):
    print(f"Loading ONNX model from: {_onnx_path}")
    model = YOLO(_onnx_path)
else:
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
print("Model loaded.")
print(f"  Classes ({len(model.names)}): { {k: v for k, v in sorted(model.names.items())} }")

print("-" * 50)
print("Runtime config:")
print(f"  PORT          : {os.environ.get('PORT', '9000')}")
print(f"  MODEL_PATH    : {MODEL_PATH}")
print(f"  INFER_IMGSZ   : {INFER_IMGSZ}")
print(f"  CPU_CORES     : {_cpu_count}")
print(f"  MAX_IMAGE_MB  : {MAX_IMAGE_SIZE_MB:g} MB")
print(f"  torch version : {torch.__version__}")
print("-" * 50)

# Warmup: first inference is always slower due to JIT/kernel init
print("Warming up model...")
_warmup_img = Image.new("RGB", (INFER_IMGSZ, INFER_IMGSZ))
model(_warmup_img, imgsz=INFER_IMGSZ, verbose=False)
del _warmup_img
print("Warmup done.")


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
    data = request.get_json(force=True, silent=True) or {}
    try:
        img = load_image_from_request(data)
    except Exception as e:
        return jsonify({"error": f"Failed to load image: {str(e)}"}), 400

    t0 = time.time()

    # retina_masks=False skips segmentation mask decoding (we only need boxes)
    results = model(img, imgsz=INFER_IMGSZ, retina_masks=False, verbose=False)
    regions = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = round(float(box.conf[0]), 3)
            cls = int(box.cls[0])
            label = model.names[cls]
            regions.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "confidence": conf,
            })

    elapsed_ms = int((time.time() - t0) * 1000)
    print(f"[regionPredict] size={img.width}x{img.height} regions={len(regions)} "
          f"labels={[r['label'] for r in regions]} infer={elapsed_ms}ms")

    return jsonify({
        "regions": regions,
        "image_size": {"width": img.width, "height": img.height},
    })


@app.route("/")
def debug_page():
    return render_template("debug.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9000))
    app.run(host="0.0.0.0", port=port, debug=False)
