import io
import os
import requests
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

app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE_BYTES

print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded.")


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


@app.route("/regionPredict", methods=["POST"])
def region_predict():
    data = request.get_json(force=True, silent=True) or {}
    image_url = data.get("image_url")
    if not image_url:
        return jsonify({"error": "image_url is required"}), 400

    try:
        img = fetch_image(image_url)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 400

    results = model(img)
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
