FROM ultralytics/ultralytics:8.3.96-cpu

WORKDIR /app

# The upstream/base build environment may carry host-local proxy values.
# Clear them so runtime image fetches do not try container-local 127.0.0.1 proxies.
ENV HTTP_PROXY="" \
    HTTPS_PROXY="" \
    http_proxy="" \
    https_proxy="" \
    NO_PROXY="" \
    no_proxy=""

# torch + ultralytics already included in base image, only install app deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY . .

# Export the segmentation model to ONNX at build time so the container starts
# with the faster ONNX runtime instead of PyTorch. opset 17 is the safest
# choice for onnxruntime >= 1.16 (already bundled in the base image).
RUN python - <<'EOF'
from ultralytics import YOLO
import os
pt = "adetailer/deepfashion2_yolov8s-seg.pt"
onnx = os.path.splitext(pt)[0] + ".onnx"
if not os.path.exists(onnx):
    YOLO(pt).export(format="onnx", imgsz=416, opset=17, simplify=True)
EOF

EXPOSE 9000
CMD exec gunicorn --bind "0.0.0.0:${PORT:-9000}" --workers 1 --threads 2 --preload --timeout 120 app:app
