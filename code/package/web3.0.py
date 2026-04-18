from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import base64
import cv2
import numpy as np
import os
import torch

try:
    from flask_cors import CORS
except ImportError:
    CORS = None


# PyTorch 2.6+ uses weights_only=True by default. This patch keeps compatibility
# with trusted local YOLO checkpoints that were saved with full objects.
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat


MODEL_PATH = "yolov8n.pt"
DEFAULT_CONF_THRESHOLD = 0.5
CUSTOM_COLORS = {
    "person": (0, 255, 0),
    "car": (255, 0, 0),
    "default": (255, 165, 0),
}

app = Flask(__name__)
if CORS is not None:
    CORS(app)
else:
    print("flask_cors is not installed, CORS is disabled")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)
print(f"Model loaded on device: {device}")


def get_color(class_name):
    return CUSTOM_COLORS.get(class_name.lower(), CUSTOM_COLORS["default"])


def decode_image_from_request(req):
    """
    Support two request formats:
    1) application/json with field "image" (base64 string)
    2) multipart/form-data with file field "file"
    """
    if req.is_json:
        data = req.get_json(silent=True) or {}
        img_base64 = data.get("image", "")
        if not img_base64:
            return None, "Missing 'image' in JSON body"

        # Accept both raw base64 and data URL format.
        if "," in img_base64:
            img_base64 = img_base64.split(",", 1)[1]

        try:
            img_bytes = base64.b64decode(img_base64)
        except Exception:
            return None, "Invalid base64 image data"
    else:
        uploaded = req.files.get("file")
        if uploaded is None:
            return None, "Missing upload file field: file"
        img_bytes = uploaded.read()
        if not img_bytes:
            return None, "Uploaded file is empty"

    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Failed to decode image"
    return img, None


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "device": device,
            "model": MODEL_PATH,
        }
    )


@app.get("/")
def index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(base_dir, "index.html")
    return jsonify({"status": "error", "message": "index.html not found"}), 404


@app.get("/favicon.ico")
def favicon():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(base_dir, "favicon.ico")
    if os.path.exists(icon_path):
        return send_from_directory(base_dir, "favicon.ico")
    return "", 204


@app.post("/detect")
def detect():
    img, err = decode_image_from_request(request)
    if err:
        return jsonify({"status": "error", "message": err}), 400

    conf = request.args.get("conf", default=DEFAULT_CONF_THRESHOLD, type=float)
    if conf is None or conf <= 0 or conf > 1:
        conf = DEFAULT_CONF_THRESHOLD

    results = model(img, conf=conf)
    result = results[0]
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = get_color(cls_name)

        detections.append(
            {
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(score, 4),
                "box": [x1, y1, x2, y2],
            }
        )

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{cls_name} {score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    ok, encoded = cv2.imencode(".jpg", img)
    if not ok:
        return jsonify({"status": "error", "message": "Failed to encode output image"}), 500

    image_base64 = base64.b64encode(encoded).decode("utf-8")
    return jsonify(
        {
            "status": "success",
            "count": len(detections),
            "detections": detections,
            "image": image_base64,
        }
    )


if __name__ == "__main__":
    print("Backend started: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)