# 后端文件：app.py
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import base64
import numpy as np

app = Flask(__name__)

# 1. 加载YOLO模型（全局仅加载1次，省算力）
model = YOLO("yolov8n.pt")
CONF_THRESHOLD = 0.5
# 自定义框色（和之前脚本一致）
CUSTOM_COLORS = {"person":(0,255,0), "car":(255,0,0), "default":(255,165,0)}

def get_color(class_name):
    return CUSTOM_COLORS.get(class_name.lower(), CUSTOM_COLORS["default"])

# 2. 核心接口：接收图片，返回检测后的图片（Base64格式，网页能直接显示）
@app.route("/detect", methods=["POST"])
def detect_image():
    try:
        # 接收前端传的Base64图片
        data = request.json
        img_base64 = data["image"]
        # Base64转成OpenCV可处理的图片
        img_bytes = base64.b64decode(img_base64)
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # 执行YOLO检测，绘制自定义框色
        results = model(img, conf=CONF_THRESHOLD)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            conf = round(float(box.conf[0]), 2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = get_color(cls_name)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{cls_name} {conf}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 检测后的图片转Base64，返回给前端
        _, img_encoded = cv2.imencode(".jpg", img)
        img_result_base64 = base64.b64encode(img_encoded).decode("utf-8")
        return jsonify({"status": "success", "image": img_result_base64})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

# 3. 启动服务（允许跨域，前端能访问）
if __name__ == "__main__":
    from flask_cors import CORS
    CORS(app)  # 解决跨域问题
    print("🌐 网页服务启动中... 访问 http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)