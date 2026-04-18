from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# -------------------------- 1. 可配置参数（按需修改，超直观）--------------------------
MODEL_PATH = "yolov8n.pt"  # 模型路径（自动下载）
CONF_THRESHOLD = 0.5  # 置信度阈值（只显示≥0.5的目标）
SAVE_DIR = "detection_results"  # 结果保存目录
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
# 自定义检测框颜色（RGB格式，可新增/修改，键为类别名，值为(R,G,B)）
# 常见类别：person(人)、car(车)、cat(猫)、dog(狗)、bottle(瓶子)，全类别见YOLO官方文档
CUSTOM_COLORS = {
    "person": (0, 255, 0),  # 人：绿色
    "car": (255, 0, 0),  # 车：红色
    "cat": (0, 0, 255),  # 猫：蓝色
    "dog": (255, 255, 0),  # 狗：黄色
    "default": (255, 165, 0)  # 其他类别：橙色
}


# -------------------------- 2. 工具函数（辅助功能）--------------------------
def get_color(class_name):
    """根据类别名获取自定义颜色，没有则用默认色"""
    return CUSTOM_COLORS.get(class_name.lower(), CUSTOM_COLORS["default"])


def load_model():
    """加载模型（全局仅加载1次，省时间）"""
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ 模型加载成功！（{MODEL_PATH}）")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        return None


# -------------------------- 3. 核心功能（新增批量检测）--------------------------
def detect_image(model, img_path, use_custom_color=True):
    """检测单张图片（支持自定义框色）"""
    if not os.path.exists(img_path):
        print(f"❌ 图片不存在：{img_path}")
        return
    # 执行检测
    results = model(img_path, conf=CONF_THRESHOLD)
    img = results[0].orig_img  # 原始图片
    # 手动绘制检测框（替换默认框色）
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        conf = round(float(box.conf[0]), 2)
        # 获取坐标（左上x,y；右下x,y）
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # 绘制框和文字
        color = get_color(cls_name) if use_custom_color else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # 框
        cv2.putText(img, f"{cls_name} {conf}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 文字
    # 保存结果
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"result_{os.path.basename(img_path)}")
    cv2.imwrite(save_path, img)
    print(f"✅ 图片检测完成：{save_path}")


def detect_batch_images(model, folder_path):
    """批量检测文件夹内所有图片（支持jpg/png/jpeg）"""
    if not os.path.isdir(folder_path):
        print(f"❌ 文件夹不存在：{folder_path}")
        return
    # 获取所有图片文件
    img_formats = (".jpg", ".jpeg", ".png", ".bmp")
    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(img_formats)]
    if not img_files:
        print(f"❌ 文件夹内无图片文件：{folder_path}")
        return
    # 批量检测
    print(f"📂 开始批量检测，共{len(img_files)}张图片...")
    for i, img_file in enumerate(img_files, 1):
        img_path = os.path.join(folder_path, img_file)
        print(f"[{i}/{len(img_files)}] 检测中：{img_file}")
        detect_image(model, img_path)
    print(f"✅ 批量检测完成！所有结果保存至：{SAVE_DIR}")


def detect_video(model, video_path):
    """检测视频/摄像头（支持自定义框色）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频/摄像头：{video_path}")
        return
    # 视频写入配置
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "result_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print("🎥 视频检测中...（按 'q' 退出）")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 执行检测并绘制自定义框色
        results = model(frame, conf=CONF_THRESHOLD)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            conf = round(float(box.conf[0]), 2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = get_color(cls_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # 显示+保存
        cv2.imshow("YOLO Detection", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 视频检测完成：{save_path}")


# -------------------------- 4. 交互入口（新增批量检测选项）--------------------------
def main():
    print("=" * 60)
    print("          YOLO 全能检测工具（电脑版·增强版）")
    print("=" * 60)
    print("1. 检测单张图片")
    print("2. 批量检测文件夹内所有图片")
    print("3. 检测本地视频（如 test.mp4）")
    print("4. 摄像头实时检测")
    print("=" * 60)

    model = load_model()
    if not model:
        return

    choice = input("请输入功能编号（1/2/3/4）：")
    if choice == "1":
        img_path = input("请输入图片路径（如：test.jpg）：")
        detect_image(model, img_path)
    elif choice == "2":
        folder_path = input("请输入图片文件夹路径（如：./images）：")
        detect_batch_images(model, folder_path)
    elif choice == "3":
        video_path = input("请输入视频路径（如：test.mp4）：")
        detect_video(model, video_path)
    elif choice == "4":
        print("📹 正在启动摄像头...（按q退出）")
        detect_video(model, 0)
    else:
        print("❌ 输入错误！请输入 1-4")


if __name__ == "__main__":
    main()