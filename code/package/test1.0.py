from ultralytics import YOLO
import cv2
import os
import torch

# 兼容 PyTorch 2.6+ 默认 weights_only=True 导致的旧权重加载失败。
# 仅在你信任权重来源时使用该兼容逻辑（官方 yolov8n.pt 可视为可信）。
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

# -------------------------- 1. 配置参数（可按需修改）--------------------------
MODEL_PATH = "yolov8n.pt"  # 模型路径（没有会自动下载）
CONF_THRESHOLD = 0.5  # 置信度阈值（低于这个的目标不显示）
SAVE_DIR = "detection_results"  # 检测结果保存目录


# -------------------------- 2. 核心功能封装 --------------------------
def load_model():
    """加载YOLO模型，返回模型对象"""
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ 模型加载成功！（模型：{MODEL_PATH}）")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        return None


def detect_image(model, img_path):
    """检测单张图片"""
    if not os.path.exists(img_path):
        print(f"❌ 图片不存在：{img_path}")
        return
    # 执行检测
    results = model(img_path, conf=CONF_THRESHOLD)
    # 保存结果（带检测框的图片）
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"result_{os.path.basename(img_path)}")

    # 关键修改：用 plot() 获取带检测框的图像，再用 cv2 保存
    annotated_img = results[0].plot()  # 获取可视化结果（BGR格式）
    cv2.imwrite(save_path, annotated_img)  # 保存图像

    print(f"✅ 图片检测完成！结果保存至：{save_path}")


def detect_video(model, video_path):
    """检测视频（支持本地视频/摄像头）"""
    # 摄像头：video_path=0；本地视频：传入路径（如"test.mp4"）
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频/摄像头：{video_path}")
        return

    # 视频写入器（保存检测后的视频）
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
        # 执行检测并绘制框
        results = model(frame, conf=CONF_THRESHOLD)
        annotated_frame = results[0].plot()  # 带检测框的帧
        # 显示+保存
        cv2.imshow("YOLO Detection", annotated_frame)
        out.write(annotated_frame)
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 视频检测完成！结果保存至：{save_path}")


# -------------------------- 3. 交互入口（用户操作界面）--------------------------
def main():
    print("=" * 50)
    print("          YOLO 模型检测工具（电脑版）")
    print("=" * 50)
    print("1. 检测单张图片")
    print("2. 检测本地视频（如 test.mp4）")
    print("3. 摄像头实时检测")
    print("=" * 50)

    # 加载模型（全局只加载1次，避免重复耗时）
    model = load_model()
    if not model:
        return

    # 接收用户选择
    choice = input("请输入功能编号（1/2/3）：")
    if choice == "1":
        img_path = input("请输入图片路径（如：test.jpg）：")
        detect_image(model, img_path)
    elif choice == "2":
        video_path = input("请输入视频路径（如：test.mp4）：")
        detect_video(model, video_path)
    elif choice == "3":
        print("📹 正在启动摄像头...")
        detect_video(model, 0)  # 0 表示默认摄像头
    else:
        print("❌ 输入错误！请输入 1/2/3")


# 运行入口
if __name__ == "__main__":
    main()