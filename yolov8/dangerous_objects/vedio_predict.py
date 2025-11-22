from ultralytics import YOLO
import cv2

# 加载模型
yolo = YOLO("best.pt", task="detect")

# 视频路径
video_path = r"D:\highlight\wonderfulVideos13561100055503969284\b4b55ae4-5950-4894-bcdf-ed83e5ef78c1\5caad65292e3b98fc16ee94de48c4228.mp4"

# 打开视频文件（逐帧读取，避免一次性加载全部帧）
cap = cv2.VideoCapture(video_path)

# 获取视频参数（用于保存输出视频）
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置输出视频（保持原视频尺寸和帧率）
output_path = "output2.mp4"  # 可自定义输出路径
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 视频读取结束
    
    # 逐帧推理（保持原始参数，仅单帧处理）
    results = yolo(frame, show=True ,conf = 0.5)  # 显示推理结果
    
    # 提取标注后的帧并保存
    annotated_frame = results[0].plot()
    out.write(annotated_frame)
    
    # 按'q'键提前退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源（必须执行，否则可能导致视频文件损坏）
cap.release()
out.release()
cv2.destroyAllWindows()