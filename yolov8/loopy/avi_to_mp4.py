import cv2
import os

def avi_to_mp4(avi_path, mp4_path=None, fps=30):
    """
    使用OpenCV将AVI视频转换为MP4格式
    :param avi_path: 输入AVI文件路径（支持绝对路径/相对路径）
    :param mp4_path: 输出MP4文件路径（默认与输入同目录同名称）
    :param fps: 输出视频帧率（默认30，建议与原视频一致）
    """
    # 检查输入文件是否存在
    if not os.path.isfile(avi_path):
        print(f"错误：文件不存在 → {avi_path}")
        return
    
    # 自动生成输出路径（若未指定）
    if not mp4_path:
        dir_name = os.path.dirname(avi_path)
        base_name = os.path.splitext(os.path.basename(avi_path))[0]
        mp4_path = os.path.join(dir_name, f"{base_name}.mp4")
    
    # 打开AVI视频
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 → {avi_path}")
        return
    
    # 获取原视频宽高（保持尺寸一致）
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 设置MP4编码器（H.264，兼容绝大多数播放器）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
    
    # 逐帧写入MP4文件
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 读取完毕
        out.write(frame)
        frame_count += 1
    
    # 释放资源
    cap.release()
    out.release()
    print(f"转换完成！\n输出文件：{mp4_path}\n总帧数：{frame_count}")

# ------------------- 替换为你的文件路径 -------------------
if __name__ == "__main__":
    # 输入AVI文件路径（务必用r开头避免转义错误）
    avi_file = r"D:\\Users\\Jay\Downloads\\8af76ee4e384980d6eaf167f46045af2.avi"  # 例如：r"D:\test.avi"
    # 调用转换函数（默认输出到同目录）
    avi_to_mp4(avi_file)