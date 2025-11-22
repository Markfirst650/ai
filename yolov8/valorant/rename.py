import os

# 目标文件夹路径（当前目录下的labels文件夹）
path = r"D:\Users\Jay\Downloads\project-4-at-2025-11-09-15-35-9d0d61df\labels"

# 获取文件夹中的所有文件/文件夹列表
files = os.listdir(path)

# 对文件列表进行排序（确保重命名顺序固定）
files.sort()

# 循环遍历文件并按顺序重命名
for i, file in enumerate(files):
    # 构建原文件的完整路径
    old_path = os.path.join(path, file)
    # 构建新文件的完整路径（3位数字编号 + .txt后缀）
    new_name = f"{i + 1:03d}.txt"  # 用f-string更简洁地实现补零
    new_path = os.path.join(path, new_name)
    # 执行重命名
    os.rename(old_path, new_path)

# 完成提示
print("所有文件已重命名完成")