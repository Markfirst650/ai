from PIL import Image
import os
import argparse


def batch_convert_webp(input_dir, output_dir, output_format="jpg", quality=95):
    # 初始化统计变量
    total_files = 0  # 总文件数
    total_webp = 0  # 总WebP文件数
    converted_success = 0  # 新增成功转换数
    converted_failed = 0  # 新增转换失败数
    non_webp_files = 0  # 非WebP文件数
    skipped_existing = 0  # 已存在的转换文件（跳过）

    if output_format not in ["jpg", "png"]:
        raise ValueError("输出格式仅支持 'jpg' 或 'png'")

    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        total_files += len(files)
        relative_path = os.path.relpath(root, input_dir)
        current_output_dir = os.path.join(output_dir, relative_path)
        os.makedirs(current_output_dir, exist_ok=True)

        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.lower().endswith(".webp"):
                total_webp += 1
                output_file = f"{os.path.splitext(file_name)[0]}.{output_format}"
                output_path = os.path.join(current_output_dir, output_file)

                # 核心优化：如果输出文件已存在，直接跳过
                if os.path.exists(output_path):
                    skipped_existing += 1
                    print(f"⏭️ 已存在，跳过：{output_path}")
                    continue

                # 否则执行转换
                try:
                    with Image.open(file_path) as img:
                        if output_format == "jpg" and img.mode in ["RGBA", "P"]:
                            background = Image.new("RGB", img.size, (255, 255, 255))
                            background.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
                            background.save(output_path, "JPEG", quality=quality)
                        else:
                            img.save(output_path, output_format.upper())
                    converted_success += 1
                    print(f"✅ 新增转换：{file_path} -> {output_path}")
                except Exception as e:
                    converted_failed += 1
                    print(f"❌ 新增转换失败：{file_path}（错误：{str(e)}）")
            else:
                non_webp_files += 1

    # 输出统计结果（新增了“已跳过的已转换文件”）
    print("\n" + "=" * 60)
    print(f"转换任务完成！")
    print(f"总文件数：{total_files} 个")
    print(f"其中WebP文件：{total_webp} 个")
    print(f"  - 新增成功转换：{converted_success} 个（本次新增的图片）")
    print(f"  - 新增转换失败：{converted_failed} 个（本次新增的图片）")
    print(f"  - 已存在并跳过：{skipped_existing} 个（之前已转换的旧图片）")
    print(f"非WebP文件（已跳过）：{non_webp_files} 个")
    print(f"输出目录：{os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebP批量转换工具（支持增量转换）")
    parser.add_argument("--input", required=True, help="WebP图片输入目录（必填）")
    parser.add_argument("--output", required=True, help="转换后图片输出目录（必填）")
    parser.add_argument("--format", default="png", choices=["jpg", "png"], help="输出格式（默认jpg）")
    parser.add_argument("--quality", type=int, default=95, help="JPG质量（1-100，默认95）")

    args = parser.parse_args()

    batch_convert_webp(
        input_dir=args.input,
        output_dir=args.output,
        output_format=args.format,
        quality=args.quality
    )