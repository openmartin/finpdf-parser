import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

import os
import traceback
import argparse
import multiprocessing
from typing import List
from pathlib import Path

import pymupdf  # PyMuPDF
from paddleocr import LayoutDetection

from process_layout_results import process_layout_results
from recovery_to_markdown import convert_info_markdown


logger = logging.getLogger(__name__)

def reset_logging():
    """
    重置日志配置，解决第三方库干扰问题
    """
    # 获取根日志器
    root_logger = logging.getLogger()

    # 清除所有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 重新配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )



def detect_layout(image_paths: List[str], output_folder: str = None) -> List[dict]:
    """
    对图片列表进行版面识别并保存结果到JSON

    Args:
        image_paths: 图片文件路径列表
        output_folder: 输出文件夹，如果为None则保存在图片同目录

    Returns:
        版面识别结果列表
    """
    if output_folder is None:
        output_folder = os.path.dirname(image_paths[0]) if image_paths else "output"

    os.makedirs(output_folder, exist_ok=True)

    total_images = len(image_paths)
    logger.info(f"阶段：版面识别 - 开始处理 {total_images} 张图片")

    # 获取系统最大可用CPU线程数
    cpu_threads = multiprocessing.cpu_count()

    # 初始化版面检测模型
    logger.info("阶段：正在初始化版面检测模型 (PP-DocLayoutV2)...")
    logger.info(f"使用CPU线程数: {cpu_threads}")
    model = LayoutDetection(model_name="PP-DocLayoutV2", device="cpu", enable_mkldnn=False, cpu_threads=cpu_threads)
    reset_logging()
    logger.info("版面检测模型初始化完成")

    all_results = []

    for i, image_path in enumerate(image_paths):
        logger.info(f"阶段：版面识别 - 正在处理第 {i + 1}/{total_images} 张图片: {os.path.basename(image_path)}")

        # 进行版面检测
        logger.info(f"第 {i + 1} 页：正在进行版面检测...")
        output = model.predict(image_path, batch_size=1, layout_nms=True)
        reset_logging()

        # 处理检测结果
        page_results = []
        for res in output:
            # 获取版面识别的boxes信息
            layout_boxes = res.json['res']['boxes']
            logger.info(f"第 {i + 1} 页：检测到 {len(layout_boxes)} 个版面区域")

            # 将检测结果转换为字典格式
            result_dict = {
                'page_number': i + 1,
                'image_path': image_path,
                'layout_info': layout_boxes,
            }
            page_results.append(result_dict)

            # 保存结果图片（可选）
            try:
                save_path = Path(image_path)
                layout_json_path = os.path.join(save_path.parent, save_path.stem + '_layout.json')
                layout_img_path = os.path.join(save_path.parent, save_path.stem + '_layout.png')
                res.save_to_json(save_path=layout_json_path)
                res.save_to_img(save_path=layout_img_path)
                logger.info(f"第 {i + 1} 页：版面识别结果已保存到 {os.path.basename(layout_json_path)} 和 {os.path.basename(layout_img_path)}")
            except Exception as e:
                logger.warning(f"第 {i + 1} 页：保存版面识别结果时出现警告: {e}")

        all_results.extend(page_results)
        logger.info(f"第 {i + 1}/{total_images} 页版面识别完成")

    # 保存所有结果到JSON文件
    # json_output_path = os.path.join(output_folder, "layout_results.json")
    # with open(json_output_path, 'w', encoding='utf-8') as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info(f"阶段：版面识别完成！共处理了 {total_images} 张图片，识别了 {len(all_results)} 个页面结果")
    return all_results


def pdf_to_images(pdf_path: str, dpi: int = 150, output_folder: str = None) -> List[str]:
    """
    将PDF文件转换为图片

    Args:
        pdf_path: PDF文件路径
        dpi: 图片分辨率，默认150
        output_folder: 输出文件夹，如果为None则保存在PDF同目录

    Returns:
        生成的图片文件路径列表
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    # 设置输出文件夹
    if output_folder is None:
        output_folder = os.path.splitext(pdf_path)[0] + "_images"

    # 创建输出文件夹
    logger.info(f"创建输出目录: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    # 打开PDF文件
    logger.info(f"阶段：PDF转图片 - 正在打开PDF文件: {os.path.basename(pdf_path)}")
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    logger.info(f"PDF文件包含 {total_pages} 页，使用DPI {dpi} 进行转换")

    image_paths = []

    # 逐页转换
    for page_num in range(total_pages):
        logger.info(f"阶段：PDF转图片 - 正在转换第 {page_num + 1}/{total_pages} 页...")
        page = doc.load_page(page_num)

        # 渲染页面为图片，使用新的API直接传dpi参数
        pix = page.get_pixmap(dpi=dpi)

        # 生成图片文件名
        image_name = f"page_{page_num + 1:03d}.png"
        image_path = os.path.join(output_folder, image_name)

        # 保存图片
        pix.save(image_path)
        image_paths.append(image_path)

        logger.info(f"第 {page_num + 1}/{total_pages} 页转换完成: {image_name}")

    doc.close()
    logger.info(f"阶段：PDF转图片完成！共生成 {len(image_paths)} 张图片，保存在 {output_folder}")
    return image_paths


def combine_markdown_files(output_dir: str, input_file: str) -> str:
    """
    合并所有生成的markdown文件为一个完整的markdown文档

    Args:
        output_dir: 输出目录路径
        input_file: 原始PDF文件路径

    Returns:
        合并后的markdown文件路径
    """
    logger.info("阶段：合并Markdown文件 - 开始查找和合并markdown文件")

    # 获取PDF文件名（不含扩展名）作为最终文件名
    pdf_name = Path(input_file).stem
    combined_md_path = os.path.join(output_dir, f"{pdf_name}_complete.md")
    logger.info(f"目标合并文件: {os.path.basename(combined_md_path)}")

    # 查找所有生成的markdown文件
    logger.info(f"正在搜索目录中的markdown文件: {output_dir}")
    md_files = []
    for file_name in os.listdir(output_dir):
        if file_name.endswith("_ocr.md"):
            md_files.append(os.path.join(output_dir, file_name))

    # 按文件名排序确保页面顺序正确
    md_files.sort()
    logger.info(f"找到 {len(md_files)} 个markdown文件需要合并")

    if not md_files:
        logger.warning("没有找到需要合并的markdown文件")
        return None

    # 合并所有markdown文件
    combined_content = []
    successful_files = 0

    for i, md_file in enumerate(md_files, 1):
        file_name = os.path.basename(md_file)
        logger.info(f"正在读取第 {i}/{len(md_files)} 个文件: {file_name}")

        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if content:
                combined_content.append(content)
                successful_files += 1
                logger.info(f"第 {i} 个文件读取成功，内容长度: {len(content)} 字符")
            else:
                logger.warning(f"第 {i} 个文件为空: {file_name}")

        except Exception as e:
            logger.error(f"读取第 {i} 个文件 {file_name} 时出错: {e}")
            continue

    logger.info(f"成功读取 {successful_files}/{len(md_files)} 个文件的内容")

    # 写入合并后的markdown文件
    try:
        logger.info(f"正在写入合并后的markdown文件...")
        with open(combined_md_path, 'w', encoding='utf-8') as f:
            combined_text = '\n\n'.join(combined_content)
            f.write(combined_text)

        logger.info(f"阶段：合并完成！文件保存为: {os.path.basename(combined_md_path)}")
        logger.info(f"合并统计 - 总文件数: {len(md_files)}, 成功合并: {successful_files}, 最终内容长度: {len(combined_text)} 字符")
        return combined_md_path

    except Exception as e:
        logger.error(f"写入合并文件时出错: {e}")
        return None


def main(input_file: str, output_dir: str, api_key: str):
    """
    主处理函数

    Args:
        input_file: 输入PDF文件路径
        output_dir: 输出目录路径
        api_key: 大模型API密钥
    """
    logger.info("=" * 60)
    logger.info("开始执行PDF解析和转换流程")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)

    # 设置API密钥环境变量或传递给相关函数使用
    os.environ['API_KEY'] = api_key

    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return False

    # 创建输出目录
    logger.info(f"准备输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 使用DPI 200转换PDF为图片
        logger.info("=" * 40)
        logger.info("第1阶段：PDF转换为图片")
        logger.info("=" * 40)
        images = pdf_to_images(input_file, dpi=200, output_folder=output_dir)
        logger.info(f"图片转换阶段完成，生成了 {len(images)} 张图片")

        # 进行版面识别
        if images:
            logger.info("=" * 40)
            logger.info("第2阶段：版面识别")
            logger.info("=" * 40)
            layout_results = detect_layout(images, output_folder=output_dir)
            logger.info(f"版面识别阶段完成，共处理 {len(layout_results)} 个页面")

            logger.info("=" * 40)
            logger.info("第3阶段：布局结果处理")
            logger.info("=" * 40)
            # 重置日志配置，确保后续日志正常输出
            reset_logging()
            processed_results = process_layout_results(layout_results, output_folder=output_dir)
            logger.info(f"布局结果处理完成，共处理 {len(processed_results)} 个页面")

            logger.info("=" * 40)
            logger.info("第4阶段：转换为Markdown格式")
            logger.info("=" * 40)
            for page_num, processed_page_result in enumerate(processed_results, 1):
                logger.info(f"正在转换第 {page_num}/{len(processed_results)} 页为Markdown...")
                image_name = f"page_{page_num:03d}.png"
                convert_info_markdown(processed_page_result, output_dir, image_name)
                logger.info(f"第 {page_num} 页Markdown转换完成")

            logger.info(f"Markdown转换阶段完成，共转换 {len(processed_results)} 个页面")

            # 合并所有markdown文件
            logger.info("=" * 40)
            logger.info("第5阶段：合并Markdown文件")
            logger.info("=" * 40)
            combined_file = combine_markdown_files(output_dir, input_file)
            if combined_file:
                logger.info(f"所有阶段完成！最终文件: {os.path.basename(combined_file)}")
            else:
                logger.warning("Markdown文件合并失败，但其他阶段已完成")

        logger.info("=" * 60)
        logger.info("PDF解析和转换流程全部完成！")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error("=" * 60)
        logger.error("处理过程中出现错误！")
        logger.error("=" * 60)
        traceback.print_exc()
        logger.error(f"错误详情: {e}")
        return False


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='PDF解析工具 - 将PDF转换为图片并进行版面识别，最终生成Markdown格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  python pdf_parse.py input.pdf ./output your-api-key
  python pdf_parse.py /path/to/document.pdf /path/to/output sk-xxxxxxxxxxxx
        '''
    )

    # 添加命令行参数
    parser.add_argument('input_file', 
                       help='输入的PDF文件路径')
    parser.add_argument('output_dir', 
                       help='输出目录路径')
    parser.add_argument('api_key', 
                       help='大模型API密钥')

    # 可选参数
    parser.add_argument('--dpi', type=int, default=200,
                       help='图片转换分辨率 (默认: 200)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细日志信息')

    # 解析命令行参数
    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 验证输入文件
    if not os.path.exists(args.input_file):
        logger.error(f"错误: 输入文件不存在 - {args.input_file}")
        parser.print_help()
        exit(1)

    if not args.input_file.lower().endswith('.pdf'):
        logger.error(f"错误: 输入文件必须是PDF格式 - {args.input_file}")
        exit(1)

    if not args.api_key.strip():
        logger.error("错误: API密钥不能为空")
        exit(1)

    # 调用主处理函数
    logger.info(f"开始处理PDF文件: {args.input_file}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"使用DPI: {args.dpi}")

    success = main(args.input_file, args.output_dir, args.api_key)

    if success:
        logger.info("处理完成！")
        exit(0)
    else:
        logger.error("处理失败！")
        exit(1)