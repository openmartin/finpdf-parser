import traceback

import pymupdf  # PyMuPDF
import os
from pathlib import Path
import json
import logging
from typing import List
from paddleocr import LayoutDetection, PaddleOCR

from process_layout_results import process_layout_results
from recovery_to_markdown import convert_info_markdown

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


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

    # 初始化版面检测模型
    model = LayoutDetection(model_name="PP-DocLayoutV2", device="cpu")

    all_results = []

    for i, image_path in enumerate(image_paths):
        logger.info(f"正在处理第 {i + 1} 张图片: {image_path}")

        # 进行版面检测
        output = model.predict(image_path, batch_size=1, layout_nms=True)

        # 处理检测结果
        page_results = []
        for res in output:
            # 获取版面识别的boxes信息
            layout_boxes = res.json['res']['boxes']

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
                res.save_to_json(save_path=os.path.join(save_path.parent, save_path.stem + '_layout.json'))
                res.save_to_img(save_path=os.path.join(save_path.parent, save_path.stem + '_layout.png'))
            except:
                pass

        all_results.extend(page_results)

    # 保存所有结果到JSON文件
    # json_output_path = os.path.join(output_folder, "layout_results.json")
    # with open(json_output_path, 'w', encoding='utf-8') as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info(f"版面识别和文本处理完成！")
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
    os.makedirs(output_folder, exist_ok=True)

    # 打开PDF文件
    doc = pymupdf.open(pdf_path)
    image_paths = []

    # 逐页转换
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # 渲染页面为图片，使用新的API直接传dpi参数
        pix = page.get_pixmap(dpi=dpi)

        # 生成图片文件名
        image_name = f"page_{page_num + 1:03d}.png"
        image_path = os.path.join(output_folder, image_name)

        # 保存图片
        pix.save(image_path)
        image_paths.append(image_path)

        logger.info(f"已转换第 {page_num + 1} 页: {image_path}")

    doc.close()
    logger.info(f"转换完成！共生成 {len(image_paths)} 张图片")
    return image_paths


if __name__ == '__main__':
    # 示例用法
    pdf_file = "ibm-2q25-earnings-press-release.pdf"  # 请替换为实际的PDF文件路径

    if os.path.exists(pdf_file):
        try:
            # 使用默认DPI (150)
            images = pdf_to_images(pdf_file, dpi=150, output_folder='output')
            logger.info(f"生成的图片文件: {images}")

            # 进行版面识别
            if images:
                logger.info("\n开始进行版面识别...")
                layout_results = detect_layout(images, output_folder='output')
                logger.info(f"版面识别完成，共处理 {len(layout_results)} 个结果")
                processed_results = process_layout_results(layout_results, output_folder='output')
                for page_num, processed_page_result in enumerate(processed_results):
                    image_name = f"page_{page_num + 1:03d}.png"
                    convert_info_markdown(processed_page_result, 'output', image_name)


        except Exception as e:
            traceback.print_exc()
            logger.error(f"处理过程中出现错误: {e}")
    else:
        logger.info(f"请将PDF文件放在当前目录，或修改pdf_file变量指向正确的PDF文件路径")
        logger.info("示例: pdf_file = 'your_document.pdf'")