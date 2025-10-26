import logging
from typing import List

from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

def process_layout_results(layout_results: List[dict]) -> List[dict]:
    """
    根据版面识别结果，对不同类型的区域进行相应处理

    Args:
        layout_results: 版面识别的boxes结果

    Returns:
        处理后的结果列表
    """
    # 初始化OCR引擎（只需要中文和英文）
    ocr = PaddleOCR(use_angle_cls=False, lang="en")

    processed_results = []

    for page_result in layout_results:
        image_path = page_result['image_path']
        for box_info in page_result['layout_info']:
            label = box_info['label']
            coordinate = box_info['coordinate']
            result = {
                'label': label,
                'coordinate': coordinate,
                'score': box_info['score']
            }
            # 根据不同类型进行处理
            if label in ['text', 'doc_title', 'paragraph_title', 'figure_title', 'vision_footnote', 'footnote']:
                # 对文本区域进行OCR识别
                try:
                    # 提取区域坐标
                    x1, y1, x2, y2 = coordinate

                    # 使用PaddleOCR进行文本识别
                    ocr_result = ocr.ocr(image_path, det=True, rec=True, cls=True)

                    # 筛选出在当前区域内的文本
                    region_texts = []
                    if ocr_result and ocr_result[0]:
                        for line in ocr_result[0]:
                            if len(line) >= 2:
                                # 获取文本框坐标
                                text_box = line[0]
                                text_content = line[1][0] if line[1] else ""

                                # 计算文本框中心点
                                box_center_x = sum(point[0] for point in text_box) / 4
                                box_center_y = sum(point[1] for point in text_box) / 4

                                # 检查文本是否在当前区域内
                                if (x1 <= box_center_x <= x2 and y1 <= box_center_y <= y2):
                                    region_texts.append(text_content)

                    result['text_content'] = ' '.join(region_texts) if region_texts else ""
                    logger.info(f"文本区域OCR识别完成，识别到 {len(region_texts)} 个文本片段")

                except Exception as e:
                    logger.error(f"OCR识别失败: {e}")
                    result['text_content'] = ""

            elif label == 'table':
                # 表格区域暂时只标记，后续可以添加专门的表格识别
                result['note'] = "表格区域，需要专门的表格识别处理"
                logger.info("检测到表格区域")

            elif label == 'figure':
                # 图片区域
                result['note'] = "图片区域"
                logger.info("检测到图片区域")

            elif label in ['header', 'footer']:
                logger.info(f"检测到{label}区域, 跳过")

            else:
                # 其他类型暂时只记录
                result['note'] = f"未处理的类型: {label}"
                logger.info(f"检测到未处理的类型: {label}")

        processed_results.append(result)

    return processed_results