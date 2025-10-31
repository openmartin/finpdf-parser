import json
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from table_rec import table_rec_mineru_vl_server

logger = logging.getLogger(__name__)

def process_layout_results(layout_results: List[dict], output_folder: str = None) -> List[dict]:
    """
    根据版面识别结果，对不同类型的区域进行相应处理

    Args:
        output_folder: 输出的文件夹
        layout_results: 版面识别的boxes结果

    Returns:
        处理后的结果列表
    """
    if output_folder is None:
        output_folder = 'output'

    # 初始化OCR引擎（只需要中文和英文）
    ocr = PaddleOCR(lang="en",
                    use_doc_orientation_classify=False,  # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
                    use_doc_unwarping=False,  # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
                    use_textline_orientation=False,  # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
                    )

    processed_results = []

    for page_result in layout_results:
        processed_page_result = []

        image_path = page_result['image_path']
        pil_image = Image.open(image_path).convert("RGB")

        img_array = np.array(pil_image)

        ocr_result = ocr.predict(input=img_array, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
        text_res = convert_ocr_result(ocr_result[0].json['res'])
        save_path = Path(image_path)
        ocr_result[0].save_to_json(save_path=os.path.join(save_path.parent, save_path.stem + '_ocr.json'))
        for box_info in page_result['layout_info']:
            label = box_info['label']
            bbox = box_info['coordinate']
            result = {
                'type': label,
                'bbox': bbox,
                'score': box_info['score'],
                'res': []
            }
            # 根据不同类型进行处理
            if label == 'table':
                # 表格区域暂时只标记，后续可以添加专门的表格识别
                logger.info("检测到表格区域")
                cropped = pil_image.crop(bbox)
                table_html = table_rec_mineru_vl_server(cropped)
                result['res'] = {'html': table_html}

            elif label == 'figure':
                logger.info("检测到图片区域")


            elif label == 'equation':
                logger.info("检测到公式区域")

            else:
                logger.info("文本区域")
                res = filter_text_res(text_res, bbox)
                result['res'] = res
                # print(res)

            processed_page_result.append(result)

        # 保存到本地
        with open(os.path.join(os.path.join(save_path.parent, save_path.stem + '_processed.json')), "w", encoding="utf-8") as f:
            json.dump(processed_page_result, f, ensure_ascii=False, indent=4)

        processed_results.append(processed_page_result)

    return processed_results



def convert_ocr_result(ocr_result: dict):
    text_res = []
    for i, rec_poly in enumerate(ocr_result['rec_polys']):
        result = {
            "text": ocr_result['rec_texts'][i],
            "score": ocr_result['rec_scores'][i],
            "text_region": rec_poly
        }
        text_res.append(result)
    return text_res


def filter_text_res(text_res: dict, bbox: tuple):
    res = []
    for r in text_res:
        box = r["text_region"]
        rect = box[0][0], box[0][1], box[2][0], box[2][1]
        if has_intersection(bbox, rect):
            res.append(r)
    return res


def has_intersection(rect1, rect2):
    x_min1, y_min1, x_max1, y_max1 = rect1
    x_min2, y_min2, x_max2, y_max2 = rect2
    if x_min1 > x_max2 or x_max1 < x_min2:
        return False
    if y_min1 > y_max2 or y_max1 < y_min2:
        return False
    return True