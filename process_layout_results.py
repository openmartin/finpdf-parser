import json
import logging
import os
import multiprocessing
from pathlib import Path
from typing import List
import asyncio
import concurrent.futures
from functools import partial

import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

from table_rec import table_rec_qwen3_vl

# 确保日志配置（如果还没有配置的话）
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

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


async def process_table_async(cropped_image, page_idx, box_idx, layout_boxes_count, executor):
    """
    异步处理单个表格识别
    """
    logger.info(f"第 {page_idx} 页：正在进行第 {box_idx}/{layout_boxes_count} bbox 表格异步识别...")

    # 使用线程池执行阻塞的表格识别
    loop = asyncio.get_event_loop()
    table_html = await loop.run_in_executor(executor, table_rec_qwen3_vl, cropped_image)

    logger.info(f"第 {page_idx} 页：第 {box_idx} 个表格识别完成")
    return table_html


def process_layout_results_sync(layout_results: List[dict], output_folder: str = None) -> List[dict]:
    """
    同步版本的处理函数（保持向后兼容）
    """
    return asyncio.run(process_layout_results_async(layout_results, output_folder))


async def process_layout_results_async(layout_results: List[dict], output_folder: str = None) -> List[dict]:
    """
    异步版本的主处理函数，支持全局并行处理表格识别
    """
    if output_folder is None:
        output_folder = 'output'

    total_pages = len(layout_results)
    logger.info(f"开始处理版面识别结果，总共 {total_pages} 页")

    # 获取系统最大可用CPU线程数
    cpu_threads = multiprocessing.cpu_count()

    # 创建线程池用于异步处理表格识别
    # 使用较少的线程数避免API并发限制和内存压力
    max_table_workers = min(2, cpu_threads)  # 稍微增加并发数
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_table_workers)

    # 初始化OCR引擎（只需要中文和英文）
    logger.info("阶段：正在初始化OCR引擎...")
    logger.info(f"使用CPU线程数: {cpu_threads}")
    logger.info(f"表格识别并发线程数: {max_table_workers}")
    ocr = PaddleOCR(lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    cpu_threads=cpu_threads,
                    )
    reset_logging()
    logger.info("OCR引擎初始化完成")

    # 存储所有页面的处理结果和表格任务信息
    processed_results = []
    all_table_tasks = []  # 所有表格识别任务
    table_metadata = []   # 表格元数据：(page_idx, box_idx, result_obj_reference)

    try:
        # 第一阶段：处理所有页面的非表格内容，收集所有表格任务
        logger.info("阶段1：处理非表格内容并收集表格任务...")

        for page_idx, page_result in enumerate(layout_results, 1):
            logger.info(f"正在处理第 {page_idx}/{total_pages} 页的非表格内容")
            processed_page_result = []

            image_path = page_result['image_path']
            logger.info(f"正在加载图片: {os.path.basename(image_path)}")
            pil_image = Image.open(image_path).convert("RGB")

            img_array = np.array(pil_image)

            logger.info(f"正在对第 {page_idx} 页进行OCR识别...")
            ocr_result = ocr.predict(input=img_array, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
            reset_logging()
            text_res = convert_ocr_result(ocr_result[0].json['res'])
            logger.info(f"第 {page_idx} 页OCR识别完成，识别到 {len(text_res)} 个文本区域")

            save_path = Path(image_path)
            ocr_result[0].save_to_json(save_path=os.path.join(save_path.parent, save_path.stem + '_ocr.json'))

            layout_boxes_count = len(page_result['layout_info'])
            table_count_in_page = 0

            # 处理每个布局区域
            for box_idx, box_info in enumerate(page_result['layout_info'], 1):
                label = box_info['label']
                bbox = box_info['coordinate']

                result = {
                    'type': label,
                    'bbox': bbox,
                    'score': box_info['score'],
                    'res': []  # 表格的结果会在后面填入
                }

                if label == 'table':
                    # 收集表格处理任务
                    table_count_in_page += 1
                    logger.info(f"第 {page_idx} 页：发现表格 {table_count_in_page}，加入全局处理队列")
                    cropped = pil_image.crop(bbox)

                    # 创建异步任务但不立即执行
                    task = asyncio.create_task(
                        process_table_async(cropped, page_idx, box_idx, layout_boxes_count, executor)
                    )
                    all_table_tasks.append(task)

                    # 记录表格元数据，用于后续结果回填
                    table_metadata.append({
                        'page_idx': page_idx - 1,  # 在processed_results中的索引
                        'box_idx': box_idx - 1,    # 在页面结果中的索引
                        'result_obj': result       # 结果对象的引用
                    })

                elif label == 'figure':
                    logger.info(f"第 {page_idx} 页：检测到图片区域 ({box_idx}/{layout_boxes_count})")

                elif label == 'equation':
                    logger.info(f"第 {page_idx} 页：检测到公式区域 ({box_idx}/{layout_boxes_count})")

                else:
                    # 处理文本区域
                    res = filter_text_res(text_res, bbox)
                    result['res'] = res

                processed_page_result.append(result)

            processed_results.append(processed_page_result)
            if table_count_in_page > 0:
                logger.info(f"第 {page_idx} 页非表格内容处理完成，发现 {table_count_in_page} 个表格")
            else:
                logger.info(f"第 {page_idx} 页处理完成（无表格）")

        # 第二阶段：并行处理所有表格
        total_tables = len(all_table_tasks)
        if total_tables > 0:
            logger.info(f"阶段2：开始并行处理所有 {total_tables} 个表格...")
            table_results = await asyncio.gather(*all_table_tasks, return_exceptions=True)

            # 第三阶段：将表格结果填回对应位置
            logger.info("阶段3：将表格识别结果填回对应位置...")
            for idx, (metadata, table_html) in enumerate(zip(table_metadata, table_results)):
                page_idx = metadata['page_idx']
                box_idx = metadata['box_idx']

                if isinstance(table_html, Exception):
                    logger.error(f"表格 {idx + 1}（第{page_idx + 1}页）识别失败: {table_html}")
                    processed_results[page_idx][box_idx]['res'] = {'html': '', 'error': str(table_html)}
                else:
                    processed_results[page_idx][box_idx]['res'] = {'html': table_html}

            logger.info(f"所有 {total_tables} 个表格识别完成并填回结果")
        else:
            logger.info("未发现表格，跳过表格处理阶段")

        # 第四阶段：保存所有结果到本地
        logger.info("阶段4：保存所有处理结果到本地...")
        for page_idx, (page_result, processed_page_result) in enumerate(zip(layout_results, processed_results), 1):
            image_path = page_result['image_path']
            save_path = Path(image_path)
            processed_file_path = os.path.join(save_path.parent, save_path.stem + '_processed.json')

            with open(processed_file_path, "w", encoding="utf-8") as f:
                json.dump(processed_page_result, f, ensure_ascii=False, indent=4)
            logger.info(f"第 {page_idx} 页处理结果已保存到: {os.path.basename(processed_file_path)}")

    finally:
        # 确保线程池被正确关闭
        executor.shutdown(wait=True)

    logger.info(f"所有页面处理完成！共处理了 {total_pages} 页，{len(all_table_tasks)} 个表格")
    return processed_results


def process_layout_results(layout_results: List[dict], output_folder: str = None) -> List[dict]:
    """
    根据版面识别结果，对不同类型的区域进行相应处理

    注意：此函数现在使用异步处理来优化表格识别性能
    如果您的代码运行在异步环境中，建议直接调用 process_layout_results_async()

    Args:
        output_folder: 输出的文件夹
        layout_results: 版面识别的boxes结果

    Returns:
        处理后的结果列表
    """
    return process_layout_results_sync(layout_results, output_folder)



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