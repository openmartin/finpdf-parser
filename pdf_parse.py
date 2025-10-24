import pymupdf  # PyMuPDF
import os
import json
from typing import List
from paddleocr import LayoutDetection, PaddleOCR


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
                    print(f"文本区域OCR识别完成，识别到 {len(region_texts)} 个文本片段")

                except Exception as e:
                    print(f"OCR识别失败: {e}")
                    result['text_content'] = ""

            elif label == 'table':
                # 表格区域暂时只标记，后续可以添加专门的表格识别
                result['note'] = "表格区域，需要专门的表格识别处理"
                print("检测到表格区域")

            elif label == 'figure':
                # 图片区域
                result['note'] = "图片区域"
                print("检测到图片区域")

            else:
                # 其他类型暂时只记录
                result['note'] = f"未处理的类型: {label}"
                print(f"检测到未处理的类型: {label}")

        processed_results.append(result)

    return processed_results


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
    model = LayoutDetection(model_name="PP-DocLayoutV2")

    all_results = []

    for i, image_path in enumerate(image_paths):
        print(f"正在处理第 {i + 1} 张图片: {image_path}")

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
                res.save_to_json(save_path=output_folder)
                res.save_to_img(save_path=output_folder)
            except:
                pass

        all_results.extend(page_results)

    # 保存所有结果到JSON文件
    json_output_path = os.path.join(output_folder, "layout_results.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"版面识别和文本处理完成！结果已保存到: {json_output_path}")
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

        print(f"已转换第 {page_num + 1} 页: {image_path}")

    doc.close()
    print(f"转换完成！共生成 {len(image_paths)} 张图片")
    return image_paths


if __name__ == '__main__':
    # 示例用法
    pdf_file = "ibm-2q25-earnings-press-release.pdf"  # 请替换为实际的PDF文件路径

    if os.path.exists(pdf_file):
        try:
            # 使用默认DPI (150)
            images = pdf_to_images(pdf_file, dpi=150, output_folder='output')
            print(f"生成的图片文件: {images}")

            # 进行版面识别
            if images:
                print("\n开始进行版面识别...")
                layout_results = detect_layout(images, output_folder='output')
                print(f"版面识别完成，共处理 {len(layout_results)} 个结果")

        except Exception as e:
            print(f"处理过程中出现错误: {e}")
    else:
        print(f"请将PDF文件放在当前目录，或修改pdf_file变量指向正确的PDF文件路径")
        print("示例: pdf_file = 'your_document.pdf'")