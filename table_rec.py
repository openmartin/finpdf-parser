import base64
import io
import os
import time

import openai
from PIL import Image
from paddleocr import TableRecognitionPipelineV2, PaddleOCRVL

from otsl2html import convert_otsl_to_html

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-vl-235b-a22b-instruct"

TABLE_REC_PROMPT ="""
## Role
你是一位有多年经验的OCR表格识别专家。

## Goals
需要通过给定的图片，识别表格里的内容，并以 HTML 表格结果格式输出。

## Constrains
- 完整识别每个单元格内容，包括占位符如“-”、“/”等；
- 表格结构必须与图片完全一致；
- 注意处理合并单元格（rowspan/colspan）；
- 不要遗漏、不要编造；
- 输出必须是标准 HTML 表格格式（使用 <table>, <tr>, <td> 标签）；
- 不要添加任何解释或额外文字，只输出 HTML 表格。

## Initialization
请仔细思考后，直接输出 HTML 表格结果。
"""



def table_rec_qwen3_vl(image: Image.Image):
    client = openai.OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=BASE_URL
    )
    b64 = image_to_base64(image)
    data_url = f"data:image/png;base64,{b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": TABLE_REC_PROMPT },
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }
    ]

    print(messages)

    response = client.chat.completions.create(
        model = MODEL_NAME,
        messages = messages,
        temperature=0,
        stream = False,
        # 若后端支持关闭思考模式，可额外传参：
        extra_body={"enable_thinking": False}  # 非标准字段，视服务端支持而定
    )

    # 5. 拿到结果
    html_table = response.choices[0].message.content
    print(html_table)
    return html_table


def image_to_base64(img: Image.Image) -> str:
    """
    将 PIL.Image.Image 对象转换为 Base64 编码字符串。

    Args:
        img (Image.Image): 要转换的 PIL.Image.Image 对象。
        format (str): 图像格式，如 'JPEG', 'PNG'。'JPEG' 适合照片，'PNG' 适合透明图。
        quality (int): 当格式为 'JPEG' 时，控制图像质量 (1-100)。

    Returns:
        str: Base64 编码的字符串，例如 'data:image/jpeg;base64,/9j/4AAQSkZJRgABA...'
             如果编码失败则返回 None。
    """
    if img is None:
        return None

    # 创建一个内存中的字节流缓冲区
    buffered = io.BytesIO()

    # 将图像保存到缓冲区中
    img.save(buffered, format='png')

    # 从缓冲区获取二进制图像数据
    img_byte_data = buffered.getvalue()

    # 对二进制数据进行 Base64 编码
    base64_str = base64.b64encode(img_byte_data).decode('utf-8')

    return base64_str

## result is not good enough
# def table_rec_pp(image: Image.Image):
#     pipeline = TableRecognitionPipelineV2(use_doc_orientation_classify=False, use_doc_unwarping=False, use_layout_detection=False)
#     output = pipeline.predict("output/table_001.png", use_doc_orientation_classify=False, use_doc_unwarping=False, use_layout_detection=False)
#     for res in output:
#         res.print()  ## 打印预测的结构化输出
#         res.save_to_img("output/")
#         res.save_to_xlsx("output/")
#         res.save_to_html("output/")
#         res.save_to_json("output/")


## can not run with no GPU
# def table_rec_pp_vl(image: Image.Image):
#     pipeline = PaddleOCRVL(use_doc_orientation_classify=False, use_doc_unwarping=False, use_layout_detection=False, use_chart_recognition=False)
#     output = pipeline.predict("output/table_001.png", prompt_label='table', use_doc_orientation_classify=False, use_doc_unwarping=False, use_layout_detection=False, use_chart_recognition=False)
#     for res in output:
#         res.print()  ## 打印预测的结构化输出
#         res.save_to_json(save_path="output")
#         res.save_to_markdown(save_path="output")


# 使用 opendatalab/MinerU2.5-2509-1.2B
# def table_rec_mineru_vl_server(image: Image.Image):
#     b64 = image_to_base64(image)
#     data_url = f"data:image/png;base64,{b64}"
#
#     base_url = "http://192.168.100.1:8080/v1"
#     model_name = "MinerU2.5-2509-1.2B"
#
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "\nTable Recognition:"
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"{data_url}" # 表格图像
#                     }
#                 }
#             ]
#         }
#     ]
#
#     client = openai.OpenAI(
#         api_key="API_KEY",
#         base_url=base_url
#     )
#
#     response = client.chat.completions.create(
#         model = model_name,
#         messages = messages,
#         temperature=0,
#         stream = False,
#         extra_body={"skip_special_tokens": False}
#     )
#
#     # 5. 拿到结果
#     otsl_table = response.choices[0].message.content
#     html_table = convert_otsl_to_html(otsl_table)
#     print(html_table)
#     return html_table


if __name__ == "__main__":
    image = Image.open("output/table_001.png")
    start = time.time()
    table_rec_qwen3_vl(image)
    end = time.time()
    elapsed = end - start
    print(f"花费 {elapsed} s")

