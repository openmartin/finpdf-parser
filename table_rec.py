import base64
import io
import time

import openai
from PIL import Image

BASE_URL = "xxx"
MODEL_NAME = "Qwen3-VL-235B-A22B-Thinking"
API_KEY = "xxxx"

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

client = openai.OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def table_rec(image: Image.Image):
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


if __name__ == "__main__":
    image = Image.open("output/table_001.png")
    start = time.time()
    table_rec(image)
    end = time.time()
    print("花费 ", end - start)

