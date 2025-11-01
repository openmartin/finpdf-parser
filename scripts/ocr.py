import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="en",
                use_doc_orientation_classify=False,  # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
                use_doc_unwarping=False,  # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
                use_textline_orientation=False,  # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
                )
pil_image = Image.open("output/text_001.png").convert("RGB")
img_array = np.array(pil_image)
ocr_result = ocr.predict(input=img_array, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
for res in ocr_result:
    res.print() ## 打印预测的结构化输出
    res.save_to_img(save_path="../output") ## 保存当前图像的公式可视化结果
    res.save_to_json(save_path="../output") ## 保存当前图像的结构化json结果