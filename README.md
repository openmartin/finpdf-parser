# 关于这个项目

这个项目主要是用于解析英文的财务报表PDF

针对目前很多的 OCR 工具或者模型，找到最适合英文财务报表解析的方法

版面解析部分使用 paddleocr 的 PP-DocLayoutV2 模型，文字使用 paddleocr 的 en-PP-OCRv5，这部分本地就可以运行

但是如果想要实现比较准确的表格识别，还是需要使用视觉模型来处理。英文财报里的表格都是没有线框的表格，使用 paddleocr 的小模型效果不好，容易错行错列

本地部署视觉模型推荐 [opendatalab/MinerU2.5-2509-1.2B](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) 或者 [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

没有显卡的话可以使用大模型云服务，本项目默认使用 qwen3-vl-235b-a22b-instruct 来识别表格，使用阿里云的大模型的 api

## 用法

```shell
python pdf_parse.py input.pdf ./output your-api-key
```

第一次运行会下载模型 PP-DocLayoutV2、PP-OCRv5_server_det、en_PP-OCRv5_mobile_rec

# About This Project

This project aims to identify the most suitable method for parsing English financial statement PDFs by evaluating various OCR tools and models.

For layout analysis, it uses the PP-DocLayoutV2 model from PaddleOCR, and for text recognition, it uses the en-PP-OCRv5 model. These components can be run locally.

However, for more accurate table recognition, a vision model is required. Tables in English financial reports are typically borderless, where PaddleOCR's smaller models are not very effective and often misidentify rows and columns.

For local deployment of a vision model, we recommend [opendatalab/MinerU2.5-2509-1.2B](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) or [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL).

If you don't have a GPU, you can use a cloud-based large model service. By default, this project uses the `qwen3-vl-235b-a22b-instruct` model for table recognition via the Alibaba Cloud API.

## Usage

```shell
python pdf_parse.py input.pdf ./output your-api-key
```

On the first run, the models PP-DocLayoutV2, PP-OCRv5_server_det, and en_PP-OCRv5_mobile_rec will be downloaded.