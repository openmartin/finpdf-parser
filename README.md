# 关于这个项目

这个项目主要是用于解析英文的财务报表PDF

针对目前很多的 OCR 工具或者模型，找到最适合英文财务报表解析的方法

版面解析部分使用 paddleocr 的 PP-DocLayoutV2 模型，文字使用 paddleocr 的 en-PP-OCRv5，这部分本地就可以运行

但是如果想要实现比较准确的表格识别，还是需要使用视觉模型来处理。英文财报里的表格都是没有线框的表格，使用 paddleocr 的小模型效果不好，容易错行错列

本地部署视觉模型推荐 [opendatalab/MinerU2.5-2509-1.2B](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) 或者 [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

没有显卡的话可以使用大模型云服务，本项目默认使用 qwen3-vl-235b-a22b-instruct 来识别表格，使用阿里云的大模型的 api

## 用法

```shell
python pdf_parse.py --input_file ibm-2q25-earnings-press-release.pdf --output_dir output --api_key xxxx
```

