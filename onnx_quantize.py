from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="model_fp32.onnx",
    model_output="model_dynamic_int8.onnx",
    weight_type=QuantType.QInt8,   # 仅权重量化
    per_channel=True
)