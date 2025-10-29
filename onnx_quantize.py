import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="mineru/onnx/decoder_model_merged.onnx",
    model_output="mineru/onnx/decoder_model_merged_int8.onnx",
    weight_type=QuantType.QInt8,   # 仅权重量化
    per_channel=True
)

quantize_dynamic(
    model_input="mineru/onnx/embed_tokens.onnx",
    model_output="mineru/onnx/embed_tokens_int8.onnx",
    weight_type=QuantType.QInt8,   # 仅权重量化
    per_channel=True
)

quantize_dynamic(
    model_input="mineru/onnx/vision_encoder.onnx",
    model_output="mineru/onnx/vision_encoder_int8.onnx",
    weight_type=QuantType.QInt8,   # 仅权重量化
    per_channel=True,
    op_types_to_quantize=['SequenceEmpty', 'ScatterND', 'Neg', 'Loop', 'Cos', 'MatMul', 'Tile', 'Sigmoid', 'Sqrt', 'Einsum', 'LayerNormalization', 'Unsqueeze', 'Shape', 'Expand', 'Sin', 'ConcatFromSequence', 'Concat', 'Reshape', 'Cast', 'Softmax', 'Gather', 'Erf', 'CumSum', 'Add', 'Pad', 'ReduceMax', 'Split', 'Where', 'Transpose', 'Gemm', 'Range', 'SplitToSequence', 'Squeeze', 'Mul', 'ConstantOfShape', 'Div', 'Equal', 'Slice', 'If', 'Flatten']
)

# model = onnx.load("mineru/onnx/vision_encoder.onnx")
# op_set = set()
# for node in model.graph.node:
#     # print(node.op_type)
#     op_set.add(node.op_type)
#
#
# print(op_set)