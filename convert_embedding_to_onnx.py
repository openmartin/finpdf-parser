import os
import numpy as np
import torch
import torch.nn as nn
import onnx
from safetensors import safe_open

class EmbeddingModel(nn.Module):
    def __init__(self, embedding_weights):
        super().__init__()
        # 根据配置文件，hidden_size = 1536
        vocab_size, hidden_size = embedding_weights.shape
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.embedding.weight.data = torch.from_numpy(embedding_weights)
    
    def forward(self, input_ids):
        return self.embedding(input_ids)

def load_bf16_embeddings(bin_file_path):
    """加载 bf16 格式的 embedding 权重"""
    try:
        # 尝试用 safetensors 格式加载
        weights = {}
        with safe_open(bin_file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "embed" in key.lower():
                    weights[key] = f.get_tensor(key)
        
        if weights:
            # 找到 embedding 权重
            for key, tensor in weights.items():
                if "weight" in key:
                    return tensor.numpy()
        
        # 如果不是 safetensors，尝试直接加载
        # 这里假设是 PyTorch 的二进制格式
        state_dict = torch.load(bin_file_path, map_location='cpu')
        
        for key, tensor in state_dict.items():
            if "embed" in key.lower() and "weight" in key:
                return tensor.numpy()
        
        raise ValueError("未找到 embedding 权重")
        
    except Exception as e:
        print(f"加载权重文件失败: {e}")
        # 如果直接加载失败，尝试手动解析 bf16 数据
        return parse_raw_bf16(bin_file_path)

def parse_raw_bf16(bin_file_path):
    """手动解析 bf16 格式的原始数据"""
    # bf16 (bfloat16) 格式每个元素占 2 字节
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    
    # 假设数据是连续的 bf16 值
    bf16_data = np.frombuffer(data, dtype=np.uint16)
    
    # 转换为 float32 (这里需要 bf16 到 fp32 的转换)
    # 简化处理：直接重新解释为 float16 然后转为 float32
    fp16_data = bf16_data.astype(np.float16)
    fp32_data = fp16_data.astype(np.float32)
    
    # 根据配置文件，hidden_size = 1536
    # 估计 vocab_size = total_elements // hidden_size
    total_elements = len(fp32_data)
    hidden_size = 1536  # 从配置文件获取
    vocab_size = total_elements // hidden_size
    
    if vocab_size * hidden_size != total_elements:
        print(f"警告: 数据大小不匹配，总元素 {total_elements}, 预期 {vocab_size * hidden_size}")
        vocab_size = (total_elements + hidden_size - 1) // hidden_size
    
    # 重塑为 embedding 矩阵
    embedding_matrix = fp32_data[:vocab_size * hidden_size].reshape(vocab_size, hidden_size)
    
    return embedding_matrix

def convert_to_onnx(bin_file_path, output_onnx_path):
    """转换 embedding 权重为 ONNX 模型"""
    print("正在加载 embedding 权重...")
    embedding_weights = load_bf16_embeddings(bin_file_path)
    print(f"权重矩阵形状: {embedding_weights.shape}")
    
    print("正在创建 PyTorch 模型...")
    model = EmbeddingModel(embedding_weights)
    model.eval()
    
    # 创建示例输入
    batch_size = 1
    seq_length = 10
    dummy_input = torch.randint(0, embedding_weights.shape[0], (batch_size, seq_length))
    
    print("正在导出 ONNX 模型...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        input_names=['input_ids'],
        output_names=['inputs_embeds'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'inputs_embeds': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=18,
        do_constant_folding=True
    )
    
    # 验证 ONNX 模型
    try:
        onnx_model = onnx.load(output_onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX 模型验证成功！保存到: {output_onnx_path}")
    except Exception as e:
        print(f"ONNX 模型验证失败: {e}")

if __name__ == "__main__":
    # 根据您的项目结构
    bin_file = "model/embeddings_bf16.bin"
    onnx_file = "model/embed_tokens.onnx"
    
    if not os.path.exists(bin_file):
        print(f"错误: 文件 {bin_file} 不存在")
    else:
        convert_to_onnx(bin_file, onnx_file)
