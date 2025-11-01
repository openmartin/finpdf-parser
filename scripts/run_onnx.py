import os

import numpy as np
import onnxruntime
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoProcessor
from transformers.image_utils import load_image


def get_2d_position_ids(grid_thw: np.ndarray):
    """
    把 HF 的 Qwen2VLPositionId2DCache 逻辑原封不动搬到 numpy
    参数
    ----
    grid_thw: np.ndarray, int64, shape (n_images, 3)
              每行 [t, h, w] 单位=patch 格子数
    返回
    ----
    np.ndarray, int64, shape (N_patch_total, 2)
                每行 [row_id, col_id]
    """
    # 1. 先按最大格子生成坐标轴
    t, h, w = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
    h_max, w_max = int(h.max()), int(w.max())
    h_range = np.arange(h_max, dtype=np.int64)  # [0..h_max)
    w_range = np.arange(w_max, dtype=np.int64)  # [0..w_max)

    # 2. 对每张图/每帧构造 (ti*hi*wi, 2)
    position_ids = []
    for ti, hi, wi in zip(t, h, w):
        h_id = h_range[:hi].reshape(1, hi, 1)
        w_id = w_range[:wi].reshape(1, 1, wi)
        # 广播到 (ti, hi, wi)
        h_id = np.broadcast_to(h_id, (ti, hi, wi))
        w_id = np.broadcast_to(w_id, (ti, hi, wi))
        pos = np.stack([h_id, w_id], axis=-1).reshape(-1, 2)  # (ti*hi*wi, 2)
        position_ids.append(pos)

    return np.concatenate(position_ids, axis=0)  # (N_patch_total, 2)


def build_position_ids_for_onnx(grid_thw: np.ndarray,
                                text_len: int,
                                batch_size: int):
    """
    一次性生成 ONNX 要的 3-D position_ids
    参数
    ----
    grid_thw   : (n_images, 3)  int64
    text_len   : 文本 token 个数
    batch_size : 默认 1（若 ONNX 把 batch 轴写死为 3 就传 3）
    返回
    ----
    np.ndarray, int64, shape (batch_size, L_total, 2)
    """
    # 1. vision 2D 编号
    image_pos = get_2d_position_ids(grid_thw)  # (N_patch, 2)

    # 2. text 部分继续顺序编号（row=0, col=继续累加）
    max_col = int(image_pos[:, 1].max())
    text_pos = np.stack([np.zeros(text_len, dtype=np.int64),
                         np.arange(text_len, dtype=np.int64) + max_col + 1],
                        axis=-1)  # (text_len, 2)

    # 3. 拼接
    total_pos = np.concatenate([image_pos, text_pos], axis=0)  # (L, 2)

    # 4. 扩 batch 维
    return np.broadcast_to(total_pos[None, :, :],
                           (batch_size, total_pos.shape[0], 2))



os.environ["HF_HOME"] = "D:\\hf_local"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "D:\\hf_local\\hub"
os.environ["TRANSFORMERS_CACHE"]       = r"D:\hf_local\hub"
os.environ["HF_HUB_CACHE"]             = r"D:\hf_local\hub"
os.environ["HUGGINGFACE_HUB_CACHE"]    = r"D:\hf_local\hub"

model_id = "opendatalab/MinerU2.5-2509-1.2B"

config = AutoConfig.from_pretrained("D:\\hf_local\\hub\\models--opendatalab--MinerU2.5-2509-1.2B\\snapshots\\879e58bdd9566632b27a8a81f0e2961873311f67")

processor = AutoProcessor.from_pretrained("D:\\hf_local\\hub\\models--opendatalab--MinerU2.5-2509-1.2B\\snapshots\\879e58bdd9566632b27a8a81f0e2961873311f67")

# vision_model_path = hf_hub_download(model_id, subfolder="onnx", filename="vision_encoder.onnx",
#                                     cache_dir="D:\\hf_local\\hub", local_files_only=True)  # graph
vision_model_path = "mineru\\onnx\\vision_encoder.onnx"

# embed_model_path = hf_hub_download(model_id, subfolder="onnx", filename="embed_tokens_fp16.onnx",
#                                    cache_dir="D:\\hf_local\\hub", local_files_only=True)  # graph
embed_model_path = "mineru\\onnx\\embed_tokens.onnx"

# decoder_model_path = hf_hub_download(model_id, subfolder="onnx", filename="decoder_model_merged.onnx",
#                                      cache_dir="D:\\hf_local\\hub", local_files_only=True)  # graph

decoder_model_path = "mineru\\onnx\\decoder_model_merged.onnx"


## Load sessions
vision_session = onnxruntime.InferenceSession(vision_model_path)
embed_session = onnxruntime.InferenceSession(embed_model_path)
decoder_session = onnxruntime.InferenceSession(decoder_model_path)

## Set config values
num_key_value_heads = config.text_config.num_key_value_heads

# 确保这两个属性存在，它们通常是 Qwen2VLTextConfig 的一部分
hidden_size = config.text_config.hidden_size
num_attention_heads = config.text_config.num_attention_heads

head_dim = hidden_size // num_attention_heads # 使用整除

num_hidden_layers = config.text_config.num_hidden_layers

eos_token_id = config.text_config.eos_token_id

image_token_id = config.image_token_id

# Create input messages
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Table Recognition:"}
        ]
    },
]

print("load image")
image = load_image(
    r'E:\python_workspace\finpdf-parser\output\table_001.png'
)

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")

# Convert torch tensors to numpy for onnxruntime
for k, v in list(inputs.items()):
    print(k)
    if isinstance(v, torch.Tensor):
        inputs[k] = v.cpu().numpy()

## Prepare decoder inputs
batch_size = inputs['input_ids'].shape[0]
past_key_values = {
    f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
    for layer in range(num_hidden_layers)
    for kv in ('key', 'value')
}

inputs['input_ids'] = inputs['input_ids'].astype(np.int64)
inputs['attention_mask'] = inputs['attention_mask'].astype(np.int64)

image_features = None
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 3. Generation loop
max_new_tokens = 4096
generated_tokens = np.array([[]], dtype=np.int64)
for i in range(max_new_tokens):
    inputs_embeds = embed_session.run(None, {'input_ids': input_ids})[0]

    if image_features is None:
        ## Only compute vision features if not already computed
        image_features = vision_session.run(None, dict(
            pixel_values=inputs['pixel_values'],
            grid_thw=inputs['image_grid_thw']
        ))[0]
        inputs_embeds[inputs['input_ids'] == image_token_id] = image_features.reshape(-1, image_features.shape[-1])

    grid_thw = inputs["image_grid_thw"]  # (n_images, 3)
    text_len = inputs["input_ids"].shape[1]
    position_ids = build_position_ids_for_onnx(grid_thw, text_len,
                                               batch_size=inputs['input_ids'].shape[0])
    logits, *present_key_values = decoder_session.run(None, dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        **past_key_values,
    ))

    ## Update values for next generation loop
    input_ids = logits[:, -1].argmax(-1, keepdims=True)
    attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=attention_mask.dtype)],
                                    axis=-1)
    for j, key in enumerate(past_key_values):
        past_key_values[key] = present_key_values[j]

    generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
    # 添加检查：如果达到最大生成长度或者遇到结束标记，则停止生成
    if (input_ids == eos_token_id).all():
        break
    # print(len(generated_tokens[0]))
    if len(generated_tokens[0]) >= 1024:
        break

    # print(processor.decode(input_ids[0]), end='')
print()


# 4. Do something with the final output
print(processor.batch_decode(generated_tokens, skip_special_tokens=False)[0])