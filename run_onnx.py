import os

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoProcessor
from transformers.image_utils import load_image

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ["HF_HOME"] = "D:\\hf_local"
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["HF_HUB_CACHE"] = "D:\\hf_local\\hub"
# os.environ["TRANSFORMERS_CACHE"]       = r"D:\hf_local\hub"
# os.environ["HF_HUB_CACHE"]             = r"D:\hf_local\hub"
# os.environ["HUGGINGFACE_HUB_CACHE"]    = r"D:\hf_local\hub"

model_id = "Qwen/Qwen2-VL-2B-Instruct"

config = AutoConfig.from_pretrained(model_id)

processor = AutoProcessor.from_pretrained(model_id)

# vision_model_path = hf_hub_download(model_id, subfolder="onnx", filename="vision_encoder.onnx",
#                                     cache_dir="D:\\hf_local\\hub", local_files_only=True)  # graph
vision_model_path = "model/onnx/visual.onnx"

# embed_model_path = hf_hub_download(model_id, subfolder="onnx", filename="embed_tokens_fp16.onnx",
#                                    cache_dir="D:\\hf_local\\hub", local_files_only=True)  # graph
embed_model_path = "model/embed_tokens.onnx"

# decoder_model_path = hf_hub_download(model_id, subfolder="onnx", filename="decoder_model_merged.onnx",
#                                      cache_dir="D:\\hf_local\\hub", local_files_only=True)  # graph

decoder_model_path = "model/onnx/llm.onnx"

## Create session options to handle type compatibility
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

## Set provider options to handle potential type issues
providers = ['CPUExecutionProvider']

## Load sessions
vision_session = onnxruntime.InferenceSession(vision_model_path, sess_options=sess_options, providers=providers)
embed_session = onnxruntime.InferenceSession(embed_model_path, sess_options=sess_options, providers=providers)
decoder_session = onnxruntime.InferenceSession(decoder_model_path, sess_options=sess_options, providers=providers)

## Set config values
num_key_value_heads = config.text_config.num_key_value_heads
head_dim = config.text_config.head_dim
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
    "output\\page_001.png"
)

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="np")

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
            pixel_attention_mask=inputs['pixel_attention_mask'].astype(np.bool_)
        ))[0]
        inputs_embeds[inputs['input_ids'] == image_token_id] = image_features.reshape(-1, image_features.shape[-1])

    logits, *present_key_values = decoder_session.run(None, dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
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