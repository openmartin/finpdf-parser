#!/usr/bin/env python3
"""
Qwen2-VL-2B-Instruct ONNX 转换脚本
将模型转换为三个 ONNX 文件：
1. embed_tokens.onnx - 文本嵌入层
2. vision_encoder.onnx - 视觉编码器
3. decoder_model_merged.onnx - 完整解码器（Transformer + LM Head + KV Cache）
"""

import os
import torch
import onnx
import onnxruntime as ort
from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.image_utils import load_image
import numpy as np
from pathlib import Path
import traceback

# 设置 Hugging Face 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class Qwen2VLToONNXConverter:
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct", output_dir: str = "model/onnx"):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"初始化转换器...")
        print(f"模型ID: {model_id}")
        print(f"输出目录: {self.output_dir}")

        self._load_model()

    def _load_model(self):
        """加载模型和处理器"""
        print("正在加载模型和处理器...")

        self.config = AutoConfig.from_pretrained(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
        ).eval()

        print("模型加载完成")

    def export_vision_encoder(self):
        """导出视觉编码器为 ONNX"""
        print("正在导出视觉编码器...")

        # 准备示例输入 - 使用标准图像尺寸
        batch_size = 1
        num_images = 1
        dummy_pixel_values = torch.randn(
            batch_size, num_images, 3, 224, 224, 
            dtype=torch.float32, 
            device=self.model.device
        )
        dummy_pixel_attention_mask = torch.ones(
            batch_size, num_images, 224, 224, 
            dtype=torch.bool, 
            device=self.model.device
        )
        dummy_image_grid_thw = torch.ones(
            batch_size, num_images, 3, 
            dtype=torch.long, 
            device=self.model.device
        )
        dummy_image_grid_thw[:, :, 0] = 224  # height
        dummy_image_grid_thw[:, :, 1] = 224  # width

        # 创建视觉编码器包装类
        class VisionWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.visual = model.visual

            def forward(self, pixel_values, image_grid_thw):
                # Qwen2VL 的视觉编码器直接处理 pixel_values
                # 需要将输入 reshape 为正确的格式
                batch_size, num_images, channels, height, width = pixel_values.shape
                pixel_values = pixel_values.view(batch_size * num_images, channels, height, width)

                # 调用视觉编码器的 patch_embed 和后续处理
                hidden_states = self.visual.patch_embed(pixel_values)

                # 准备 rotary_pos_emb
                grid_thw = image_grid_thw.view(-1, 3)
                rotary_pos_emb = self.visual.rot_pos_emb(grid_thw)

                # 通过所有层
                cu_seqlens = torch.arange(
                    0, (batch_size * num_images + 1) * hidden_states.shape[1],
                    step=hidden_states.shape[1],
                    dtype=torch.int32,
                    device=hidden_states.device
                )

                for blk in self.visual.blocks:
                    hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

                # 合并时空维度
                hidden_states = self.visual.merger(hidden_states)

                return hidden_states

        vision_wrapper = VisionWrapper(self.model)

        # 导出视觉编码器
        output_path = self.output_dir / "vision_encoder.onnx"
        torch.onnx.export(
            vision_wrapper,
            (dummy_pixel_values, dummy_image_grid_thw),
            str(output_path),
            input_names=["pixel_values", "image_grid_thw"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {0: "batch_size", 1: "num_images"},
                "image_grid_thw": {0: "batch_size", 1: "num_images"},
                "image_embeds": {0: "batch_size", 1: "seq_length"}
            },
            opset_version=18,
            do_constant_folding=True,
        )

        print(f"视觉编码器导出完成: {output_path}")

    def export_embed_tokens(self):
        """导出嵌入层为 ONNX"""
        print("正在导出嵌入层...")

        # 准备示例输入
        batch_size = 1
        seq_length = 512
        dummy_input_ids = torch.randint(
            0, self.config.text_config.vocab_size, 
            (batch_size, seq_length), 
            dtype=torch.long, 
            device=self.model.device
        )

        # 创建嵌入层包装类
        class EmbedTokensWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                # 修正: Qwen2VL 的 embed_tokens 直接在 language_model 中
                self.embed_tokens = model.language_model.embed_tokens

            def forward(self, input_ids):
                return self.embed_tokens(input_ids)

        embed_wrapper = EmbedTokensWrapper(self.model)

        # 导出嵌入层
        output_path = self.output_dir / "embed_tokens.onnx"
        torch.onnx.export(
            embed_wrapper,
            dummy_input_ids,
            str(output_path),
            input_names=["input_ids"],
            output_names=["inputs_embeds"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_length"},
                "inputs_embeds": {0: "batch_size", 1: "seq_length"}
            },
            opset_version=18,
            do_constant_folding=True,
        )

        print(f"嵌入层导出完成: {output_path}")

    def export_decoder_model_merged(self):
        """导出完整的解码器模型（包含 Transformer + LM Head + KV Cache）为 ONNX"""
        print("正在导出解码器模型...")

        # 准备示例输入
        batch_size = 1
        seq_length = 1  # 生成阶段通常是单 token
        num_key_value_heads = self.config.text_config.num_key_value_heads
        head_dim = self.config.text_config.hidden_size // self.config.text_config.num_attention_heads
        num_hidden_layers = self.config.text_config.num_hidden_layers

        dummy_inputs_embeds = torch.randn(
            batch_size, seq_length, self.config.text_config.hidden_size,
            dtype=torch.float32, 
            device=self.model.device
        )
        dummy_attention_mask = torch.ones(
            batch_size, seq_length, 
            dtype=torch.long, 
            device=self.model.device
        )

        # 可选的视觉输入参数（用于支持多模态输入）
        dummy_image_embeds = torch.randn(
            batch_size, 1, self.config.vision_config.hidden_size,
            dtype=torch.float32, 
            device=self.model.device
        )
        dummy_image_grid_thw = torch.ones(
            batch_size, 1, 3, 
            dtype=torch.long, 
            device=self.model.device
        )
        dummy_image_grid_thw[:, :, 0] = 224  # height
        dummy_image_grid_thw[:, :, 1] = 224  # width

        # 创建 past_key_values
        past_key_values = tuple([
            (
                torch.zeros(batch_size, num_key_value_heads, 0, head_dim, 
                           dtype=torch.float32, device=self.model.device),
                torch.zeros(batch_size, num_key_value_heads, 0, head_dim, 
                           dtype=torch.float32, device=self.model.device)
            )
            for _ in range(num_hidden_layers)
        ])

        # 创建解码器包装类
        class DecoderWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                # 修正:使用 language_model 而不是整个模型
                self.model = model.language_model

            def forward(self, inputs_embeds, attention_mask, image_embeds=None, image_grid_thw=None):
                # 始终以无缓存开始，避免传入 tuple 导致 get_seq_length 报错
                model_kwargs = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "past_key_values": None,
                    "use_cache": True,
                    "return_dict": True
                }

                # 添加视觉参数（如提供）
                if image_embeds is not None and image_grid_thw is not None:
                    model_kwargs["image_embeds"] = image_embeds
                    model_kwargs["image_grid_thw"] = image_grid_thw

                outputs = self.model(**model_kwargs)

                # 返回 logits 和展平的 present_key_values
                logits = outputs.logits
                present_kvs = []
                for key, value in outputs.past_key_values:
                    present_kvs.extend([key, value])

                return tuple([logits] + present_kvs)

        decoder_wrapper = DecoderWrapper(self.model)

        # 准备输入输出名称（不再将 past 作为输入）
        input_names = ["inputs_embeds", "attention_mask", "image_embeds", "image_grid_thw"]
        output_names = ["logits"]
        for i in range(num_hidden_layers):
            output_names.extend([f"present.{i}.key", f"present.{i}.value"])

        # 使用 dynamic_shapes（匹配 4 个输入的结构）
        from torch.export import Dim
        batch = Dim("batch_size")
        seq = Dim("seq_length")
        total_seq = Dim("total_seq_length")
        num_imgs = Dim("num_images")

        dynamic_shapes = (
            {0: batch, 1: seq},        # inputs_embeds: [B, T, H]
            {0: batch, 1: total_seq},  # attention_mask: [B, T_total]
            {0: batch, 1: num_imgs},   # image_embeds: [B, N, H_v]
            {0: batch, 1: num_imgs},   # image_grid_thw: [B, N, 3]
        )

        # 导出解码器
        output_path = self.output_dir / "decoder_model_merged.onnx"
        torch.onnx.export(
            decoder_wrapper,
            (dummy_inputs_embeds, dummy_attention_mask, dummy_image_embeds, dummy_image_grid_thw),
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
            do_constant_folding=True,
        )

        print(f"解码器模型导出完成: {output_path}")

    def validate_onnx_models(self):
        """验证导出的 ONNX 模型"""
        print("正在验证 ONNX 模型...")

        models_to_check = [
            "embed_tokens.onnx",
            "vision_encoder.onnx",
            "decoder_model_merged.onnx"
        ]

        all_valid = True
        for model_name in models_to_check:
            model_path = self.output_dir / model_name
            if model_path.exists():
                try:
                    onnx_model = onnx.load(str(model_path))
                    onnx.checker.check_model(onnx_model)

                    # 检查模型大小
                    file_size_mb = model_path.stat().st_size / (1024 * 1024)
                    print(f"✓ {model_name} 验证通过 (大小: {file_size_mb:.1f} MB)")
                except Exception as e:
                    print(f"✗ {model_name} 验证失败: {e}")
                    all_valid = False
            else:
                print(f"✗ {model_name} 文件不存在")
                all_valid = False

        return all_valid

    def convert_all(self):
        """执行完整的转换流程"""
        print("开始 Qwen2-VL-2B-Instruct ONNX 转换流程...")

        try:
            # 导出各个组件
            # self.export_embed_tokens()
            # self.export_vision_encoder()
            self.export_decoder_model_merged()

            # 验证模型
            if self.validate_onnx_models():
                print("\n🎉 所有 ONNX 模型转换成功完成！")
                print(f"模型文件保存在: {self.output_dir.absolute()}")

                # 列出生成的文件
                print("\n生成的文件:")
                for file_path in self.output_dir.glob("*.onnx"):
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"  - {file_path.name} ({size_mb:.1f} MB)")
            else:
                print("\n❌ 模型验证失败，请检查转换过程")

        except Exception as e:
            print(f"\n❌ 转换过程中出现错误: {e}")
            traceback.print_exc()

def main():
    """主函数"""
    # 可以通过环境变量或命令行参数自定义模型ID和输出目录
    model_id = os.getenv("MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
    output_dir = os.getenv("OUTPUT_DIR", "onnx-model")

    converter = Qwen2VLToONNXConverter(model_id, output_dir)
    converter.convert_all()

if __name__ == "__main__":
    main()
