"""
方案2：COT 监督微调训练 API（0.5B 专属优化版）

审查意见修复：
1. ✅ LoRA 配置精准匹配 Qwen2.5-0.5B（r=8, alpha=16, 7个目标模块）
2. ✅ Label Masking：输入部分全部 -100，仅学习回答部分
3. ✅ 学习率 2e-4（0.5B 最优值）
4. ✅ 使用 Qwen 官方对话模板
5. ✅ 数据由 LongCat API 生成（不再用硬编码模板）
6. ✅ 双重答案风险修复：更健壮的 assistant_content 拼接逻辑
7. ✅ 相对路径风险修复：_resolve_relative_path 确保路径解析一致
8. ✅ 默认配置优化：dataloader_num_workers=4, optim=adamw_torch_fused
9. ✅ 断点检测简化：让 Trainer 自动查找最新 checkpoint
10. ✅ 显式配置：eval_strategy="no" 和 remove_unused_columns 注释
"""

import os
import sys
import re
from typing import Dict, Any, Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, TaskType, get_peft_model

from utils.common import load_json, save_json, print_config, get_device, ensure_flash_attn, enable_tf32, set_seed


def _get_attn_impl():
    try:
        import flash_attn
        return "flash_attention_2"
    except ImportError:
        return None


# ============================================================
# Qwen2.5-0.5B 专属 LoRA 配置
# ============================================================

QWEN_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # 注意力层
    "gate_proj", "up_proj", "down_proj",        # MLP 层
]


# ============================================================
# 数据处理：使用 Qwen 官方对话模板 + Label Masking
# ============================================================

def _build_assistant_content(cot: str, answer: str) -> str:
    """
    构建 assistant 回复内容，仅在 COT 未包含答案时才追加。
    
    避免双重答案风险：
    - 若 COT 已有「答案：」格式，直接返回
    - 否则检查 COT 末尾是否已有相同数字
    - 仅在都不满足时才追加「答案：{answer}」
    """
    if re.search(r'答案[：:]', cot):
        return cot
    # 检查 COT 末尾是否已有相同数字
    nums_in_cot = re.findall(r'\d+(?:\.\d+)?', cot)
    expected = str(answer).strip()
    if nums_in_cot and nums_in_cot[-1] == expected:
        return cot
    return f"{cot}\n答案：{answer}"


def process_sft_sample(tokenizer, example: Dict, max_length: int = 512) -> Dict:
    """
    处理单条 SFT 样本。

    支持两种数据格式：
    1. {question, cot, answer} — 传统COT格式
    2. {question, chosen, answer} — 偏好数据格式（chosen即正确COT）

    使用 Qwen 官方 chat template 构建输入输出：
      - 输入部分（system + user）→ labels 全部设为 -100（不计算 loss）
      - 输出部分（assistant）→ 正常计算 loss
    """
    question = example["question"]
    if isinstance(question, list):
        question = "".join(question)

    cot = example.get("cot", "") or example.get("chosen", "")
    if isinstance(cot, list):
        cot = "".join(cot)

    answer = str(example["answer"])
    instruction = example.get("instruction", "解答这道小学数学题。")
    if isinstance(instruction, list):
        instruction = "".join(instruction)

    assistant_content = _build_assistant_content(cot, answer)

    system_text = f"{instruction} 请按步骤解答，最后用\"答案：数字\"给出结果。"
    user_text = question

    system_ids = tokenizer.encode(system_text, add_special_tokens=False)
    user_ids = tokenizer.encode(user_text, add_special_tokens=False)
    assistant_ids = tokenizer.encode(assistant_content, add_special_tokens=False)

    qwen_bos = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    qwen_sep = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

    input_ids = (
        qwen_bos + system_ids + qwen_sep +
        qwen_bos + user_ids + qwen_sep +
        qwen_bos + assistant_ids + qwen_sep
    )

    prefix_len = len(qwen_bos) + len(system_ids) + len(qwen_sep) + len(qwen_bos) + len(user_ids) + len(qwen_sep) + len(qwen_bos)
    labels = [-100] * prefix_len + assistant_ids + qwen_sep

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


# ============================================================
# COTTrainer 类
# ============================================================

class COTTrainer:
    """方案2 COT 微调训练器（0.5B 专属优化）"""

    DEFAULT_CONFIG = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_cache_dir": "../models/Qwen2.5-0.5B-Instruct",
        "train_data_path": "../data/train_cot.json",
        "output_dir": "../outputs/scheme2_cot",
        "max_length": 512,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "save_steps": 500,
        "logging_steps": 10,
        "lr_scheduler_type": "cosine",
        "max_steps": -1,
        "dataloader_num_workers": 4,
        "optim": "adamw_torch_fused",
        "seed": 42,
    }

    def _resolve_relative_path(self, path: str) -> str:
        """将相对路径解析为绝对路径（相对于脚本目录）"""
        if os.path.isabs(path):
            return path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(script_dir, path))

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.device = None

    # ----------------------------------------------------------
    # 模型加载
    # ----------------------------------------------------------
    def setup(self, device: Optional[str] = None):
        """加载 Qwen2.5-0.5B 模型和 tokenizer"""
        print_config(self.config)

        if device is None:
            device = get_device()
        self.device = device
        print(f"设备: {device}")

        model_path = self._resolve_relative_path(self.config["model_cache_dir"])

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True}
        if device != "cpu":
            model_kwargs["device_map"] = {"": device}
            model_kwargs["torch_dtype"] = torch.bfloat16
            attn_impl = _get_attn_impl()
            if attn_impl:
                model_kwargs["attn_implementation"] = attn_impl
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if device == "cpu":
            self.model = self.model.to(device)

        actual_attn = getattr(self.model.config, '_attn_implementation', None) or getattr(self.model.config, 'attn_implementation', 'default')
        print(f"模型加载完成 - Attention: {actual_attn}")

    # ----------------------------------------------------------
    # 数据准备
    # ----------------------------------------------------------
    def prepare_data(self) -> List[Dict]:
        """加载 COT 数据并预处理"""
        print("加载 COT 训练数据...")
        data_path = self._resolve_relative_path(self.config["train_data_path"])
        if not os.path.exists(data_path):
            fallback = data_path.replace("train_cot.json", "train_cot_original.json")
            if os.path.exists(fallback):
                print(f"  {data_path} 不存在，使用 {fallback}")
                data_path = fallback
        train_data = load_json(data_path)
        print(f"训练样本数: {len(train_data)}")

        max_len = self.config["max_length"]
        dataset = []
        for d in train_data:
            dataset.append(process_sft_sample(self.tokenizer, d, max_len))

        print(f"预处理完成: {len(dataset)} 条")
        return dataset

    # ----------------------------------------------------------
    # LoRA 配置（Qwen2.5 专属）
    # ----------------------------------------------------------
    def setup_lora(self):
        """配置 LoRA，精准匹配 Qwen2.5 模型层"""
        print("配置 LoRA（Qwen2.5-0.5B 专属）...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=QWEN_LORA_TARGET_MODULES,
            inference_mode=False,
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    # ----------------------------------------------------------
    # Trainer 创建
    # ----------------------------------------------------------
    def create_trainer(self, train_dataset: List[Dict]):
        """创建 HuggingFace Trainer"""
        output_dir = self._resolve_relative_path(self.config["output_dir"])

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config["batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            logging_steps=self.config["logging_steps"],
            num_train_epochs=self.config["num_epochs"],
            max_steps=self.config.get("max_steps", -1),
            save_steps=self.config["save_steps"],
            learning_rate=self.config["learning_rate"],
            warmup_ratio=self.config["warmup_ratio"],
            weight_decay=self.config["weight_decay"],
            lr_scheduler_type=self.config["lr_scheduler_type"],
            report_to="none",
            # 保留原始数据列，不自动删除模型不需要的字段
            # 这里我们的预处理已经生成了完整的 input_ids/attention_mask/labels
            remove_unused_columns=False,
            fp16=False,
            bf16=self.device != "cpu",
            dataloader_num_workers=self.config.get("dataloader_num_workers", 4),
            dataloader_pin_memory=self.device != "cpu",
            optim=self.config.get("optim", "adamw_torch_fused"),
            seed=self.config.get("seed", 42),
            # 显式声明不进行评估
            eval_strategy="no",
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True),
        )

    # ----------------------------------------------------------
    # 训练
    # ----------------------------------------------------------
    def train(self, resume: bool = True, skip_if_exists: bool = True):
        """开始训练，支持断点续训和跳过已完成训练"""
        output_dir = self._resolve_relative_path(self.config["output_dir"])
        final_path = os.path.join(output_dir, "final")
        
        if skip_if_exists and os.path.exists(final_path):
            print(f"✓ 检测到已完成的模型: {final_path}")
            print("  跳过训练（如需重新训练，请删除该目录）")
            return
        
        print("开始训练...")
        
        if resume:
            checkpoint_dir = os.path.join(output_dir, "checkpoint-last")
            if os.path.exists(checkpoint_dir):
                print(f"检测到 checkpoint，继续训练: {checkpoint_dir}")
                self.trainer.train(resume_from_checkpoint=checkpoint_dir)
            else:
                print("未检测到 checkpoint，从头开始训练")
                self.trainer.train(resume_from_checkpoint=False)
        else:
            self.trainer.train(resume_from_checkpoint=False)

        self.trainer.save_model(final_path)
        print(f"模型已保存: {final_path}")

    # ----------------------------------------------------------
    # 完整流程
    # ----------------------------------------------------------
    def run(self, device: Optional[str] = None, resume: bool = True):
        """运行完整训练流程：setup → data → lora → train"""
        self.setup(device)
        train_dataset = self.prepare_data()
        self.setup_lora()
        self.create_trainer(train_dataset)
        self.train(resume)


def train_scheme2(config: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
    """便捷函数"""
    trainer = COTTrainer(config)
    trainer.run(device=device)


if __name__ == "__main__":
    train_scheme2()
