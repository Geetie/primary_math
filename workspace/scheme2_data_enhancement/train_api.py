"""
方案2：COT 监督微调训练 API（0.5B 专属优化版）

审查意见修复：
1. ✅ LoRA 配置精准匹配 Qwen2.5-0.5B（r=8, alpha=16, 7个目标模块）
2. ✅ Label Masking：输入部分全部 -100，仅学习回答部分
3. ✅ 学习率 2e-4（0.5B 最优值）
4. ✅ 使用 Qwen 官方对话模板
5. ✅ 数据由 LongCat API 生成（不再用硬编码模板）
"""

import os
import sys
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

from utils.common import load_json, save_json, print_config, get_device


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

def process_sft_sample(tokenizer, example: Dict, max_length: int = 512) -> Dict:
    """
    处理单条 SFT 样本。

    使用 Qwen 官方 chat template 构建输入输出：
      - 输入部分（system + user）→ labels 全部设为 -100（不计算 loss）
      - 输出部分（assistant）→ 正常计算 loss

    Args:
        tokenizer: Qwen tokenizer
        example: 包含 question, cot, answer 的字典
        max_length: 最大序列长度
    Returns:
        {input_ids, attention_mask, labels}
    """
    question = example["question"]
    if isinstance(question, list):
        question = "".join(question)
    cot = example.get("cot", "")
    if isinstance(cot, list):
        cot = "".join(cot)
    answer = str(example["answer"])
    instruction = example.get("instruction", "解答这道小学数学题。")
    if isinstance(instruction, list):
        instruction = "".join(instruction)

    # 使用 Qwen 官方 chat template 构建 messages
    messages = [
        {"role": "system", "content": f"{instruction} 请按步骤解答，最后用\"答案：数字\"给出结果。"},
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"{cot}\n答案：{answer}"},
    ]

    # 用 chat_template 拼接，获取完整 token 序列
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    all_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    # 找到 assistant 回复的起始位置
    # 重建 assistant 之前的部分来计算长度
    prefix_messages = messages[:2]  # system + user
    prefix_text = tokenizer.apply_chat_template(prefix_messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]

    # 构建 labels：prefix 部分 -100，assistant 部分正常
    prefix_len = len(prefix_ids)
    labels = [-100] * prefix_len + all_ids[prefix_len:]

    # 截断
    if len(all_ids) > max_length:
        all_ids = all_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": all_ids,
        "attention_mask": [1] * len(all_ids),
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
        "train_data_path": "../data/train_cot_original.json",
        "output_dir": "../outputs/scheme2_cot",
        "max_length": 512,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "save_steps": 500,
        "logging_steps": 10,
        "lr_scheduler_type": "cosine",
        "max_steps": -1,
    }

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

        model_path = self.config["model_cache_dir"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device if device != "cpu" else None,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
        )
        if device == "cpu":
            self.model = self.model.to(device)

        self.model.enable_input_require_grads()
        print("模型加载完成")

    # ----------------------------------------------------------
    # 数据准备
    # ----------------------------------------------------------
    def prepare_data(self) -> List[Dict]:
        """加载 COT 数据并预处理"""
        print("加载 COT 训练数据...")
        data_path = self.config["train_data_path"]
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
        output_dir = self.config["output_dir"]

        # 检查断点
        has_checkpoint = False
        if os.path.exists(output_dir):
            has_checkpoint = any(
                f.startswith("checkpoint-")
                for f in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, f))
            )

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
            save_on_each_node=True,
            gradient_checkpointing=True,
            report_to="none",
            remove_unused_columns=False,
            fp16=self.device != "cpu",
            bf16=False,
            dataloader_num_workers=0,
            resume_from_checkpoint=has_checkpoint,
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
    def train(self, resume: bool = True):
        """开始训练，支持断点续训"""
        print("开始训练...")
        checkpoint = None
        if resume:
            output_dir = self.config["output_dir"]
            if os.path.exists(output_dir):
                checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
                if checkpoints:
                    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
                    checkpoint = os.path.join(output_dir, latest)
                    print(f"从 checkpoint 恢复: {checkpoint}")

        self.trainer.train(resume_from_checkpoint=checkpoint)

        final_path = os.path.join(self.config["output_dir"], "final")
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
