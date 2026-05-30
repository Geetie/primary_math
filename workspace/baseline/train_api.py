"""
Baseline方案训练API（0.5B 专属优化版）

修复：
1. ✅ LoRA: r=8, alpha=16, dropout=0.05
2. ✅ Label Masking: 使用 Qwen chat_template，输入部分 -100
3. ✅ 学习率 2e-4, cosine scheduler
4. ✅ pad_token 兜底设置
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

from utils.common import load_json, print_config, get_device

# Qwen2.5 LoRA 目标模块
QWEN_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def process_baseline_sample(tokenizer, example: Dict, max_length: int = 384) -> Dict:
    """
    处理 Baseline 样本（无 COT，直接问答）。
    使用 Qwen chat_template，输入部分 labels=-100。
    """
    messages = [
        {"role": "system", "content": example.get("instruction", "解答这道小学数学题，直接输出数字答案。")},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    all_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    prefix_messages = messages[:2]
    prefix_text = tokenizer.apply_chat_template(prefix_messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]

    prefix_len = len(prefix_ids)
    labels = [-100] * prefix_len + all_ids[prefix_len:]

    if len(all_ids) > max_length:
        all_ids = all_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": all_ids,
        "attention_mask": [1] * len(all_ids),
        "labels": labels,
    }


class BaselineTrainer:
    """Baseline 训练器（0.5B 专属优化）"""

    DEFAULT_CONFIG = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_cache_dir": "../models/Qwen2.5-0.5B-Instruct",
        "train_data_path": "../data/train.json",
        "output_dir": "../outputs/baseline",
        "max_length": 384,
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
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.device = None

    def setup(self, device: Optional[str] = None):
        """加载模型"""
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

    def prepare_data(self) -> List[Dict]:
        """预处理训练数据"""
        print("加载训练数据...")
        train_data = load_json(self.config["train_data_path"])
        print(f"训练样本数: {len(train_data)}")
        max_len = self.config["max_length"]
        return [process_baseline_sample(self.tokenizer, d, max_len) for d in train_data]

    def setup_lora(self):
        """配置 LoRA"""
        print("配置 LoRA（Qwen2.5-0.5B）...")
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

    def create_trainer(self, train_dataset: List[Dict]):
        """创建 Trainer"""
        output_dir = self.config["output_dir"]
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
            save_steps=self.config["save_steps"],
            learning_rate=self.config["learning_rate"],
            warmup_ratio=self.config["warmup_ratio"],
            weight_decay=self.config["weight_decay"],
            lr_scheduler_type=self.config["lr_scheduler_type"],
            save_on_each_node=True,
            gradient_checkpointing=True,
            report_to="none",
            remove_unused_columns=False,
            fp16=False,
            bf16=self.device != "cpu",
            dataloader_num_workers=2,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True),
        )

    def train(self, resume: bool = True):
        """训练"""
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

    def run(self, device: Optional[str] = None, resume: bool = True):
        """完整流程"""
        self.setup(device)
        train_dataset = self.prepare_data()
        self.setup_lora()
        self.create_trainer(train_dataset)
        self.train(resume)


def train_baseline(config: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
    """便捷函数"""
    trainer = BaselineTrainer(config)
    trainer.run(device=device)


if __name__ == "__main__":
    train_baseline()
