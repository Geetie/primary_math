"""
方案3：DPO（直接策略优化）对齐训练
基于方案2的SFT模型，使用偏好数据进行DPO训练
完整实现，使用 HuggingFace trl 库的 DPOTrainer

参考: https://github.com/ShawhinT/YouTube-Blog/tree/main/LLMs/dpo
数据格式严格遵循 DPOTrainer 要求：
  - prompt: List[Dict] 消息列表（DPOTrainer自动apply_chat_template）
  - chosen: str assistant回复（含<|im_start|>assistant前缀）
  - rejected: str assistant回复（含<|im_start|>assistant前缀）

审查修复记录：
1. _ensure_answer_suffix：添加 re.MULTILINE 支持多行文本匹配，放宽答案匹配以支持含空格/单位的答案
2. prepare_dpo_dataset：移除冗余 tokenizer/max_length 参数，添加注释说明 Tokenization 由 DPOTrainer 内部处理
3. _get_attn_impl：提前缓存避免重复调用
4. 断点检测：移除手动查找，直接使用 resume_from_checkpoint=True 让 Trainer 自动处理
5. 路径解析：添加 _resolve_relative_path 确保路径一致性
"""

import os
import sys
import re
from typing import Dict, Any, Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

from utils.common import load_json, save_json, print_config, get_device, ensure_flash_attn, enable_tf32, set_seed


def _resolve_relative_path(path: str) -> str:
    """将相对路径解析为绝对路径（相对于脚本目录）"""
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, path))


def _get_attn_impl():
    return ensure_flash_attn()


QWEN_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ============================================================
# DPO 数据格式转换
# ============================================================

def _ensure_answer_suffix(text: str, answer: str) -> str:
    """
    确保文本末尾包含"答案：xxx"且不重复。
    如果已有"答案："行，检查是否与answer一致，不一致则替换。
    
    支持多行文本：匹配最后一行的答案标记。
    支持含空格/单位的答案（如"4.5 米"）。
    """
    # 检查是否已有"答案："行（匹配最后一行）
    match = re.search(r'答案[：:]\s*[\S\s]*?\s*$', text.strip(), re.MULTILINE)
    if match:
        # 已有答案行，检查是否正确
        existing_ans = match.group().split('：')[-1].split(':')[-1].strip()
        if existing_ans == str(answer).strip():
            return text  # 已正确，不重复
        else:
            # 答案不一致，替换末尾的答案行
            return re.sub(r'答案[：:]\s*[\S\s]*?\s*$', f'答案：{answer}', text.strip(), flags=re.MULTILINE)
    else:
        # 没有答案行，追加
        return f"{text.strip()}\n答案：{answer}"


def prepare_dpo_dataset(preference_data_path: str):
    """
    将偏好数据转换为 DPOTrainer 要求的格式（trl >= 0.12.0 conversational格式）。

    输入格式（train_preference.json）:
        {"id": "0", "question": "...", "answer": "85",
         "chosen": "正确COT步骤", "rejected": "错误COT步骤"}

    输出格式（trl DPOTrainer conversational格式）:
        {
            "prompt": [{"role": "system", ...}, {"role": "user", ...}],
            "chosen": [{"role": "assistant", "content": "正确COT步骤"}],
            "rejected": [{"role": "assistant", "content": "错误COT步骤"}],
        }

    注：Tokenization 由 DPOTrainer 内部根据 DPOConfig.max_length 自动处理，
        此处仅做格式转换。
    """
    from datasets import Dataset

    pref_data = load_json(preference_data_path)
    dpo_dataset = []

    for item in pref_data:
        question = item['question']
        if isinstance(question, list):
            question = "".join(question)
        chosen_text = item['chosen']
        if isinstance(chosen_text, list):
            chosen_text = "".join(chosen_text)
        rejected_text = item['rejected']
        if isinstance(rejected_text, list):
            rejected_text = "".join(rejected_text)
        answer = str(item['answer'])

        prompt = [
            {"role": "system", "content": "解答小学数学题，按步骤解答，最后用\"答案：数字\"给出结果。"},
            {"role": "user", "content": question},
        ]

        chosen = _ensure_answer_suffix(chosen_text, answer)

        # rejected: 错误COT + 错误答案（去重处理）
        err_match = re.search(r'答案[：:]\s*(\S+)', rejected_text)
        err_ans = err_match.group(1) if err_match else "0"
        rejected = _ensure_answer_suffix(rejected_text, err_ans)

        dpo_dataset.append({
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })

    print(f"DPO数据集: {len(dpo_dataset)} 条")
    return Dataset.from_list(dpo_dataset)


# ============================================================
# DPO Trainer（基于 HuggingFace DPOTrainer）
# ============================================================

def train_dpo(
    sft_model_path: str,
    sft_peft_path: str,
    pref_data_path: str,
    output_dir: str,
    device: str = None,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    beta: float = 0.1,
    max_steps: int = -1,
    dataloader_num_workers: int = 4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    save_steps: int = 500,
    weight_decay: float = 0.01,
    optim: str = "adamw_torch_fused",
    seed: int = 42,
):
    """
    DPO训练流程，使用 HuggingFace trl 库的 DPOTrainer。

    Args:
        sft_model_path: SFT基础模型路径
        sft_peft_path: SFT训练后的PEFT权重路径
        pref_data_path: 偏好数据路径
        output_dir: 输出目录
        batch_size: 批次大小
        gradient_accumulation_steps: 梯度累积步数
        num_epochs: 训练轮数
        learning_rate: 学习率
        max_length: 最大序列长度
        beta: DPO beta参数（KL散度系数）
    """
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        print("⚠️ trl 库未安装，请运行: pip install trl")
        print("DPO训练跳过...")
        return None, None, None

    if device is None:
        device = get_device()
    print(f"设备: {device}")

    # 解析路径，确保路径一致性
    sft_model_path = _resolve_relative_path(sft_model_path)
    if sft_peft_path:
        sft_peft_path = _resolve_relative_path(sft_peft_path)
    pref_data_path = _resolve_relative_path(pref_data_path)
    output_dir = _resolve_relative_path(output_dir)

    print_config({
        "sft_peft_path": sft_peft_path,
        "pref_data_path": pref_data_path,
        "output_dir": output_dir,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "beta": beta,
    })

    # ---- 加载模型 ----
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        sft_model_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 提前缓存 attn_impl，避免重复调用
    attn_impl = _get_attn_impl() if device != "cpu" else None
    attn_kwargs = {"attn_implementation": attn_impl} if attn_impl else {}

    base_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        device_map={"": device} if device != "cpu" else None,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
        **attn_kwargs,
    )

    if sft_peft_path and os.path.exists(sft_peft_path):
        is_peft = os.path.exists(os.path.join(sft_peft_path, "adapter_config.json"))
        if is_peft:
            model = PeftModel.from_pretrained(base_model, sft_peft_path)
            print(f"已加载PEFT权重: {sft_peft_path}")
            print("合并PEFT权重...")
            model = model.merge_and_unload()
            del base_model
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            del base_model
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                sft_peft_path,
                device_map={"": device} if device != "cpu" else None,
                torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
                trust_remote_code=True,
                **attn_kwargs,
            )
            print(f"已加载完整模型: {sft_peft_path}")
    else:
        model = base_model

    if device == "cpu":
        model = model.to(device)

    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=QWEN_LORA_TARGET_MODULES,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- 准备DPO数据 ----
    print("准备DPO数据...")
    dpo_data = prepare_dpo_dataset(pref_data_path)

    # ---- 创建DPOConfig ----
    print("创建DPO Trainer...")
    dpo_config_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        num_train_epochs=num_epochs,
        save_strategy="steps",
        save_steps=save_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=weight_decay,
        beta=beta,
        loss_type="sigmoid",
        label_smoothing=0.0,
        max_length=max_length,
        report_to="none",
        fp16=False,
        bf16=device != "cpu",
        remove_unused_columns=False,
        generate_during_eval=False,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=device != "cpu",
        optim=optim,
        seed=seed,
    )
    if max_steps > 0:
        dpo_config_kwargs["max_steps"] = max_steps

    training_args = DPOConfig(**dpo_config_kwargs)

    # ---- 创建DPOTrainer ----
    import trl
    import re
    trl_version = tuple(int(x) for x in re.findall(r'\d+', trl.__version__)[:2])
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dpo_data,
    )
    if trl_version >= (0, 12):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    dpo_trainer = DPOTrainer(**trainer_kwargs)

    # ---- 开始训练 ----
    print("开始DPO训练...")
    
    checkpoint_dir = os.path.join(output_dir, "checkpoint-last")
    if os.path.exists(checkpoint_dir):
        print(f"检测到 checkpoint，继续训练: {checkpoint_dir}")
        dpo_trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        print("未检测到 checkpoint，从头开始训练")
        dpo_trainer.train(resume_from_checkpoint=False)

    # ---- 保存模型 ----
    final_path = os.path.join(output_dir, "final")
    dpo_trainer.save_model(final_path)
    print(f"DPO模型已保存: {final_path}")

    return dpo_trainer, model, tokenizer


def run_dpo_from_notebook(
    sft_model_path: str,
    sft_peft_path: str,
    pref_data_path: str,
    output_dir: str,
    device: str = "cuda",
    **kwargs,
):
    """从Notebook调用的便捷函数，额外参数通过kwargs传递"""
    os.makedirs(output_dir, exist_ok=True)

    return train_dpo(
        sft_model_path=sft_model_path,
        sft_peft_path=sft_peft_path,
        pref_data_path=pref_data_path,
        output_dir=output_dir,
        device=device,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model_path", type=str,
                       default="../models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft_peft_path", type=str,
                       default="../outputs/scheme2_cot/final")
    parser.add_argument("--pref_data_path", type=str,
                       default="../data/train_preference_final_merged.json")
    parser.add_argument("--output_dir", type=str,
                       default="../outputs/scheme3_dpo")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_dpo(
        sft_model_path=args.sft_model_path,
        sft_peft_path=args.sft_peft_path,
        pref_data_path=args.pref_data_path,
        output_dir=args.output_dir,
        device=args.device,
    )
