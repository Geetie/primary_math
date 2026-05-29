"""
训练流水线统一配置模块
- GPU/CPU 配置分离
- 路径自动适配（Kaggle / ModelScope / 本地）
- Notebook 只需导入配置，无需手写参数
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import get_device as _get_device


def get_device() -> str:
    return _get_device()


def get_paths() -> dict:
    """
    返回所有路径配置，自动适配运行环境。

    Returns:
        {
            'base_model': str,
            'data_dir': str,
            'output_dir': str,
            'train_cot': str,
            'train_preference': str,
            'train_cot_original': str,
        }
    """
    if os.path.exists('/kaggle'):
        root = '/kaggle/working'
    elif os.path.exists('/mnt/workspace'):
        root = '/mnt/workspace'
    else:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_dir = os.path.join(root, 'data')
    output_dir = os.path.join(root, 'outputs')
    model_dir = os.path.join(os.path.dirname(root), 'models', 'qwen', 'Qwen2___5-0___5B-Instruct')

    if not os.path.exists(model_dir):
        alt = os.path.join(root, 'models', 'qwen', 'Qwen2___5-0___5B-Instruct')
        if os.path.exists(alt):
            model_dir = alt

    train_cot = os.path.join(data_dir, 'train_cot.json')
    if not os.path.exists(train_cot):
        train_cot = os.path.join(data_dir, 'train_cot_original.json')

    return {
        'base_model': model_dir,
        'data_dir': data_dir,
        'output_dir': output_dir,
        'train_cot': train_cot,
        'train_preference': os.path.join(data_dir, 'train_preference.json'),
        'train_cot_original': os.path.join(data_dir, 'train_cot_original.json'),
    }


def get_sft_config(device: str = None, paths: dict = None) -> dict:
    """
    返回方案2 SFT训练配置，GPU/CPU自动分离。

    Args:
        device: 设备类型，None则自动检测
        paths: 路径配置，None则自动获取
    """
    if device is None:
        device = get_device()
    if paths is None:
        paths = get_paths()

    is_gpu = device != "cpu"

    config = {
        'model_cache_dir': paths['base_model'],
        'train_data_path': paths['train_cot'],
        'output_dir': os.path.join(paths['output_dir'], 'scheme2_cot'),
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'learning_rate': 2e-4,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'lr_scheduler_type': 'cosine',
        'save_steps': 500,
        'logging_steps': 10,
    }

    if is_gpu:
        config.update({
            'max_length': 512,
            'batch_size': 8,
            'gradient_accumulation_steps': 2,
            'num_epochs': 3,
        })
    else:
        config.update({
            'max_length': 256,
            'batch_size': 1,
            'gradient_accumulation_steps': 16,
            'num_epochs': 1,
            'max_steps': 2,
        })

    return config


def get_dpo_config(device: str = None, paths: dict = None) -> dict:
    """
    返回方案3 DPO训练配置，GPU/CPU自动分离。

    Returns:
        train_dpo() 的关键字参数字典
    """
    if device is None:
        device = get_device()
    if paths is None:
        paths = get_paths()

    is_gpu = device != "cpu"
    sft_peft_path = os.path.join(paths['output_dir'], 'scheme2_cot', 'final')

    config = {
        'sft_model_path': paths['base_model'],
        'sft_peft_path': sft_peft_path if os.path.exists(sft_peft_path) else None,
        'pref_data_path': paths['train_preference'],
        'output_dir': os.path.join(paths['output_dir'], 'scheme3_dpo'),
        'device': device,
        'learning_rate': 5e-7,
        'beta': 0.1,
        'max_length': 512 if is_gpu else 256,
    }

    if is_gpu:
        config.update({
            'batch_size': 2,
            'gradient_accumulation_steps': 4,
            'num_epochs': 3,
        })
    else:
        config.update({
            'batch_size': 1,
            'gradient_accumulation_steps': 8,
            'num_epochs': 1,
            'max_steps': 2,
        })

    return config


def get_grpo_config(device: str = None, paths: dict = None) -> dict:
    """
    返回方案4 GRPO训练配置，GPU/CPU自动分离。

    Returns:
        train_grpo() 的关键字参数字典
    """
    if device is None:
        device = get_device()
    if paths is None:
        paths = get_paths()

    is_gpu = device != "cpu"
    dpo_model_path = os.path.join(paths['output_dir'], 'scheme3_dpo', 'final')
    sft_peft_path = os.path.join(paths['output_dir'], 'scheme2_cot', 'final')

    if os.path.exists(dpo_model_path):
        peft_path = dpo_model_path
    elif os.path.exists(sft_peft_path):
        peft_path = sft_peft_path
    else:
        peft_path = None

    config = {
        'sft_model_path': paths['base_model'],
        'sft_peft_path': peft_path,
        'grpo_data_path': paths['train_cot'],
        'output_dir': os.path.join(paths['output_dir'], 'scheme4_grpo'),
        'device': device,
        'learning_rate': 1e-6,
    }

    if is_gpu:
        config.update({
            'group_size': 4,
            'num_iterations': 50,
        })
    else:
        config.update({
            'group_size': 2,
            'num_iterations': 2,
            'max_new_tokens': 64,
            'max_samples': 3,
        })

    return config
