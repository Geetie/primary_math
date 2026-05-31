"""
训练流水线统一配置模块
- GPU/CPU 配置分离
- 路径自动适配（Kaggle / ModelScope / 本地）
- Notebook 只需导入配置，无需手写参数

数据源：train_preference_final_merged.json（完整11955条COT偏好数据）
  - SFT: 从 chosen 字段提取COT
  - DPO: 直接使用 chosen/rejected
  - GRPO: 使用 question/answer
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .common import get_device as _get_device, enable_tf32 as _enable_tf32, set_seed as _set_seed


def get_device() -> str:
    return _get_device()


def ensure_model_downloaded(model_dir: str) -> str:
    if os.path.exists(model_dir):
        return model_dir

    print(f"模型目录不存在: {model_dir}")
    print("尝试自动下载 Qwen2.5-0.5B-Instruct ...")

    target_dir = os.path.dirname(model_dir)

    if os.path.exists('/mnt/workspace'):
        cache_dir = os.path.join('/mnt/workspace', 'models')
    elif os.path.exists('/kaggle'):
        cache_dir = '/kaggle/models'
    else:
        cache_dir = target_dir

    os.makedirs(cache_dir, exist_ok=True)
    print(f"模型下载目标目录: {cache_dir}")

    try:
        from modelscope import snapshot_download
        downloaded = snapshot_download(
            'Qwen/Qwen2.5-0.5B-Instruct',
            cache_dir=cache_dir,
        )
        model_name = 'Qwen2___5-0___5B-Instruct'
        expected_path = os.path.join(cache_dir, 'qwen', model_name)
        if os.path.exists(expected_path):
            print(f"模型已下载到: {expected_path}")
            return expected_path
        print(f"模型已下载到: {downloaded}")
        return downloaded
    except ImportError:
        pass
    except Exception as e:
        print(f"ModelScope 下载失败: {e}")

    try:
        from huggingface_hub import snapshot_download
        downloaded = snapshot_download(
            'Qwen/Qwen2.5-0.5B-Instruct',
            cache_dir=cache_dir,
        )
        print(f"模型已下载到: {downloaded}")
        return downloaded
    except ImportError:
        pass
    except Exception as e:
        print(f"HuggingFace 下载失败: {e}")

    print("自动下载失败，请手动下载模型到 models/ 目录")
    return model_dir


def _find_preference_data(data_dir: str) -> str:
    candidates = [
        os.path.join(data_dir, 'train_preference_final_merged.json'),
        os.path.join(data_dir, 'train_preference.json'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def _ensure_sft_data_from_preference(pref_path: str) -> str:
    """
    从 train_preference_final_merged.json 派生 SFT 训练数据。
    将 chosen 字段转为 cot 字段，保存为 train_cot.json。
    如果 train_cot.json 已存在且来源一致，则跳过。
    """
    sft_path = pref_path.replace('train_preference_final_merged.json', 'train_cot.json').replace('train_preference.json', 'train_cot.json')

    if os.path.exists(sft_path):
        with open(sft_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        if len(existing) > 0 and existing[0].get('cot'):
            return sft_path

    if not os.path.exists(pref_path):
        return sft_path

    print(f"从偏好数据派生SFT数据: {pref_path} → {sft_path}")
    with open(pref_path, 'r', encoding='utf-8') as f:
        pref_data = json.load(f)

    sft_data = []
    for item in pref_data:
        cot_text = item.get('chosen', '')
        if isinstance(cot_text, list):
            cot_text = "".join(cot_text)
        if not cot_text.strip():
            continue
        sft_data.append({
            "id": item.get("id", ""),
            "question": item["question"],
            "answer": str(item["answer"]),
            "cot": cot_text,
            "instruction": item.get("instruction", "解答这道小学数学题。"),
        })

    os.makedirs(os.path.dirname(sft_path) or '.', exist_ok=True)
    with open(sft_path, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    print(f"SFT数据已生成: {len(sft_data)} 条 → {sft_path}")

    return sft_path


def get_paths() -> dict:
    if os.path.exists('/kaggle'):
        root = '/kaggle/working'
    elif os.path.exists('/mnt/workspace'):
        root = '/mnt/workspace'
    else:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_dir = os.path.join(root, 'data')
    output_dir = os.path.join(root, 'outputs')

    model_dir = None
    candidates = [
        os.path.join(os.path.dirname(root), 'models', 'qwen', 'Qwen2___5-0___5B-Instruct'),
        os.path.join(root, 'models', 'qwen', 'Qwen2___5-0___5B-Instruct'),
        os.path.join(os.path.dirname(root), 'models', 'Qwen2.5-0.5B-Instruct'),
        os.path.join(root, 'models', 'Qwen2.5-0.5B-Instruct'),
    ]
    for c in candidates:
        if os.path.exists(c):
            model_dir = c
            break

    if model_dir is None:
        model_dir = candidates[0]

    train_preference = _find_preference_data(data_dir)
    train_cot = _ensure_sft_data_from_preference(train_preference)

    model_dir = ensure_model_downloaded(model_dir)

    return {
        'base_model': model_dir,
        'data_dir': data_dir,
        'output_dir': output_dir,
        'train_cot': train_cot,
        'train_preference': train_preference,
        'train_cot_original': os.path.join(data_dir, 'train_cot_original.json'),
    }


def get_sft_config(device: str = None, paths: dict = None) -> dict:
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
        'save_steps': 2000,
        'logging_steps': 50,
        'seed': 42,
    }

    if is_gpu:
        _enable_tf32()
        _set_seed(42)
        config.update({
            'max_length': 320,
            'batch_size': 8,
            'gradient_accumulation_steps': 8,
            'num_epochs': 3,
            'dataloader_num_workers': 2,
            'optim': 'adamw_torch_fused',
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
        'learning_rate': 3e-5,
        'beta': 0.3,
        'weight_decay': 0.01,
        'max_length': 384 if is_gpu else 256,
        'dataloader_num_workers': 2 if is_gpu else 0,
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'save_steps': 1000,
        'seed': 42,
    }

    if is_gpu:
        config.update({
            'batch_size': 4,
            'gradient_accumulation_steps': 4,
            'num_epochs': 2,
            'optim': 'adamw_torch_fused',
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
        'learning_rate': 5e-6,
        'weight_decay': 0.01,
        'max_length': 512 if is_gpu else 256,
        'max_steps_per_epoch': 500 if is_gpu else 10,
        'gradient_accumulation_steps': 2,
        'save_steps': 50,
        'dataloader_num_workers': 4 if is_gpu else 0,
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'seed': 42,
    }

    if is_gpu:
        config.update({
            'group_size': 8,
            'num_iterations': 3,
            'max_new_tokens': 256,
            'optim': 'adamw_torch_fused',
        })
    else:
        config.update({
            'group_size': 2,
            'num_iterations': 1,
            'max_new_tokens': 64,
        })

    return config
