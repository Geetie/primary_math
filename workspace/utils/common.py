"""
通用工具函数模块
包含所有方案共享的工具函数
"""

import json
import re
import os
from typing import List, Dict, Any


def load_json(path: str) -> List[Dict]:
    """加载JSON文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[Dict], path: str):
    """保存JSON文件"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(results: List[tuple], path: str):
    """保存CSV结果文件"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("id,ret\n")
        for id_val, ret in results:
            f.write(f"{id_val},{ret}\n")


def extract_number(text: str) -> str:
    """
    从文本中提取数字答案。
    优先匹配「答案：数字」格式，其次匹配最后一个数字。
    支持整数、小数、负数。
    """
    text = text.strip()
    if not text:
        return ""

    # 1. 优先匹配 "答案：xxx" / "答案:xxx" 格式
    ans_match = re.search(r'答案[：:]\s*(-?\d+\.?\d*)', text)
    if ans_match:
        return ans_match.group(1)

    # 2. 回退：取最后一个数字
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches:
        return matches[-1]

    return ""


def check_model_downloaded(model_path: str) -> bool:
    """检查模型是否已下载"""
    return os.path.exists(model_path) and os.path.isdir(model_path) and len(os.listdir(model_path)) > 0


def get_device() -> str:
    """获取可用的设备"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_config(config: Dict[str, Any]):
    """打印配置信息"""
    print("=" * 50)
    print("配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)
