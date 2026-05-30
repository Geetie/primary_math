"""
模型加载工具模块
支持自动下载和缓存模型
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import check_model_downloaded


def download_model_qwen(cache_dir: str = "./models/Qwen2.5-0.5B-Instruct") -> str:
    """
    下载Qwen2.5-0.5B-Instruct模型
    如果已下载则跳过
    """
    if check_model_downloaded(cache_dir):
        print(f"模型已存在于: {cache_dir}")
        return cache_dir
    
    print("正在下载Qwen2.5-0.5B-Instruct模型...")
    try:
        from modelscope import snapshot_download
        model_dir = snapshot_download(
            "Qwen/Qwen2.5-0.5B-Instruct",
            cache_dir=os.path.dirname(cache_dir),
            revision="master"
        )
        print(f"模型下载完成: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"模型下载失败: {e}")
        print("尝试使用transformers下载...")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            # 这会触发下载
            _ = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=os.path.dirname(cache_dir))
            _ = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=os.path.dirname(cache_dir))
            return cache_dir
        except Exception as e2:
            print(f"transformers下载也失败: {e2}")
            raise


def load_model_and_tokenizer(model_path: str, use_peft: bool = False, peft_path: str = None, device: str = None):
    """
    加载模型和tokenizer
    
    Args:
        model_path: 基础模型路径
        use_peft: 是否使用PEFT模型
        peft_path: PEFT模型路径
        device: 设备，None则自动选择
    
    Returns:
        model, tokenizer
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device if device != "cpu" else None,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    # 加载PEFT权重
    if use_peft and peft_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, peft_path)
        print(f"已加载PEFT权重: {peft_path}")
    
    return model, tokenizer


def predict(messages, model, tokenizer, max_new_tokens: int = 512, device: str = "cuda"):
    """
    使用模型进行预测
    
    Args:
        messages: 对话消息列表
        model: 模型
        tokenizer: tokenizer
        max_new_tokens: 最大生成token数
        device: 设备
    
    Returns:
        生成的文本
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 使用贪心解码，更稳定
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response.strip()


