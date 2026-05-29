"""
多模型 API 调用模块（支持 LongCat / DeepSeek / Gemini）
- 正确COT：DeepSeek R1（数学推理最强，价格极低）
- 错误COT：LongCat-Flash-Lite（速度快，生成错误步骤不需要深度推理）
兼容 OpenAI SDK 格式
"""

import os
import time
from typing import Optional


# ============================================================
# API 配置（支持多提供商）
# ============================================================

# DeepSeek R1 配置（推荐，数学推理专精）
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL_R1 = "deepseek-reasoner"  # DeepSeek R1 推理模型

# LongCat API 配置（备用）
LONGCAT_BASE_URL = "https://api.longcat.chat/openai/v1"
LONGCAT_MODEL_THINKING = "LongCat-Flash-Thinking-2601"
LONGCAT_MODEL_LITE = "LongCat-Flash-Lite"

# Gemini 配置（备用）
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL = "gemini-2.5-pro-preview-06-05"

# 当前使用的提供商（可切换）
CURRENT_PROVIDER = "deepseek"  # 可选: "deepseek", "longcat", "gemini"


def _get_api_config() -> tuple:
    """根据当前提供商返回 API 配置"""
    if CURRENT_PROVIDER == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError("环境变量 DEEPSEEK_API_KEY 未设置")
        return DEEPSEEK_BASE_URL, key, DEEPSEEK_MODEL_R1
    
    elif CURRENT_PROVIDER == "longcat":
        key = os.environ.get("LONGCAT_API_KEY", "")
        if not key:
            raise ValueError("环境变量 LONGCAT_API_KEY 未设置")
        return LONGCAT_BASE_URL, key, LONGCAT_MODEL_THINKING
    
    elif CURRENT_PROVIDER == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("环境变量 GEMINI_API_KEY 未设置")
        return GEMINI_BASE_URL, key, GEMINI_MODEL
    
    else:
        raise ValueError(f"不支持的提供商: {CURRENT_PROVIDER}")


def call_api(
    prompt: str,
    system: str = "",
    model: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    retries: int = 3,
    timeout: int = 120,
) -> str:
    """
    调用 API（非流式），自动选择当前提供商。

    Args:
        prompt: 用户消息
        system: 系统提示（可选）
        model: 模型名称（默认使用当前提供商的默认模型）
        max_tokens: 最大输出 token
        temperature: 采样温度
        retries: 失败重试次数
        timeout: 请求超时（秒）
    Returns:
        模型响应文本
    """
    import requests

    base_url, api_key, default_model = _get_api_config()
    if model is None:
        model = default_model

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content.strip()
        except Exception as e:
            print(f"  API调用失败 [{model}] (第{attempt}次): {e}")
            if attempt < retries:
                time.sleep(2 * attempt)
            else:
                raise


# 兼容旧接口
def call_longcat(*args, **kwargs):
    """兼容旧接口，转发到 call_api"""
    return call_api(*args, **kwargs)


# ============================================================
# COT 生成专用函数
# ============================================================

SYSTEM_PROMPT_CORRECT = (
    "你是一位小学数学老师，请为下面的数学应用题写出详细的解题步骤。\n"
    "要求：\n"
    "1. 用简洁的中文逐步列出计算过程\n"
    "2. 每一步包含算式和结果\n"
    "3. 最后一行必须是\"答案：数字\"（仅数字，不带单位）\n"
    "4. 不要输出多余内容"
)

SYSTEM_PROMPT_WRONG = (
    "你是一位经常算错的小学生，请为下面的数学应用题写出一个包含错误的解题过程。\n"
    "要求：\n"
    "1. 看起来像在认真解题，但中间至少包含一个计算错误或理解错误\n"
    "2. 用简洁的中文逐步列出计算过程\n"
    "3. 最后一行必须是\"答案：数字\"（这个数字因为前面的错误所以是错的）\n"
    "4. 不要输出多余内容"
)


def generate_correct_cot(question: str, answer: str) -> str:
    """
    生成正确的 COT 解题步骤。
    使用 DeepSeek R1（推理专精）或当前配置的模型。
    """
    prompt = f"题目：{question}\n正确答案：{answer}\n请写出解题步骤。"
    return call_api(
        prompt,
        system=SYSTEM_PROMPT_CORRECT,
        max_tokens=1024,
        temperature=0.3,
        timeout=120,
    )


def generate_wrong_cot(question: str, correct_answer: str) -> str:
    """
    生成错误的 COT 解题步骤（用于 DPO 偏好数据）。
    使用 LongCat-Flash-Lite（速度快，错误步骤不需要深度推理）。
    """
    prompt = (
        f"题目：{question}\n"
        f"注意：正确答案不是{correct_answer}，请写出一个包含计算错误的解题过程，"
        f"得出一个不同于{correct_answer}的错误答案。"
    )
    
    # 错误COT用 LongCat Lite（需要单独配置）
    key = os.environ.get("LONGCAT_API_KEY", "")
    if key:
        import requests
        url = f"{LONGCAT_BASE_URL}/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_WRONG},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": LONGCAT_MODEL_LITE,
            "messages": messages,
            "stream": False,
            "max_tokens": 512,
            "temperature": 0.7,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    
    # 如果没有 LongCat key，用当前提供商
    return call_api(
        prompt,
        system=SYSTEM_PROMPT_WRONG,
        max_tokens=512,
        temperature=0.7,
        timeout=60,
    )
