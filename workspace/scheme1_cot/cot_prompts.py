"""
COT（Chain of Thought）提示模板
针对 Qwen2.5-0.5B 小模型优化：
- 精简Prompt，避免上下文溢出（0.5B有效上下文仅2k）
- 强制固定输出格式：步骤：xxx 答案：数字
- Few-shot仅1个示例
"""

# ===== 标准提示（无COT，直接输出） =====
STANDARD_PROMPT = "解答这道小学数学题，直接输出数字答案，不要单位。"

# ===== Zero-shot COT（精简版） =====
ZERO_SHOT_COT_SIMPLE = "请一步步思考这道小学数学题，最后用\"答案：数字\"给出结果。"

ZERO_SHOT_COT = (
    "解答小学数学题，按以下格式输出：\n"
    "步骤：你的计算过程\n"
    "答案：数字（仅数字，不带单位）"
)

# ===== Few-shot COT（仅1个示例，适配0.5B小上下文） =====
FEW_SHOT_COT_EXAMPLES = (
    "解答小学数学题。\n"
    "例：商店有4框苹果每框55千克，卖出135千克，还剩多少？\n"
    "步骤：4×55=220，220-135=85\n"
    "答案：85\n"
    "请按同样格式解答："
)

# 提示类型映射
PROMPT_MAP = {
    "standard": STANDARD_PROMPT,
    "zero_shot_simple": ZERO_SHOT_COT_SIMPLE,
    "zero_shot": ZERO_SHOT_COT,
    "few_shot": FEW_SHOT_COT_EXAMPLES,
    "grpo": ZERO_SHOT_COT,
}


def get_cot_prompt(prompt_type: str = "zero_shot") -> str:
    """获取COT提示"""
    return PROMPT_MAP.get(prompt_type, ZERO_SHOT_COT)


def create_messages_with_cot(question, prompt_type: str = "zero_shot") -> list:
    """
    创建带COT的消息列表

    Args:
        question: 问题文本（str 或 list）
        prompt_type: 提示类型
    Returns:
        messages列表
    """
    # 处理 question 可能是 list 的情况
    if isinstance(question, list):
        question = "".join(question) if question else ""
    
    return [
        {"role": "system", "content": get_cot_prompt(prompt_type)},
        {"role": "user", "content": question},
    ]
