"""
方案2：COT数据生成 + 数据增强

⚠️ 已废弃：完整COT数据已就绪（train_preference_final_merged.json，11955条）
  - chosen 字段即正确COT，无需再调API生成
  - pipeline_config 会自动从偏好数据派生SFT数据
  - 本文件仅保留数字替换增强功能供参考

数据逻辑（已简化）：
  主数据源：train_preference_final_merged.json
    → SFT: chosen 字段作为 cot（pipeline_config 自动派生 train_cot.json）
    → DPO: chosen/rejected 直接使用
    → GRPO: question/answer 直接使用

如需数字替换增强（可选），可调用 augment_data_with_number_replacement()
"""

import os
import sys
import re
import random
import time
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common import load_json, save_json
from utils.longcat_api import generate_correct_cot, generate_wrong_cot


# ============================================================
# 阶段1：API生成原始COT（正确 + 错误）
# ============================================================

def generate_cot_for_original_data(
    train_data: List[Dict],
    output_cot_path: str,
    output_preference_path: str,
    max_samples: int = None,
    resume: bool = True,
    api_interval: float = 0.3,
) -> Tuple[List[Dict], List[Dict]]:
    """
    对原始数据调用LongCat API生成COT。
    同时产出SFT数据和DPO偏好数据。
    支持断点续生成（已生成的跳过）。
    """
    existing_cot = {}
    existing_pref = {}
    if resume and os.path.exists(output_cot_path):
        for item in load_json(output_cot_path):
            existing_cot[str(item["id"])] = item
    if resume and os.path.exists(output_preference_path):
        for item in load_json(output_preference_path):
            existing_pref[str(item["id"])] = item

    data_to_process = train_data[:max_samples] if max_samples else train_data
    total = len(data_to_process)
    cot_results = list(existing_cot.values())
    pref_results = list(existing_pref.values())
    remaining = [d for d in data_to_process if str(d.get("id", "")) not in existing_cot]

    print(f"总样本: {total}, 已有COT: {len(existing_cot)}, 待生成: {len(remaining)}")

    for idx, item in enumerate(remaining):
        qid = str(item.get("id", idx))
        question = item["question"]
        answer = str(item["answer"])

        print(f"[{idx+1}/{len(remaining)}] ID={qid} 生成COT...")

        try:
            correct_cot = generate_correct_cot(question, answer)
            time.sleep(api_interval)
            wrong_cot = generate_wrong_cot(question, answer)
            time.sleep(api_interval)

            cot_item = {
                "id": qid,
                "question": question,
                "answer": answer,
                "cot": correct_cot,
                "instruction": item.get("instruction", ""),
                "augmented": False,
            }
            cot_results.append(cot_item)

            pref_item = {
                "id": qid,
                "question": question,
                "answer": answer,
                "chosen": correct_cot,
                "rejected": wrong_cot,
            }
            pref_results.append(pref_item)

            if (idx + 1) % 50 == 0:
                save_json(cot_results, output_cot_path)
                save_json(pref_results, output_preference_path)
                print(f"  checkpoint已保存 ({idx+1}条)")

        except Exception as e:
            print(f"  ✗ 失败: {e}，跳过")
            continue

    save_json(cot_results, output_cot_path)
    save_json(pref_results, output_preference_path)
    print(f"✓ 原始COT生成完成: {len(cot_results)}条")
    return cot_results, pref_results


# ============================================================
# 阶段2：数据增强（数字替换 + 代码自动算答案，零API调用）
# ============================================================

def _extract_all_numbers(text: str) -> List[Tuple[str, int]]:
    """提取文本中所有数字，返回[(数字字符串, 数字值), ...]"""
    return [(m.group(), int(m.group())) for m in re.finditer(r'\d+', text)]


def _compute_answer_from_cot(cot_text: str) -> Optional[int]:
    """
    从COT文本中提取算式并计算结果。
    匹配格式如 "4×55=220" 或 "220-135=85"，返回最后一个计算结果。
    """
    # 匹配 "A±B=C" 或 "A×B=C" 格式
    patterns = [
        r'(\d+)\s*[×x\*]\s*(\d+)\s*=\s*(\d+)',      # 乘法
        r'(\d+)\s*[+\-]\s*(\d+)\s*=\s*(\d+)',     # 加减
        r'(\d+)\s*/\s*(\d+)\s*=\s*(\d+)',          # 除法
        r'答案[：:]\s*(\d+)',                       # 答案行
    ]

    results = []
    for p in patterns:
        for m in re.finditer(p, cot_text):
            try:
                groups = m.groups()
                if len(groups) == 3 and groups[2]:
                    results.append(int(groups[2]))
            except (ValueError, TypeError):
                pass

    if results:
        return results[-1]  # 返回最后一个计算结果

    # 如果没有找到算式，尝试直接匹配答案
    ans_match = re.search(r'答案[：:]\s*(\d+)', cot_text)
    if ans_match:
        return int(ans_match.group(1))

    return None


def _replace_numbers(text: str, num_map: Dict[int, int]) -> str:
    """将text中的数字按num_map替换"""
    for orig, new in num_map.items():
        text = text.replace(str(orig), str(new), 1)
    return text


def augment_data_with_number_replacement(
    cot_data: List[Dict],
    augment_per_sample: int = 2,
    output_path: str = None,
) -> List[Dict]:
    """
    数字替换数据增强（零API调用）：
    1. 复制原始COT
    2. 替换题目/步骤中的数字
    3. 从修改后的COT中提取算式，自动计算新答案
    4. 不改变解题逻辑结构
    """
    augmented = []

    for item in cot_data:
        if item.get("augmented", False):
            continue  # 跳过已增强的数据

        original_q = item["question"]
        original_cot = item["cot"]
        original_answer = item["answer"]

        # 提取题目中的数字
        nums_in_q = _extract_all_numbers(original_q)
        if len(nums_in_q) < 2:
            continue  # 数字太少，跳过

        for _ in range(augment_per_sample):
            # 随机选择1-2个数字进行替换
            k = min(len(nums_in_q), 2)
            selected = random.sample(nums_in_q, k)
            num_map = {}

            for num_str, num_val in selected:
                # 生成新数字（原数字的0.5~2.0倍）
                factor = random.uniform(0.5, 2.0)
                new_val = max(1, int(num_val * factor))
                # 避免相同
                if new_val == num_val:
                    new_val = max(1, new_val + random.choice([-1, 1]))
                num_map[num_val] = new_val

            # 替换题目中的数字
            new_q = _replace_numbers(original_q, num_map)

            # 替换COT中的数字
            new_cot = _replace_numbers(original_cot, num_map)

            # 从修改后的COT中计算新答案
            new_answer = _compute_answer_from_cot(new_cot)
            if new_answer is None:
                # 回退：用简单比例估算
                try:
                    orig_ans = int(original_answer)
                    if num_map:
                        last_ratio = list(num_map.values())[-1] / list(num_map.keys())[-1]
                        new_answer = max(0, int(orig_ans * last_ratio))
                    else:
                        new_answer = orig_ans
                except (ValueError, IndexError):
                    continue

            augmented.append({
                "id": f"{item['id']}_aug{_}",
                "question": new_q,
                "answer": str(new_answer),
                "cot": new_cot,
                "instruction": item.get("instruction", ""),
                "augmented": True,
                "original_id": item["id"],
            })

    total = len(cot_data) + len(augmented)
    print(f"数据增强: {len(cot_data)} → {total} 条 (+{len(augmented)})")

    # 合并原始 + 增强
    combined = list(cot_data) + augmented

    if output_path:
        save_json(combined, output_path)
        print(f"已保存: {output_path}")

    return combined


# ============================================================
# 完整流水线
# ============================================================

def run_data_pipeline(
    train_data_path: str,
    output_cot_path: str,
    output_preference_path: str,
    output_final_path: str,
    augment_per_sample: int = 2,
    max_api_samples: int = None,
    resume: bool = True,
):
    """
    完整数据构建流水线：

    Step 1: API生成原始COT（正确+错误）
    Step 2: 数字替换增强（代码自动算答案，零API）
    Step 3: 输出最终训练数据
    """
    print("=" * 60)
    print("方案2 数据构建流水线")
    print("=" * 60)

    # Step 1: 加载原始数据
    print("\n[Step 1] 加载原始数据...")
    train_data = load_json(train_data_path)
    print(f"  原始数据: {len(train_data)} 条")

    # Step 2: API生成COT
    print(f"\n[Step 2] 调用LongCat API生成COT...")
    print(f"  ⚠ API调用量: {max_api_samples or len(train_data)} 条")
    print(f"  💡 增强数据将跳过API，使用代码自动处理")
    cot_data, pref_data = generate_cot_for_original_data(
        train_data,
        output_cot_path=output_cot_path,
        output_preference_path=output_preference_path,
        max_samples=max_api_samples,
        resume=resume,
    )
    print(f"  ✓ 原始COT数据: {len(cot_data)} 条")

    # Step 3: 数字替换增强（零API）
    print(f"\n[Step 3] 数字替换增强（零API调用）...")
    print(f"  每条原始数据生成 {augment_per_sample} 条增强")
    final_data = augment_data_with_number_replacement(
        cot_data,
        augment_per_sample=augment_per_sample,
        output_path=output_final_path,
    )

    print("\n" + "=" * 60)
    print("数据构建完成！")
    print(f"  SFT最终数据: {output_final_path} ({len(final_data)}条)")
    print(f"  DPO偏好数据: {output_preference_path} ({len(pref_data)}条)")
    print("=" * 60)
    return final_data, pref_data


if __name__ == "__main__":
    run_data_pipeline(
        train_data_path="./data/train.json",
        output_cot_path="./data/train_cot_original.json",
        output_preference_path="./data/train_preference.json",
        output_final_path="./data/train_cot.json",
        augment_per_sample=2,
    )
