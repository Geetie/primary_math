"""
方案4：GRPO（组相对策略优化）训练
参考DeepSeek-R1论文，专为数学推理设计

核心算法：
1. 对每个问题采样 group_size 个回答（多路径推理）
2. 用规则奖励函数给每个回答打分（答案正确性 + 格式奖励）
3. 计算组内相对优势（归一化，零均值）
4. 用 PPO-style 裁剪损失 + KL散度惩罚 更新策略

关键实现细节：
- 采样时记录 old_log_probs（旧策略的log概率）
- 训练时计算当前策略的 log_probs
- 重要性采样比 ratio = exp(log_prob_current - log_prob_old)
- PPO裁剪 + KL惩罚
"""

import os
import sys
import re
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils.common import load_json, print_config, get_device


# ============================================================
# GRPO 配置
# ============================================================

class GRPOConfig:
    """GRPO训练配置"""
    group_size: int = 4           # 每组采样数
    num_iterations: int = 100     # 迭代次数
    max_length: int = 512         # 最大序列长度
    max_new_tokens: int = 128     # 最大生成token数

    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # GRPO特有参数
    epsilon: float = 0.2          # PPO裁剪参数
    kl_coef: float = 0.04         # KL散度惩罚系数
    reward_correct: float = 1.0   # 答案正确奖励
    reward_format: float = 0.1    # 格式奖励（有步骤+有答案）


# ============================================================
# 奖励函数
# ============================================================

def extract_number(text: str) -> str:
    """从文本中提取数字答案"""
    text = text.strip()
    if not text:
        return ""
    ans_match = re.search(r'答案[：:]\s*(-?\d+\.?\d*)', text)
    if ans_match:
        return ans_match.group(1)
    matches = re.findall(r'-?\d+\.?\d*', text)
    return matches[-1] if matches else ""


def compute_reward(generated_text: str, ground_truth: str) -> float:
    """
    计算总奖励 = 答案正确性奖励 + 格式奖励

    答案正确性：完全正确=1.0，错误=0.0
    格式奖励：有"步骤"关键词=0.05，有"答案："标记=0.05，最多0.1
    """
    reward = 0.0

    # 1. 答案正确性奖励（核心）
    gen_num = extract_number(generated_text)
    truth_num = extract_number(ground_truth)
    if gen_num and truth_num:
        try:
            gen_val = float(gen_num) if '/' not in gen_num else eval(gen_num)
            truth_val = float(truth_num) if '/' not in truth_num else eval(truth_num)
            if abs(gen_val - truth_val) < 0.01:
                reward += 1.0
        except (ValueError, ZeroDivisionError):
            pass

    # 2. 格式奖励（鼓励结构化输出）
    if re.search(r'步骤|先|再|然后|因此|所以', generated_text):
        reward += 0.05
    if re.search(r'答案[：:]', generated_text):
        reward += 0.05

    return reward


# ============================================================
# 组内相对优势计算
# ============================================================

def compute_group_relative_advantages(rewards: List[float]) -> List[float]:
    """
    计算组内相对优势（归一化，零均值）。

    GRPO核心：不关注绝对分数，只关注组内相对优劣。
    正确样本优势为正，错误样本优势为负。
    """
    n = len(rewards)
    if n <= 1:
        return [0.0] * n

    mean_reward = sum(rewards) / n
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / n) ** 0.5

    if std_reward < 1e-6:
        return [0.0] * n

    # 归一化优势
    advantages = [(r - mean_reward) / std_reward for r in rewards]
    return advantages


# ============================================================
# Log Probability 计算
# ============================================================

def compute_response_log_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    response: str,
    max_length: int = 512,
    no_grad: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    计算 response 部分的 log probability。

    Args:
        no_grad: True=旧策略采样（不需要梯度），False=当前策略训练（需要梯度）

    Returns:
        (log_probs, response_length): response部分的log probs和长度
    """
    full_text = prompt + response
    inputs = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(model.device)

    prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        outputs = model(**inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # 取response部分的logits和targets
    # logits[i] 预测 token[i+1]
    response_logits = logits[0, prompt_len - 1: -1]  # 预测response的每个token
    response_ids = inputs["input_ids"][0, prompt_len:]  # response的实际token

    # 计算 log softmax
    log_probs = F.log_softmax(response_logits, dim=-1)

    # 取每个token的实际log prob
    token_log_probs = log_probs.gather(
        -1, response_ids.unsqueeze(-1)
    ).squeeze(-1)  # [response_len]

    return token_log_probs, len(response_ids)


# ============================================================
# GRPO Loss
# ============================================================

def compute_grpo_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
    kl_coef: float = 0.04,
) -> Tuple[torch.Tensor, float]:
    """
    GRPO损失 = PPO裁剪损失 + KL散度惩罚

    Args:
        current_log_probs: 当前策略的log probs [seq_len]
        old_log_probs: 旧策略的log probs [seq_len]
        advantages: 相对优势标量
        epsilon: PPO裁剪参数
        kl_coef: KL散度惩罚系数
    Returns:
        (loss, kl_div): 损失标量和KL散度值
    """
    # 重要性采样比
    ratio = torch.exp(current_log_probs - old_log_probs)

    # PPO裁剪损失
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # KL散度惩罚（防止偏离旧策略太远）
    kl_div = (old_log_probs - current_log_probs).mean()

    # 总损失
    loss = policy_loss + kl_coef * kl_div

    return loss, kl_div.item()


# ============================================================
# GRPO Dataset
# ============================================================

class GRPODataset(Dataset):
    """GRPO训练数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data = load_json(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        if isinstance(question, list):
            question = "".join(question)
        answer = str(item["answer"])

        messages = [
            {"role": "system", "content": "解答小学数学题，按以下格式输出：\n步骤：你的计算过程\n答案：数字（仅数字，不带单位）"},
            {"role": "user", "content": question},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return {
            "prompt": prompt_text,
            "question": question,
            "answer": answer,
            "id": item.get("id", idx),
        }


# ============================================================
# GRPO 训练器
# ============================================================

class GRPOTrainer:
    """
    GRPO 训练器（正确实现版）

    训练流程（每个问题）：
    1. 采样 group_size 个回答，同时记录 old_log_probs
    2. 计算每个回答的奖励
    3. 计算组内相对优势（归一化）
    4. 对每个回答：
       a. 用当前策略计算 new_log_probs
       b. 计算 GRPO loss = PPO裁剪 + KL惩罚
       c. 累积梯度
    5. 一次性更新参数（group内所有回答共享一次更新）
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[GRPOConfig] = None,
        output_dir: str = "../outputs/scheme4_grpo",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GRPOConfig()
        self.output_dir = output_dir

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        os.makedirs(output_dir, exist_ok=True)

    def _sample_with_log_probs(
        self, prompt: str, group_size: int = 4,
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        采样 group_size 个回答，同时记录每个回答的 old_log_probs。

        Returns:
            (responses, old_log_probs_list)
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.model.device)
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=True)["input_ids"])

        responses = []
        old_log_probs_list = []

        for _ in range(group_size):
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # 裁掉输入部分
            out = generated[0][prompt_len:]
            response = self.tokenizer.decode(out, skip_special_tokens=True).strip()
            responses.append(response)

            # 计算 old_log_probs（采样时的策略）
            full_ids = generated[0]
            response_ids = full_ids[prompt_len:]
            response_len = len(response_ids)

            if response_len > 0:
                with torch.no_grad():
                    outputs = self.model(input_ids=generated)
                    logits = outputs.logits[0]
                    # logits[i] 预测 token[i+1]
                    resp_logits = logits[prompt_len - 1: prompt_len - 1 + response_len]
                    log_probs = F.log_softmax(resp_logits, dim=-1)
                    token_log_probs = log_probs.gather(
                        -1, response_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    old_log_probs_list.append(token_log_probs)
            else:
                old_log_probs_list.append(torch.tensor([], device=self.model.device))

        return responses, old_log_probs_list

    def train(self, train_dataset: GRPODataset, num_iterations: int = 100,
              group_size: int = 4, save_steps: int = 10, max_samples: int = None):
        """完整GRPO训练流程"""
        print(f"开始GRPO训练，{num_iterations}次迭代，group_size={group_size}")

        dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        global_step = 0
        progress_bar = tqdm(total=num_iterations, desc="GRPO训练")

        for iteration in range(num_iterations):
            self.model.train()
            iteration_metrics = {"reward": [], "loss": [], "kl": []}
            sample_count = 0

            for batch in dataloader:
                if max_samples and sample_count >= max_samples:
                    break
                sample_count += 1
                prompt = batch["prompt"][0]
                answer = batch["answer"][0]

                # ===== 步骤1：采样 + 记录 old_log_probs =====
                responses, old_log_probs_list = self._sample_with_log_probs(
                    prompt, group_size=group_size
                )

                # ===== 步骤2：计算奖励 =====
                rewards = [compute_reward(resp, answer) for resp in responses]

                # ===== 步骤3：计算组内相对优势 =====
                advantages = compute_group_relative_advantages(rewards)

                # ===== 步骤4：累积梯度，一次性更新 =====
                self.optimizer.zero_grad()
                group_loss = 0.0
                group_kl = 0.0
                active_samples = 0

                for response, old_log_probs, advantage in zip(
                    responses, old_log_probs_list, advantages
                ):
                    if abs(advantage) < 1e-6 or len(old_log_probs) == 0:
                        continue

                    # 用当前策略计算 new_log_probs（需要梯度）
                    current_log_probs, _ = compute_response_log_probs(
                        self.model, self.tokenizer, prompt, response,
                        max_length=self.config.max_length,
                        no_grad=False,
                    )

                    # 确保 old 和 current 长度一致
                    min_len = min(len(current_log_probs), len(old_log_probs))
                    if min_len == 0:
                        continue

                    current_log_probs = current_log_probs[:min_len]
                    old_log_probs = old_log_probs[:min_len]
                    adv_tensor = torch.tensor(
                        advantage, device=self.model.device, dtype=torch.float32
                    )

                    # 计算 GRPO loss（PPO裁剪 + KL惩罚）
                    loss, kl = compute_grpo_loss(
                        current_log_probs, old_log_probs, adv_tensor,
                        epsilon=self.config.epsilon,
                        kl_coef=self.config.kl_coef,
                    )

                    # 累积梯度（不立即step）
                    loss.backward()
                    group_loss += loss.item()
                    group_kl += kl
                    active_samples += 1

                # 一次性更新参数（group内共享）
                if active_samples > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    iteration_metrics["loss"].append(group_loss / active_samples)
                    iteration_metrics["kl"].append(group_kl / active_samples)

                iteration_metrics["reward"].extend(rewards)
                global_step += 1

            # ---- 日志输出 ----
            if (iteration + 1) % 10 == 0:
                avg_reward = sum(iteration_metrics["reward"]) / max(len(iteration_metrics["reward"]), 1)
                avg_loss = sum(iteration_metrics["loss"]) / max(len(iteration_metrics["loss"]), 1)
                avg_kl = sum(iteration_metrics["kl"]) / max(len(iteration_metrics["kl"]), 1)
                print(f"  迭代 {iteration+1}: reward={avg_reward:.3f}, loss={avg_loss:.4f}, kl={avg_kl:.4f}")

                if (iteration + 1) % save_steps == 0:
                    ckpt_path = os.path.join(self.output_dir, f"checkpoint-{iteration+1}")
                    self.model.save_pretrained(ckpt_path)
                    self.tokenizer.save_pretrained(ckpt_path)
                    print(f"  已保存: {ckpt_path}")

            progress_bar.update(1)

        # 保存最终模型
        final_path = os.path.join(self.output_dir, "final")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"GRPO模型已保存: {final_path}")
        progress_bar.close()


# ============================================================
# 入口函数
# ============================================================

def train_grpo(
    sft_model_path: str,
    sft_peft_path: str,
    grpo_data_path: str,
    output_dir: str,
    device: str = None,
    group_size: int = 4,
    num_iterations: int = 100,
    learning_rate: float = 1e-6,
    max_new_tokens: int = None,
    max_samples: int = None,
):
    """GRPO训练入口函数"""
    if device is None:
        device = get_device()
    print(f"设备: {device}")

    actual_data_path = grpo_data_path
    if not os.path.exists(actual_data_path):
        fallback = actual_data_path.replace("train_cot.json", "train_cot_original.json")
        if os.path.exists(fallback):
            print(f"  {actual_data_path} 不存在，使用 {fallback}")
            actual_data_path = fallback

    print_config({
        "sft_peft_path": sft_peft_path,
        "grpo_data_path": actual_data_path,
        "output_dir": output_dir,
        "group_size": group_size,
        "num_iterations": num_iterations,
        "learning_rate": learning_rate,
    })

    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        sft_model_path, use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        device_map=device if device != "cpu" else None,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    )

    if sft_peft_path and os.path.exists(sft_peft_path):
        is_peft = os.path.exists(os.path.join(sft_peft_path, "adapter_config.json"))
        if is_peft:
            model = PeftModel.from_pretrained(base_model, sft_peft_path)
            print(f"已加载PEFT权重: {sft_peft_path}")
            model = model.merge_and_unload()
        else:
            del base_model
            model = AutoModelForCausalLM.from_pretrained(
                sft_peft_path,
                device_map=device if device != "cpu" else None,
                torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
                trust_remote_code=True,
            )
            print(f"已加载完整模型: {sft_peft_path}")
    else:
        model = base_model

    if device == "cpu":
        model = model.to(device)

    print("准备GRPO数据集...")
    dataset = GRPODataset(actual_data_path, tokenizer)

    config = GRPOConfig()
    config.group_size = group_size
    config.num_iterations = num_iterations
    config.learning_rate = learning_rate
    if max_new_tokens is not None:
        config.max_new_tokens = max_new_tokens

    trainer = GRPOTrainer(model, tokenizer, config, output_dir)
    trainer.train(train_dataset=dataset, num_iterations=num_iterations, group_size=group_size, max_samples=max_samples)

    return trainer


def run_grpo_from_notebook(
    sft_model_path: str,
    sft_peft_path: str,
    grpo_data_path: str,
    output_dir: str,
    device: str = "cuda",
):
    """从Notebook调用的便捷函数"""
    os.makedirs(output_dir, exist_ok=True)
    return train_grpo(
        sft_model_path=sft_model_path,
        sft_peft_path=sft_peft_path,
        grpo_data_path=grpo_data_path,
        output_dir=output_dir,
        device=device,
        group_size=4,
        num_iterations=50,
        learning_rate=1e-6,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model_path", type=str, default="../models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft_peft_path", type=str, default="../outputs/scheme2_cot/final")
    parser.add_argument("--grpo_data_path", type=str, default="../data/train_cot.json")
    parser.add_argument("--output_dir", type=str, default="../outputs/scheme4_grpo")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=50)
    args = parser.parse_args()

    train_grpo(
        sft_model_path=args.sft_model_path,
        sft_peft_path=args.sft_peft_path,
        grpo_data_path=args.grpo_data_path,
        output_dir=args.output_dir,
        device=args.device,
        group_size=args.group_size,
        num_iterations=args.num_iterations,
    )
