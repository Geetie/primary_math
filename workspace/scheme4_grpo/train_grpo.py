"""
方案4：GRPO（组相对策略优化）训练
参考DeepSeek-R1论文，专为数学推理设计

核心算法：
1. 对每个问题采样 group_size 个回答（多路径推理）
2. 用规则奖励函数给每个回答打分（答案正确性 + 格式奖励）
3. 计算组内相对优势（归一化，零均值）
4. 用 PPO-style 裁剪损失 + KL散度惩罚 更新策略

A10 24GB 优化：
- 梯度检查点：降低显存占用
- 混合精度训练：bf16 autocast
- 采样+log_prob合并：减少冗余前向传播
- mini-batch训练：控制每轮步数
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


class GRPOConfig:
    """GRPO训练配置"""
    group_size: int = 4
    num_epochs: int = 3
    max_steps_per_epoch: int = 200
    max_length: int = 512
    max_new_tokens: int = 128
    gradient_accumulation_steps: int = 2

    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    epsilon: float = 0.2
    kl_coef: float = 0.04
    reward_correct: float = 1.0
    reward_format: float = 0.1


def extract_number(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    ans_match = re.search(r'答案[：:]\s*(-?\d+\.?\d*)', text)
    if ans_match:
        return ans_match.group(1)
    matches = re.findall(r'-?\d+\.?\d*', text)
    return matches[-1] if matches else ""


def compute_reward(generated_text: str, ground_truth: str) -> float:
    reward = 0.0

    gen_num = extract_number(generated_text)
    truth_num = extract_number(ground_truth)
    if gen_num and truth_num:
        try:
            if '/' in gen_num:
                parts = gen_num.split('/')
                gen_val = float(parts[0]) / float(parts[1]) if len(parts) == 2 and float(parts[1]) != 0 else float('inf')
            else:
                gen_val = float(gen_num)
            if '/' in truth_num:
                parts = truth_num.split('/')
                truth_val = float(parts[0]) / float(parts[1]) if len(parts) == 2 and float(parts[1]) != 0 else float('inf')
            else:
                truth_val = float(truth_num)
            if abs(gen_val - truth_val) < 0.01:
                reward += 1.0
        except (ValueError, ZeroDivisionError, IndexError):
            pass

    if re.search(r'步骤|先|再|然后|因此|所以', generated_text):
        reward += 0.05
    if re.search(r'答案[：:]', generated_text):
        reward += 0.05

    return reward


def compute_group_relative_advantages(rewards: List[float]) -> List[float]:
    n = len(rewards)
    if n <= 1:
        return [0.0] * n

    mean_reward = sum(rewards) / n
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / n) ** 0.5

    if std_reward < 1e-6:
        return [0.0] * n

    advantages = [(r - mean_reward) / std_reward for r in rewards]
    return advantages


def compute_response_log_probs(
    model, tokenizer, prompt_text: str, response: str,
    max_length: int = 512, no_grad: bool = True,
) -> Tuple[torch.Tensor, int]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    prompt_len = len(prompt_ids)

    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    full_ids = prompt_ids + response_ids
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        response_ids = full_ids[prompt_len:]

    input_ids = torch.tensor([full_ids], device=model.device)

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]

    response_len = len(response_ids)
    if response_len == 0:
        return torch.tensor([], device=model.device), 0

    resp_logits = logits[prompt_len - 1: prompt_len - 1 + response_len]
    resp_ids_tensor = torch.tensor(response_ids, device=model.device)

    log_probs = F.log_softmax(resp_logits, dim=-1)
    token_log_probs = log_probs.gather(
        -1, resp_ids_tensor.unsqueeze(-1)
    ).squeeze(-1)

    return token_log_probs, response_len


def compute_grpo_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
    kl_coef: float = 0.04,
) -> Tuple[torch.Tensor, float]:
    ratio = torch.exp(current_log_probs - old_log_probs)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    kl_div = (old_log_probs - current_log_probs).mean()

    loss = policy_loss + kl_coef * kl_div

    return loss, kl_div.item()


class GRPODataset(Dataset):
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


class GRPOTrainer:
    """
    GRPO 训练器（A10 24GB 优化版）

    优化点：
    1. 梯度检查点：降低显存 ~60%
    2. 混合精度：bf16 autocast 加速训练
    3. 采样+log_prob合并：generate后直接提取log_probs，省4次前向
    4. 梯度累积：减少optimizer步数
    5. mini-batch：每轮最多 max_steps_per_epoch 步
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
        inputs = self.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

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

            response_ids = generated[0, prompt_len:]
            response_len = len(response_ids)
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            responses.append(response)

            if response_len > 0:
                with torch.no_grad():
                    outputs = self.model(input_ids=generated)
                    logits = outputs.logits[0]
                    resp_logits = logits[prompt_len - 1: prompt_len - 1 + response_len]
                    log_probs = F.log_softmax(resp_logits, dim=-1)
                    token_log_probs = log_probs.gather(
                        -1, response_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    old_log_probs_list.append(token_log_probs)
            else:
                old_log_probs_list.append(torch.tensor([], device=self.model.device))

        return responses, old_log_probs_list

    def train(self, train_dataset: GRPODataset, num_epochs: int = 3,
              group_size: int = 4, max_steps_per_epoch: int = 200,
              gradient_accumulation_steps: int = 2, save_steps: int = 50,
              max_samples: int = None):
        total_steps = num_epochs * max_steps_per_epoch
        print(f"开始GRPO训练: {num_epochs}轮, 每轮最多{max_steps_per_epoch}步, group_size={group_size}")
        print(f"  梯度累积: {gradient_accumulation_steps}, 总步数上限: {total_steps}")

        dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        global_step = 0
        progress_bar = tqdm(total=total_steps, desc="GRPO训练")

        for epoch in range(num_epochs):
            self.model.train()
            epoch_metrics = {"reward": [], "loss": [], "kl": []}
            step_in_epoch = 0

            for batch in dataloader:
                if step_in_epoch >= max_steps_per_epoch:
                    break
                if max_samples and step_in_epoch >= max_samples:
                    break

                prompt = batch["prompt"][0]
                answer = batch["answer"][0]

                responses, old_log_probs_list = self._sample_with_log_probs(
                    prompt, group_size=group_size
                )

                rewards = [compute_reward(resp, answer) for resp in responses]
                advantages = compute_group_relative_advantages(rewards)

                self.optimizer.zero_grad()
                group_loss = 0.0
                group_kl = 0.0
                active_samples = 0

                for response, old_log_probs, advantage in zip(
                    responses, old_log_probs_list, advantages
                ):
                    if abs(advantage) < 1e-6 or len(old_log_probs) == 0:
                        continue

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        enabled=self.model.device.type == "cuda"):
                        current_log_probs, _ = compute_response_log_probs(
                            self.model, self.tokenizer, prompt, response,
                            max_length=self.config.max_length,
                            no_grad=False,
                        )

                    min_len = min(len(current_log_probs), len(old_log_probs))
                    if min_len == 0:
                        continue

                    current_log_probs = current_log_probs[:min_len]
                    old_log_probs = old_log_probs[:min_len].detach()
                    adv_tensor = torch.tensor(
                        advantage, device=self.model.device, dtype=torch.float32
                    )

                    loss, kl = compute_grpo_loss(
                        current_log_probs, old_log_probs, adv_tensor,
                        epsilon=self.config.epsilon,
                        kl_coef=self.config.kl_coef,
                    )

                    loss = loss / (group_size * gradient_accumulation_steps)
                    loss.backward()
                    group_loss += loss.item() * group_size * gradient_accumulation_steps
                    group_kl += kl
                    active_samples += 1

                if active_samples > 0 and (step_in_epoch + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    epoch_metrics["loss"].append(group_loss / active_samples)
                    epoch_metrics["kl"].append(group_kl / active_samples)

                epoch_metrics["reward"].extend(rewards)
                global_step += 1
                step_in_epoch += 1
                progress_bar.update(1)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if global_step % save_steps == 0 and global_step > 0:
                    ckpt_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                    self.model.save_pretrained(ckpt_path)
                    self.tokenizer.save_pretrained(ckpt_path)
                    print(f"  已保存: {ckpt_path}")

            avg_reward = sum(epoch_metrics["reward"]) / max(len(epoch_metrics["reward"]), 1)
            avg_loss = sum(epoch_metrics["loss"]) / max(len(epoch_metrics["loss"]), 1)
            avg_kl = sum(epoch_metrics["kl"]) / max(len(epoch_metrics["kl"]), 1)
            print(f"  Epoch {epoch+1}/{num_epochs}: reward={avg_reward:.3f}, loss={avg_loss:.4f}, kl={avg_kl:.4f}, steps={step_in_epoch}")

        final_path = os.path.join(self.output_dir, "final")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"GRPO模型已保存: {final_path}")
        progress_bar.close()


def train_grpo(
    sft_model_path: str,
    sft_peft_path: str,
    grpo_data_path: str,
    output_dir: str,
    device: str = None,
    group_size: int = 4,
    num_iterations: int = 3,
    learning_rate: float = 1e-6,
    max_new_tokens: int = None,
    max_samples: int = None,
):
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
        "num_epochs": num_iterations,
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
            del base_model
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

    model.enable_input_require_grads()

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("已启用梯度检查点")

    from peft import LoraConfig, TaskType, get_peft_model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("准备GRPO数据集...")
    dataset = GRPODataset(actual_data_path, tokenizer)

    config = GRPOConfig()
    config.group_size = group_size
    config.num_epochs = num_iterations
    config.learning_rate = learning_rate
    if max_new_tokens is not None:
        config.max_new_tokens = max_new_tokens

    trainer = GRPOTrainer(model, tokenizer, config, output_dir)
    trainer.train(
        train_dataset=dataset,
        num_epochs=num_iterations,
        group_size=group_size,
        max_steps_per_epoch=config.max_steps_per_epoch,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_samples=max_samples,
    )

    return trainer


def run_grpo_from_notebook(
    sft_model_path: str,
    sft_peft_path: str,
    grpo_data_path: str,
    output_dir: str,
    device: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    return train_grpo(
        sft_model_path=sft_model_path,
        sft_peft_path=sft_peft_path,
        grpo_data_path=grpo_data_path,
        output_dir=output_dir,
        device=device,
        group_size=4,
        num_iterations=3,
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
    parser.add_argument("--num_iterations", type=int, default=3)
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
