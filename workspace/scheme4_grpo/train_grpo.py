"""
方案4：GRPO（组相对策略优化）训练
参考DeepSeek-R1论文，专为数学推理设计

核心算法：
1. 对每个问题采样 group_size 个回答（多路径推理）
2. 用规则奖励函数给每个回答打分（答案正确性 + 格式奖励）
3. 计算组内相对优势（归一化，零均值）
4. 用 PPO-style 裁剪损失 + KL散度惩罚 更新策略

A10 24GB 优化：
- 混合精度训练：bf16 autocast
- 采样+log_prob合并：减少冗余前向传播
- mini-batch训练：控制每轮步数

审查修复记录：
1. optimizer_step 持久化：修复断点续训时 warmup 重启问题
2. 指标记录：移到条件外部，每次都记录 loss/kl
3. DataLoader seed：固定 shuffle 种子，确保断点续训可复现
4. empty_cache 频率：从 100 步降低到 500 步
5. CLI 默认值：统一 group_size 默认值为 8
6. _get_attn_impl：提前缓存避免重复调用
7. 死代码删除：移除未使用的 compute_response_log_probs
8. 路径解析：添加 _resolve_relative_path 确保路径一致性
"""

import os
import sys
import re
import gc
from itertools import islice
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

from utils.common import load_json, print_config, get_device, ensure_flash_attn, enable_tf32, set_seed


def _resolve_relative_path(path: str) -> str:
    """将相对路径解析为绝对路径（相对于脚本目录）"""
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, path))


def _get_attn_impl():
    try:
        import flash_attn
        return "flash_attention_2"
    except ImportError:
        return None


@dataclass
class GRPOConfig:
    group_size: int = 4
    num_epochs: int = 3
    max_steps_per_epoch: int = 1000
    max_length: int = 512
    max_new_tokens: int = 128
    gradient_accumulation_steps: int = 2

    learning_rate: float = 5e-6
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


def parse_number(num_str: str) -> float | None:
    """解析数字字符串，支持分数。解析失败返回 None。"""
    if not num_str:
        return None
    try:
        if '/' in num_str:
            parts = num_str.split('/')
            if len(parts) != 2:
                return None
            numerator, denominator = float(parts[0]), float(parts[1])
            if denominator == 0:
                return None
            return numerator / denominator
        else:
            return float(num_str)
    except (ValueError, ZeroDivisionError):
        return None


def compute_reward(generated_text: str, ground_truth: str, reward_correct: float = 1.0, reward_format: float = 0.1) -> float:
    reward = 0.0

    gen_num = extract_number(generated_text)
    truth_num = extract_number(ground_truth)
    gen_val = parse_number(gen_num)
    truth_val = parse_number(truth_num)

    if gen_val is not None and truth_val is not None:
        if abs(gen_val - truth_val) < 0.01:
            reward += reward_correct

    if re.search(r'步骤|先|再|然后|因此|所以', generated_text):
        reward += reward_format / 2
    if re.search(r'答案[：:]', generated_text):
        reward += reward_format / 2

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


def compute_batch_grpo_loss(
    current_log_probs_list: List[torch.Tensor],
    old_log_probs_list: List[torch.Tensor],
    advantages_list: List[float],
    epsilon: float = 0.2,
    kl_coef: float = 0.04,
) -> Tuple[torch.Tensor, float, int]:
    total_loss = None
    total_kl = 0.0
    valid_count = 0

    for current_log_probs, old_log_probs, advantage in zip(
        current_log_probs_list, old_log_probs_list, advantages_list
    ):
        min_len = min(len(current_log_probs), len(old_log_probs))
        if min_len == 0 or abs(advantage) < 1e-6:
            continue

        cur_lp = current_log_probs[:min_len]
        old_lp = old_log_probs[:min_len]
        adv = torch.tensor(advantage, device=current_log_probs.device, dtype=torch.float32)

        ratio = torch.exp(cur_lp - old_lp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        kl_div = (old_lp - cur_lp).mean()

        loss = policy_loss + kl_coef * kl_div

        if total_loss is None:
            total_loss = loss
        else:
            total_loss = total_loss + loss

        total_kl += kl_div.item()
        valid_count += 1

    return total_loss, total_kl, valid_count


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
    1. 混合精度：bf16 autocast 加速训练
    2. 采样+log_prob合并：generate后直接提取log_probs，省4次前向
    3. 梯度累积：减少optimizer步数
    4. mini-batch：每轮最多 max_steps_per_epoch 步
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[GRPOConfig] = None,
        output_dir: str = "../outputs/scheme4_grpo",
        optim: str = "adamw_torch_fused",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GRPOConfig()
        self.output_dir = output_dir

        self.optimizer = self._create_optimizer(self.config, optim)
        self.scheduler = None

        os.makedirs(output_dir, exist_ok=True)

    def _create_optimizer(self, config, optim: str = "adamw_torch_fused"):
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if optim.startswith("adamw"):
            try:
                return torch.optim.AdamW(
                    params,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    fused=True if "fused" in optim else False,
                )
            except TypeError:
                return torch.optim.AdamW(
                    params,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
        else:
            # 默认使用 AdamW
            return torch.optim.AdamW(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

    def _sample_responses(self, prompt: str, group_size: int = 4,
                          no_grad_log_probs: bool = True,
                          ) -> Tuple[List[str], List[torch.Tensor]]:
        inputs = self.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        expanded = {k: v.expand(group_size, -1) for k, v in inputs.items()}

        was_training = self.model.training
        self.model.eval()

        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=self.model.device.type == "cuda"
        ):
            generated = self.model.generate(
                **expanded,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_cpu = generated.cpu()
        del generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        responses = []
        log_probs_list = []
        response_lens = []

        for i in range(group_size):
            response_ids = generated_cpu[i, prompt_len:]
            response_len = len(response_ids)
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            responses.append(response)
            response_lens.append(response_len)

        valid_indices = [i for i, rl in enumerate(response_lens) if rl > 0]

        if valid_indices:
            max_resp_len = max(response_lens[i] for i in valid_indices)
            max_seq_len = prompt_len + max_resp_len
            padded = torch.full(
                (len(valid_indices), max_seq_len),
                self.tokenizer.pad_token_id or 0,
                dtype=torch.long,
            )
            attn_mask = torch.zeros(len(valid_indices), max_seq_len, dtype=torch.long)

            for batch_idx, i in enumerate(valid_indices):
                seq = generated_cpu[i, :prompt_len + response_lens[i]]
                padded[batch_idx, :len(seq)] = seq
                attn_mask[batch_idx, :len(seq)] = 1

            ctx = torch.no_grad() if no_grad_log_probs else torch.enable_grad()
            with ctx, torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
                enabled=self.model.device.type == "cuda"
            ):
                batch_outputs = self.model(input_ids=padded.to(self.model.device),
                                           attention_mask=attn_mask.to(self.model.device))
                batch_logits = batch_outputs.logits

            for batch_idx, i in enumerate(valid_indices):
                rl = response_lens[i]
                resp_logits = batch_logits[batch_idx, prompt_len - 1: prompt_len - 1 + rl]
                log_probs = F.log_softmax(resp_logits, dim=-1)
                response_ids_tensor = generated_cpu[i, prompt_len: prompt_len + rl].to(self.model.device)
                token_log_probs = log_probs.gather(
                    -1, response_ids_tensor.unsqueeze(-1)
                ).squeeze(-1)
                if no_grad_log_probs:
                    token_log_probs = token_log_probs.detach()
                log_probs_list.append((i, token_log_probs))

            del padded, attn_mask, batch_outputs, batch_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result_log_probs = [
            torch.tensor([], device=self.model.device) for _ in range(group_size)
        ]
        for i, lp in log_probs_list:
            result_log_probs[i] = lp

        if was_training:
            self.model.train()

        return responses, result_log_probs

    def _batch_compute_log_probs(self, prompt: str, responses: List[str]) -> List[torch.Tensor]:
        """批量计算多个response的log_probs（带梯度），单次前向传播"""
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)

        all_seq_ids = []
        all_resp_lens = []
        for resp in responses:
            resp_ids = self.tokenizer(resp, add_special_tokens=False)["input_ids"]
            full_ids = prompt_ids + resp_ids
            if len(full_ids) > self.config.max_length:
                full_ids = full_ids[:self.config.max_length]
                resp_ids = full_ids[prompt_len:]
            all_seq_ids.append(full_ids)
            all_resp_lens.append(len(resp_ids))

        max_seq_len = max(len(s) for s in all_seq_ids)
        batch_size = len(all_seq_ids)
        padded = torch.full((batch_size, max_seq_len), self.tokenizer.pad_token_id or 0, dtype=torch.long)
        attn_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

        for i, seq in enumerate(all_seq_ids):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn_mask[i, :len(seq)] = 1

        with torch.enable_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=self.model.device.type == "cuda"
        ):
            outputs = self.model(input_ids=padded.to(self.model.device),
                                 attention_mask=attn_mask.to(self.model.device))
            batch_logits = outputs.logits

        result = []
        for i in range(batch_size):
            rl = all_resp_lens[i]
            if rl == 0:
                result.append(torch.tensor([], device=self.model.device))
                continue
            resp_logits = batch_logits[i, prompt_len - 1: prompt_len - 1 + rl]
            log_probs = F.log_softmax(resp_logits, dim=-1)
            resp_ids_tensor = torch.tensor(all_seq_ids[i][prompt_len: prompt_len + rl],
                                           device=self.model.device)
            token_log_probs = log_probs.gather(-1, resp_ids_tensor.unsqueeze(-1)).squeeze(-1)
            result.append(token_log_probs)

        del padded, attn_mask, batch_logits, outputs
        return result

    def _save_checkpoint(self, global_step, epoch, step_in_epoch, optimizer_step):
        ckpt_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
        self.model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)
        torch.save({
            "global_step": global_step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "optimizer_step": optimizer_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }, os.path.join(ckpt_path, "training_state.pt"))
        print(f"  已保存: {ckpt_path}")

    def _load_checkpoint(self):
        if not os.path.exists(self.output_dir):
            return 0, 0, 0, 0, None
        checkpoints = [f for f in os.listdir(self.output_dir) if f.startswith("checkpoint-")]
        if not checkpoints:
            return 0, 0, 0, 0, None
        latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        ckpt_path = os.path.join(self.output_dir, latest)
        state_path = os.path.join(ckpt_path, "training_state.pt")
        if not os.path.exists(state_path):
            return 0, 0, 0, 0, None
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        optimizer_step = state.get("optimizer_step", 0)
        print(f"  从 checkpoint 恢复: {ckpt_path} (global_step={state['global_step']}, optimizer_step={optimizer_step})")
        return state["global_step"], state["epoch"], state["step_in_epoch"], optimizer_step, ckpt_path

    @staticmethod
    def find_latest_checkpoint(output_dir: str):
        if not os.path.exists(output_dir):
            return None
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
        if not checkpoints:
            return None
        latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        ckpt_path = os.path.join(output_dir, latest)
        state_path = os.path.join(ckpt_path, "training_state.pt")
        if not os.path.exists(state_path):
            return None
        return ckpt_path

    def train(self, train_dataset: GRPODataset, num_epochs: int = 3,
              group_size: int = 4, max_steps_per_epoch: int = 500,
              gradient_accumulation_steps: int = 2, save_steps: int = 50,
              max_samples: int = None, dataloader_num_workers: int = 4,
              seed: int = 42):
        total_steps = num_epochs * max_steps_per_epoch
        total_optimizer_steps = total_steps // gradient_accumulation_steps
        print(f"开始GRPO训练: {num_epochs}轮, 每轮最多{max_steps_per_epoch}步, group_size={group_size}")
        print(f"  梯度累积: {gradient_accumulation_steps}, 总步数上限: {total_steps}, optimizer步数: {total_optimizer_steps}")

        # 固定 shuffle 种子，确保断点续训时数据顺序可复现
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
                               num_workers=dataloader_num_workers, generator=generator)
        global_step = 0
        optimizer_step = 0
        progress_bar = tqdm(total=total_steps, desc="GRPO训练")

        warmup_steps = int(total_optimizer_steps * self.config.warmup_ratio)
        cosine_steps = total_optimizer_steps - warmup_steps
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step / max(warmup_steps, 1), 1.0)
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(cosine_steps, 1), eta_min=1e-7
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        resume_global_step, resume_epoch, resume_step_in_epoch, resume_opt_step, resume_ckpt_path = self._load_checkpoint()
        if resume_global_step > 0:
            global_step = resume_global_step
            optimizer_step = resume_opt_step
            progress_bar.update(global_step)
            remaining_steps = total_steps - global_step
            progress_bar.total = remaining_steps + global_step

        start_epoch = resume_epoch if resume_global_step > 0 else 0

        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            epoch_metrics = {"reward": [], "loss": [], "kl": []}
            step_in_epoch = resume_step_in_epoch if epoch == start_epoch and resume_global_step > 0 else 0

            if step_in_epoch > 0:
                dataloader_iter = iter(dataloader)
                dataloader = list(islice(dataloader_iter, step_in_epoch, None))
                dataloader = iter(dataloader)
            else:
                dataloader = iter(dataloader)

            for batch_idx, batch in enumerate(dataloader):
                if step_in_epoch >= max_steps_per_epoch:
                    break
                if max_samples and step_in_epoch >= max_samples:
                    break

                prompt = batch["prompt"][0]
                answer = batch["answer"][0]

                responses, old_log_probs_list = self._sample_responses(
                    prompt, group_size=group_size, no_grad_log_probs=True
                )

                rewards = [compute_reward(resp, answer, self.config.reward_correct, self.config.reward_format) for resp in responses]
                advantages = compute_group_relative_advantages(rewards)

                group_loss = 0.0
                group_kl = 0.0
                active_samples = 0

                active_items = []
                for idx, (response, old_log_probs, advantage) in enumerate(
                    zip(responses, old_log_probs_list, advantages)
                ):
                    if abs(advantage) < 1e-6 or len(old_log_probs) == 0:
                        continue
                    active_items.append((response, old_log_probs, advantage))

                if active_items:
                    all_current_log_probs = self._batch_compute_log_probs(
                        prompt, [r for r, _, _ in active_items]
                    )

                    active_old_log_probs = [lp for _, lp, _ in active_items]
                    active_advantages = [adv for _, _, adv in active_items]

                    total_loss, total_kl, active_samples = compute_batch_grpo_loss(
                        all_current_log_probs,
                        active_old_log_probs,
                        active_advantages,
                        epsilon=self.config.epsilon,
                        kl_coef=self.config.kl_coef,
                    )

                    if total_loss is not None and active_samples > 0:
                        scaled_loss = total_loss / (len(active_items) * gradient_accumulation_steps)
                        scaled_loss.backward()
                        group_loss = total_loss.item()
                        group_kl = total_kl

                if active_samples > 0 and (step_in_epoch + 1) % gradient_accumulation_steps == 0:
                    trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    self.optimizer.step()
                    optimizer_step += 1
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if active_samples > 0:
                    epoch_metrics["loss"].append(group_loss / active_samples)
                    epoch_metrics["kl"].append(group_kl / active_samples)

                epoch_metrics["reward"].extend(rewards)
                global_step += 1
                step_in_epoch += 1
                progress_bar.update(1)

                if global_step % 500 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if global_step % save_steps == 0 and global_step > 0:
                    self._save_checkpoint(global_step, epoch, step_in_epoch, optimizer_step)

            avg_reward = sum(epoch_metrics["reward"]) / max(len(epoch_metrics["reward"]), 1)
            avg_loss = sum(epoch_metrics["loss"]) / max(len(epoch_metrics["loss"]), 1)
            avg_kl = sum(epoch_metrics["kl"]) / max(len(epoch_metrics["kl"]), 1)
            print(f"  Epoch {epoch+1}/{num_epochs}: reward={avg_reward:.3f}, loss={avg_loss:.4f}, kl={avg_kl:.4f}, steps={step_in_epoch}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
    group_size: int = 8,
    num_iterations: int = 3,
    learning_rate: float = 5e-6,
    max_new_tokens: int = 256,
    max_length: int = 512,
    max_samples: int = None,
    max_steps_per_epoch: int = 1000,
    gradient_accumulation_steps: int = 2,
    save_steps: int = 50,
    dataloader_num_workers: int = 4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    weight_decay: float = 0.01,
    optim: str = "adamw_torch_fused",
    seed: int = 42,
):
    if device is None:
        device = get_device()
    print(f"设备: {device}")

    # 解析路径，确保路径一致性
    sft_model_path = _resolve_relative_path(sft_model_path)
    if sft_peft_path:
        sft_peft_path = _resolve_relative_path(sft_peft_path)
    grpo_data_path = _resolve_relative_path(grpo_data_path)
    output_dir = _resolve_relative_path(output_dir)

    actual_data_path = grpo_data_path
    if not os.path.exists(actual_data_path):
        fallback = actual_data_path.replace("train_cot.json", "train_cot_original.json")
        if os.path.exists(fallback):
            print(f"  {actual_data_path} 不存在，使用 {fallback}")
            actual_data_path = fallback

    # 检查是否已经训练完成
    final_path = os.path.join(output_dir, "final")
    if os.path.exists(final_path):
        print(f"✓ 检测到已完成的GRPO模型: {final_path}")
        print("  跳过训练（如需重新训练，请删除该目录）")
        return None

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
            model = model.merge_and_unload()
            del base_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            del base_model
            gc.collect()
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

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    if device == "cpu":
        model = model.to(device)

    checkpoint_path = GRPOTrainer.find_latest_checkpoint(output_dir)
    if checkpoint_path:
        print(f"从 checkpoint 恢复模型权重: {checkpoint_path}")
        model = PeftModel.from_pretrained(model.get_base_model(), checkpoint_path)
        print(f"已从 checkpoint 恢复模型权重")

    print("准备GRPO数据集...")
    dataset = GRPODataset(actual_data_path, tokenizer, max_length=max_length)

    config = GRPOConfig()
    config.group_size = group_size
    config.num_epochs = num_iterations
    config.learning_rate = learning_rate
    config.max_steps_per_epoch = max_steps_per_epoch
    config.gradient_accumulation_steps = gradient_accumulation_steps
    config.max_length = max_length
    config.weight_decay = weight_decay
    config.max_new_tokens = max_new_tokens

    trainer = GRPOTrainer(model, tokenizer, config, output_dir, optim=optim)
    trainer.train(
        train_dataset=dataset,
        num_epochs=num_iterations,
        group_size=group_size,
        max_steps_per_epoch=max_steps_per_epoch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        max_samples=max_samples,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
    )

    return trainer


def run_grpo_from_notebook(
    sft_model_path: str,
    sft_peft_path: str,
    grpo_data_path: str,
    output_dir: str,
    device: str = "cuda",
    **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)
    return train_grpo(
        sft_model_path=sft_model_path,
        sft_peft_path=sft_peft_path,
        grpo_data_path=grpo_data_path,
        output_dir=output_dir,
        device=device,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model_path", type=str, default="../models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft_peft_path", type=str, default="../outputs/scheme2_cot/final")
    parser.add_argument("--grpo_data_path", type=str, default="../data/train_cot.json")
    parser.add_argument("--output_dir", type=str, default="../outputs/scheme4_grpo")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--optim", type=str, default="adamw_torch_fused")
    args = parser.parse_args()

    train_grpo(
        sft_model_path=args.sft_model_path,
        sft_peft_path=args.sft_peft_path,
        grpo_data_path=args.grpo_data_path,
        output_dir=args.output_dir,
        device=args.device,
        group_size=args.group_size,
        num_iterations=args.num_iterations,
        max_new_tokens=args.max_new_tokens,
        optim=args.optim,
    )
