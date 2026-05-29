# 小学数学应用题自动解题 - FDU AI课程作业

复旦大学《人工智能-LLM实践》课程作业项目

## 项目结构

```
primary_math/
├── data/                       # 数据集
│   ├── train.json             # 训练数据（12000条）
│   ├── test.json              # 测试数据（8000条）
│   ├── submit.csv             # 提交模板
│   ├── train_with_cot.json    # 带COT的训练数据（方案2生成）
│   └── train_augmented.json   # 增强的训练数据（方案2生成）
├── models/                     # 模型缓存目录
│   └── Qwen2.5-0.5B-Instruct/ # Qwen模型
├── outputs/                    # 输出目录
│   ├── baseline/              # Baseline模型
│   ├── scheme1_cot/           # 方案1结果
│   ├── scheme2_cot/           # 方案2模型
│   ├── baseline_submit.csv    # Baseline提交文件
│   ├── scheme1_submit_*.csv   # 方案1提交文件
│   └── scheme2_submit.csv     # 方案2提交文件
├── utils/                      # 工具模块
│   ├── common.py              # 通用工具函数
│   └── model_loader.py        # 模型加载工具
├── baseline/                   # Baseline方案
│   ├── train.py               # 训练脚本
│   └── inference.py           # 推理脚本
├── scheme1_cot/               # 方案1: COT优化
│   ├── cot_prompts.py         # COT提示模板
│   └── inference.py           # 推理脚本
├── scheme2_data_enhancement/  # 方案2: 数据构建与增强
│   ├── generate_cot_data.py   # 生成COT数据
│   ├── train_with_cot.py      # 使用COT数据训练
│   └── inference.py           # 推理脚本
├── scheme3_dpo/               # 方案3: RLHF+DPO（可选）
├── scheme4_grpo/              # 方案4: GRPO（可选）
├── run_baseline.bat           # 运行Baseline
├── run_scheme1.bat            # 运行方案1
├── run_scheme2.bat            # 运行方案2
└── README.md                  # 项目说明
```

## 环境要求

- Python 3.8+
- PyTorch
- Transformers
- PEFT
- 其他依赖见 `requirements.txt`

## 快速开始

### 1. 安装依赖

```bash
pip install transformers peft torch tqdm modelscope
```

### 2. 运行Baseline

双击运行 `run_baseline.bat` 或在命令行执行：

```bash
# 训练
py baseline\train.py

# 推理
py baseline\inference.py
```

### 3. 运行方案1（COT优化）

```bash
py scheme1_cot\inference.py
```

### 4. 运行方案2（数据增强）

```bash
# 生成COT数据
py scheme2_data_enhancement\generate_cot_data.py

# 训练
py scheme2_data_enhancement\train_with_cot.py

# 推理
py scheme2_data_enhancement\inference.py
```

## 方案说明

### Baseline方案
- 基于LoRA微调Qwen2.5-0.5B
- 使用原始训练数据直接微调
- 输出数字答案

### 方案1：COT优化
- Zero-shot COT：通过提示词激活模型推理能力
- Few-shot COT：提供示例引导模型推理
- 对比不同提示方式的效果

### 方案2：数据构建与增强
- 为训练数据生成COT步骤
- 通过数字替换进行数据增强
- 使用带COT的数据进行SFT微调

### 方案3：RLHF+DPO（可选）
- 构建偏好数据（正确vs错误COT）
- 使用DPO进行对齐优化

### 方案4：GRPO（可选）
- 实现组相对策略优化
- 参考Open-R1复现代码

## 提交文件

所有方案的提交文件都保存在 `outputs/` 目录：
- `baseline_submit.csv` - Baseline结果
- `scheme1_submit_*.csv` - 方案1不同提示的结果
- `scheme2_submit.csv` - 方案2结果

提交格式：
```csv
id,ret
0,85
1,64
...
```

## 注意事项

1. **模型限制**：仅使用0.5B及以下参数模型（Qwen2.5-0.5B）
2. **数据约束**：严禁处理测试数据，仅可对训练数据进行增强
3. **断点续训**：所有训练脚本支持断点续训
4. **自动下载**：模型会自动下载并缓存，首次运行需要联网

## 评分标准

- 比赛结果分：`s1 = 正确率 * 15`
- 工作量分：`s2 = 完成方案数 * 5`（个人）或 `* 3`（组队）
- 总分：`s = min(s1 + s2, 15)`

## 截止时间

- 比赛提交：2026年06月11日
- 报告提交：2026年06月19日

## 参考资料

- [比赛官网](https://www.datafountain.cn/competitions/467)
- [Qwen2.5模型](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [官方Baseline仓库](https://github.com/AI-FDU/Math_Solver)
