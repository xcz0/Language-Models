# Language-Models

使用 PyTorch Lightning 框架实现各类基础语言模型的教学项目，包括 Bigram、MLP、RNN、GPT 等模型。本项目专注于字符级语言建模，适合学习和理解不同模型架构的工作原理。

## 📋 项目概述

本项目实现了多种经典的语言模型架构：

- **Bigram**: 最简单的统计语言模型，基于二元组
- **Bag of Words (BoW)**: 词袋模型，不考虑词序的文本表示
- **MLP**: 多层感知机，简单的前馈神经网络
- **RNN**: 循环神经网络，处理序列数据的经典架构
- **GPT**: 基于 Transformer 解码器的生成式预训练模型

## 🚀 快速开始

### 环境要求

- Python 3.12+
- PyTorch
- PyTorch Lightning
- 其他依赖见 `pyproject.toml`

### 安装

使用 uv 管理依赖（推荐）：

```bash
# 克隆项目
git clone https://github.com/xcz0/Language-Models.git
cd Language-Models

# 使用 uv 安装依赖
uv sync
```

或使用 pip：

```bash
pip install -e .
```

### 数据集

项目使用人名数据集（`data/names.txt`），包含约 32,000 个英文人名，用于字符级语言建模任务。

## 🏃 运行示例

### 训练模型

训练 Bigram 模型：
```bash
uv run train.py --config configs/bigram.yaml
```

训练 MLP 模型：
```bash
uv run train.py --config configs/mlp.yaml
```

训练 RNN 模型：
```bash
uv run train.py --config configs/rnn.yaml
```

训练 GPT 模型：
```bash
uv run train.py --config configs/gpt.yaml
```

### 生成文本

训练完成后，使用模型生成新的人名：

```bash
uv run sample.py --checkpoint out/bigram/logs/checkpoints/xxx.ckpt --num_samples 10
```

## 📁 项目结构

```
Language-Models/
├── configs/               # 配置文件目录
│   ├── base_config.yaml  # 基础配置
│   ├── bigram.yaml       # Bigram 模型配置
│   ├── bow.yaml          # BoW 模型配置
│   ├── mlp.yaml          # MLP 模型配置
│   ├── rnn.yaml          # RNN 模型配置
│   └── gpt.yaml          # GPT 模型配置
├── data/
│   └── names.txt         # 人名数据集
├── src/
│   ├── data/             # 数据处理模块
│   │   ├── datamodule.py # PyTorch Lightning 数据模块
│   │   └── dataset.py    # 数据集实现
│   └── models/           # 模型实现
│       ├── base.py       # 基础模型类
│       ├── bigram.py     # Bigram 模型
│       ├── bow.py        # BoW 模型
│       ├── mlp.py        # MLP 模型
│       ├── rnn.py        # RNN 模型
│       ├── GPT.py        # GPT 模型
│       └── layers/       # 自定义层实现
├── train.py              # 训练脚本
├── sample.py             # 文本生成脚本
└── pyproject.toml        # 项目配置
```

## ⚙️ 配置说明

### 基础配置（base_config.yaml）

所有模型共享的基础配置包括：

- **system**: 系统配置（工作目录、随机种子）
- **data**: 数据配置（数据路径、批量大小等）
- **training**: 训练配置（训练步数、验证间隔等）
- **optimizer**: 优化器配置（学习率、权重衰减等）

### 模型特定配置

每个模型都有对应的配置文件，继承 `base_config.yaml` 并覆盖特定参数：

- `bigram.yaml`: Bigram 模型配置，训练步数较少（1000 步）
- `mlp.yaml`: MLP 模型配置，包含隐藏层维度等参数
- `rnn.yaml`: RNN 模型配置，包含隐藏状态维度、层数等
- `gpt.yaml`: GPT 模型配置，包含注意力头数、层数等 Transformer 参数

## 🎯 模型特点

### Bigram 模型
- 最简单的统计语言模型
- 仅基于前一个字符预测下一个字符
- 训练快速，适合理解基础概念

### MLP 模型
- 固定窗口大小的前馈神经网络
- 可配置隐藏层维度和层数
- 比 Bigram 更强的表达能力

### RNN 模型
- 循环神经网络，理论上可以处理任意长度序列
- 支持多层 RNN
- 适合理解序列建模的基础

### GPT 模型
- 基于 Transformer 架构的生成式模型
- 使用自注意力机制
- 当前最先进的语言建模方法之一

## 📊 训练监控

项目使用 TensorBoard 进行训练监控：

```bash
# 启动 TensorBoard
tensorboard --logdir out/
```

可以观察：
- 训练和验证损失曲线
- 学习率变化
- 模型参数分布

## 🔧 开发

### 添加新模型

1. 在 `src/models/` 下创建新的模型文件
2. 继承 `LitBaseModel` 基类
3. 在 `src/models/__init__.py` 中注册新模型
4. 创建对应的配置文件

### 自定义数据集

修改 `src/data/dataset.py` 和 `src/data/datamodule.py` 以支持新的数据格式。

## 📈 实验结果

不同模型在人名生成任务上的表现：

- **Bigram**: 简单统计模型，生成质量较低但训练极快
- **MLP**: 比 Bigram 好，但受限于固定窗口大小
- **RNN**: 可以捕获更长的依赖关系，生成质量较好
- **GPT**: 表现最佳，生成的人名最接近真实分布

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 高级用法

### 自定义生成参数

使用 `sample.py` 时可以调整生成参数：

```bash
# 生成更多样本，调整温度和 top-k
uv run sample.py \
    --checkpoint out/gpt/logs/checkpoints/best.ckpt \
    --num_samples 20 \
    --max_new_tokens 50 \
    --temperature 0.8 \
    --top_k 10
```

### 混合精度训练

在配置文件中启用混合精度训练以提高性能：

```yaml
# 在模型配置文件中添加
training:
  trainer_args:
    precision: "16-mixed"  # 使用 FP16 混合精度
```

### 多 GPU 训练

```yaml
training:
  trainer_args:
    accelerator: "gpu"
    devices: 2  # 使用 2 个 GPU
    strategy: "ddp"  # 分布式数据并行
```

## ❓ 常见问题

### Q: 如何修改训练数据？

A: 将新的文本数据保存为 `.txt` 文件，然后在配置文件中修改 `data.input_file` 路径。

### Q: 训练过程中出现内存不足怎么办？

A: 可以尝试：
- 减小 `batch_size`
- 使用梯度累积：在配置中添加 `accumulate_grad_batches`
- 启用混合精度训练

### Q: 如何恢复中断的训练？

A: 使用 PyTorch Lightning 的自动检查点功能：

```bash
uv run train.py --config configs/gpt.yaml --ckpt_path out/gpt/logs/checkpoints/last.ckpt
```

### Q: 模型收敛很慢怎么办？

A: 可以尝试：
- 调整学习率（增大或使用学习率调度）
- 检查数据预处理是否正确
- 增加模型复杂度
- 检查损失函数是否合适

## 致谢

- PyTorch Lightning 提供的优秀深度学习框架
- Andrej Karpathy 的语言模型教程给予的启发