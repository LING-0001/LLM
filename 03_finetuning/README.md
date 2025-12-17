# Phase 3: Fine-tuning (模型微调)

## 🎯 学习目标

通过微调，让模型学习你的领域知识和特定风格，而不仅仅是"查询"知识库。

## 📋 Fine-tuning vs RAG

| 维度 | RAG | Fine-tuning | 最佳实践 |
|------|-----|-------------|----------|
| **知识来源** | 外部文档 | 模型内部 | 两者结合 |
| **响应速度** | 慢（需检索） | 快（直接生成） | - |
| **知识更新** | 实时 | 需重训 | RAG更新快 |
| **专业理解** | 浅层 | 深层 | Fine-tuning更深 |
| **成本** | 低 | 高（GPU） | - |

## 🗂️ 学习路径

```
03_finetuning/
├── README.md                       # 本文件
├── step1_basics/                   # 基础概念
│   ├── README.md
│   ├── 01_what_is_finetuning.py   # 什么是微调
│   └── 02_prepare_environment.py  # 环境准备
├── step2_data_preparation/         # 数据准备
│   ├── README.md
│   ├── 01_data_format.py          # 数据格式
│   ├── 02_create_dataset.py       # 创建数据集
│   └── 03_data_quality.py         # 数据质量
├── step3_lora_finetuning/         # LoRA微调
│   ├── README.md
│   ├── 01_what_is_lora.py         # LoRA原理
│   ├── 02_simple_finetune.py      # 简单微调
│   └── 03_advanced_finetune.py    # 高级微调
├── step4_evaluation/              # 效果评估
│   ├── README.md
│   ├── 01_compare_models.py       # 对比模型
│   └── 02_performance_test.py     # 性能测试
└── step5_deployment/              # 部署使用
    ├── README.md
    ├── 01_load_finetuned.py       # 加载微调模型
    └── 02_rag_plus_finetune.py    # RAG+微调结合
```

## 🚀 快速开始

### Step 1: 理解微调 (5分钟)
```bash
cd 03_finetuning/step1_basics
python 01_what_is_finetuning.py
```

### Step 2: 准备数据 (10分钟)
```bash
cd ../step2_data_preparation
python 01_data_format.py
python 02_create_dataset.py
```

### Step 3: 开始微调 (30分钟)
```bash
cd ../step3_lora_finetuning
python 02_simple_finetune.py
```

### Step 4: 评估效果 (5分钟)
```bash
cd ../step4_evaluation
python 01_compare_models.py
```

## 💡 核心概念

### 1. 什么是Fine-tuning？

```
预训练模型 (通用知识)
    ↓
+ 你的数据 (领域知识)
    ↓
微调后的模型 (专业模型)
```

### 2. 什么是LoRA？

**LoRA** (Low-Rank Adaptation) = 低成本微调方法

```
传统微调: 更新所有参数 (3B参数 → 需要大量GPU)
LoRA微调: 只训练小部分参数 (可能只有1M参数 → 普通电脑可行)
```

### 3. 微调场景

| 场景 | 是否需要微调 | 原因 |
|------|-------------|------|
| 回答交通法问题 | ❌ 不需要 | RAG足够 |
| 生成特定格式法律文书 | ✅ 需要 | 需要学习格式和风格 |
| 使用专业术语和表达 | ✅ 需要 | 需要深度理解 |
| 多轮对话的记忆和理解 | ✅ 需要 | 需要对话能力 |

## 🎓 预备知识

在开始前，你应该已经掌握：
- ✅ LLM基本使用 (Phase 1)
- ✅ RAG完整流程 (Phase 2)
- ✅ Python编程
- ✅ 基本的机器学习概念

## ⚠️ 注意事项

### 硬件要求
- **最低**: 16GB RAM + Apple M1/M2 (MPS)
- **推荐**: 24GB RAM + GPU
- **我们的环境**: M1芯片，纯CPU微调（会比较慢）

### 时间预期
- 准备数据: 10-30分钟
- 微调训练: 30分钟-2小时（取决于数据量和硬件）
- 评估测试: 10分钟

### 成本
- 使用本地模型: 免费（只需电费）
- 云端GPU: $0.5-2/小时

## 📚 学习资源

- **官方文档**: https://huggingface.co/docs/peft
- **LoRA论文**: https://arxiv.org/abs/2106.09685
- **实战教程**: 我们的step-by-step指南

## 🎯 学习成果

完成本阶段后，你将能够：
1. ✅ 理解微调的原理和应用场景
2. ✅ 准备高质量的训练数据
3. ✅ 使用LoRA微调小型语言模型
4. ✅ 评估和对比微调效果
5. ✅ 部署和使用微调后的模型
6. ✅ 结合RAG和Fine-tuning构建完整系统

---

**准备好了吗？开始第一步！** 🚀

```bash
cd step1_basics
python 01_what_is_finetuning.py
```

