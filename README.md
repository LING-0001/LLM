# LLM学习路线 - 从零到精通

> 零基础手把手学习大语言模型：RAG 和 Fine-tuning

---

## 📁 项目结构

```
MyLLM/
├── 00_basics/              # 阶段0：基础准备 ✅
│   ├── test_model.py       # 基本模型测试
│   ├── chatbot.py          # 简单聊天机器人
│   ├── test_parameters.py  # 参数实验
│   ├── prompt_engineering.py
│   └── practical_examples.py
│
├── 01_inference/           # 阶段1：推理方式对比 ✅
│   ├── test_llamacpp.py    # llama.cpp方式
│   ├── test_ollama.py      # Ollama方式
│   └── test_transformers.py # HuggingFace方式
│
├── 02_rag/                 # 阶段2：RAG（检索增强生成）⬅️ 当前
│   ├── step1_understanding/  # 第1步：理解RAG原理
│   ├── step2_chunking/       # 第2步：文本切块
│   ├── step3_embedding/      # 第3步：向量化
│   ├── step4_vectorstore/    # 第4步：向量数据库
│   ├── step5_retrieval/      # 第5步：检索与生成
│   └── final_project/        # 最终项目：文档问答系统
│
├── 03_finetune/            # 阶段3：Fine-tuning（模型微调）
│   ├── step1_understanding/  # 第1步：理解微调原理
│   ├── step2_data_prep/      # 第2步：数据准备
│   ├── step3_training/       # 第3步：训练过程
│   └── final_project/        # 最终项目：领域模型微调
│
├── data/                   # 数据目录
│   ├── documents/          # RAG用的文档
│   └── finetune/           # 微调用的数据集
│
├── models/                 # 模型目录（软链接到 ~/llama.cpp/models）
│
├── START_HERE.md          # 详细教程（从头开始）
└── requirements.txt       # Python依赖
```

---

## 🎯 学习路线

### ✅ 阶段0：基础准备（已完成）
- [x] 环境配置（Conda + Python 3.10）
- [x] 模型下载（Qwen2.5-3B）
- [x] 基本推理测试
- [x] 提示词工程入门

### ✅ 阶段1：推理方式对比（已完成）
- [x] llama.cpp 推理
- [x] Ollama 推理
- [x] HuggingFace Transformers 推理

### 📍 阶段2：RAG - 检索增强生成（当前学习）

#### **什么是RAG？**
RAG = Retrieval Augmented Generation（检索增强生成）

简单说：让AI能够**查询你的文档**后再回答问题。

**核心流程：**
```
你的文档 → 切块 → 向量化 → 存入向量库
      ↓
用户提问 → 向量化 → 搜索相关文档 → 连同文档一起给LLM → 生成答案
```

#### **学习步骤：**

**Step 1: 理解RAG原理** (1天)
- RAG是什么，为什么需要它？
- RAG vs 直接提问的区别
- RAG的应用场景

**Step 2: 文本切块（Chunking）** (1天)
- 为什么要切块？
- 固定长度切块
- 智能切块（按段落、句子）
- 重叠切块技巧

**Step 3: 向量化（Embedding）** (1-2天)
- 什么是向量化？
- 使用中文Embedding模型
- 计算文本相似度
- 实战：找到最相似的文档

**Step 4: 向量数据库** (1-2天)
- 什么是向量数据库？
- ChromaDB 使用
- 存储和检索
- 实战：构建自己的知识库

**Step 5: 检索与生成** (2天)
- 整合检索和生成
- 优化检索策略
- 处理多轮对话
- 实战：完整的RAG系统

**Final Project: 文档问答系统** (2-3天)
- 上传PDF/TXT文档
- 智能问答
- 引用来源
- Web界面（可选）

**预计时间：7-10天**

---

### 阶段3：Fine-tuning - 模型微调（后续）

#### **什么是Fine-tuning？**
在预训练模型基础上，用你自己的数据继续训练，让模型适应特定任务。

#### **学习步骤：**

**Step 1: 理解微调原理** (1天)
- 什么是微调？
- Full Fine-tuning vs LoRA vs QLoRA
- 何时需要微调？

**Step 2: 数据准备** (2-3天)
- 数据格式要求
- 构造训练数据
- 数据清洗和验证
- 实战：准备自己的数据集

**Step 3: 训练过程** (3-5天)
- 使用 Unsloth 或 LLaMA-Factory
- 配置训练参数
- 监控训练过程
- 模型评估

**Final Project: 领域模型微调** (3-5天)
- 选择领域（客服、写作、代码等）
- 准备专业数据集
- 完整微调流程
- 对比微调前后效果

**预计时间：10-15天**

---

## 🚀 快速开始

### 1. 环境激活
```bash
# 激活conda环境（简写命令）
llm

# 或者完整命令
conda activate llm-learning
```

### 2. 当前进度检查
```bash
cd ~/code/MyLLM

# 查看项目结构
tree -L 2

# 测试基础环境
python 00_basics/test_model.py
```

### 3. 开始RAG学习
```bash
# 进入RAG目录
cd 02_rag

# 阅读第一步教程
cat step1_understanding/README.md
```

---

## 📚 学习资源

### 阶段0 & 1：基础
- 文档：`START_HERE.md`（完整的安装和配置指南）
- 代码：`00_basics/` 和 `01_inference/`

### 阶段2：RAG
- 主文档：`02_rag/README.md`（即将创建）
- 每个步骤都有独立的 README 和示例代码
- 从简单到复杂，循序渐进

### 阶段3：Fine-tuning
- 主文档：`03_finetune/README.md`（后续创建）
- 完整的微调教程和最佳实践

---

## 💡 学习建议

1. **按顺序学习**：不要跳步，每个步骤都很重要
2. **动手实践**：每个示例都要自己运行一遍
3. **理解原理**：不要只会调包，要理解为什么这样做
4. **记录笔记**：遇到的问题和解决方案都记下来
5. **提出问题**：不懂的地方随时问

---

## ⚙️ 当前环境

- **Python**: 3.10.19 (conda环境)
- **模型**: Qwen2.5-3B-Instruct (Q4量化, ~2GB)
- **位置**: `~/llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf`
- **推理库**: llama-cpp-python

---

## 📞 下一步

**开始学习RAG！**

请阅读：`02_rag/README.md`

或运行：
```bash
cd ~/code/MyLLM/02_rag/step1_understanding
python 01_what_is_rag.py
```

---

**准备好了吗？让我们开始RAG的学习之旅！** 🚀

