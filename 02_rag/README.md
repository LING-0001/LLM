# 阶段2：RAG（检索增强生成）学习路线

> 从零开始，手把手学习RAG技术

---

## 🎯 学习目标

完成本阶段后，你将能够：
- ✅ 理解RAG的工作原理
- ✅ 实现文本切块和向量化
- ✅ 使用向量数据库存储和检索
- ✅ 构建完整的文档问答系统

---

## 📚 什么是RAG？

### 问题场景

假设你有一个公司内部的技术文档库（1000页PDF），用户问：
> "我们公司的请假流程是什么？"

**传统LLM的问题：**
- ❌ LLM没见过你的公司文档
- ❌ 会胡编乱造（幻觉问题）
- ❌ 无法引用具体来源

**RAG的解决方案：**
1. 把所有文档向量化，存入数据库
2. 用户提问时，先搜索相关文档
3. 把搜到的文档和问题一起给LLM
4. LLM基于**真实文档**回答问题

---

## 🔄 RAG工作流程

```
准备阶段（一次性）：
┌─────────────┐
│  你的文档   │ (PDF/TXT/Markdown...)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  文本切块   │ 切成小段落（Chunk）
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  向量化     │ 转换为数字向量
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 向量数据库  │ 存储所有向量
└─────────────┘


查询阶段（每次提问）：
┌─────────────┐
│  用户提问   │ "请假流程是什么？"
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  向量化     │ 把问题也转换为向量
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  检索       │ 找到最相关的3-5个文档片段
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  组合提示词 │ 问题 + 相关文档
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LLM生成    │ 基于文档回答问题
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  返回答案   │ + 引用来源
└─────────────┘
```

---

## 📖 学习步骤

### Step 1: 理解RAG原理 (1天)
📁 `step1_understanding/`

**内容：**
- RAG是什么？为什么需要它？
- RAG vs 普通LLM对比实验
- RAG的应用场景

**文件：**
- `01_what_is_rag.py` - 直观理解RAG
- `02_rag_vs_normal.py` - 对比实验
- `README.md` - 详细说明

---

### Step 2: 文本切块（Chunking） (1天)
📁 `step2_chunking/`

**内容：**
- 为什么要切块？
- 固定长度切块
- 智能切块（按段落、句子）
- 重叠切块（Overlap）

**文件：**
- `01_simple_chunking.py` - 基础切块
- `02_smart_chunking.py` - 智能切块
- `03_overlap_chunking.py` - 重叠策略
- `README.md` - 切块最佳实践

---

### Step 3: 向量化（Embedding） (1-2天)
📁 `step3_embedding/`

**内容：**
- 什么是向量（Vector）？
- 文本 → 向量的转换
- 计算文本相似度
- 使用中文Embedding模型

**文件：**
- `01_what_is_embedding.py` - 向量基础
- `02_similarity.py` - 相似度计算
- `03_chinese_embedding.py` - 中文模型
- `README.md` - Embedding详解

---

### Step 4: 向量数据库 (1-2天)
📁 `step4_vectorstore/`

**内容：**
- 什么是向量数据库？
- ChromaDB 入门
- 存储和检索
- 高级检索策略

**文件：**
- `01_chromadb_basics.py` - ChromaDB基础
- `02_store_and_search.py` - 存储检索
- `03_advanced_search.py` - 高级搜索
- `README.md` - 数据库使用指南

---

### Step 5: 检索与生成 (2天)
📁 `step5_retrieval/`

**内容：**
- 整合检索 + 生成
- 提示词工程（带上下文）
- 多轮对话处理
- 引用来源标注

**文件：**
- `01_simple_rag.py` - 简单RAG系统
- `02_with_context.py` - 上下文管理
- `03_cite_sources.py` - 引用来源
- `README.md` - RAG最佳实践

---

### Final Project: 文档问答系统 (2-3天)
📁 `final_project/`

**目标：构建一个完整的文档问答系统**

**功能：**
- ✅ 上传PDF/TXT文档
- ✅ 自动切块和向量化
- ✅ 智能问答
- ✅ 显示引用来源
- ✅ 多轮对话
- ✅ （可选）Web界面

**文件：**
- `doc_qa_system.py` - 完整系统
- `web_interface.py` - Web界面（可选）
- `README.md` - 使用说明

---

## 🚀 快速开始

### 1. 进入Step 1
```bash
cd ~/code/MyLLM/02_rag/step1_understanding
```

### 2. 阅读说明
```bash
cat README.md
```

### 3. 运行第一个示例
```bash
python 01_what_is_rag.py
```

---

## 📦 需要安装的依赖

```bash
# 激活环境
llm

# 安装RAG相关库
pip install chromadb sentence-transformers pypdf langchain-text-splitters
```

**包说明：**
- `chromadb` - 向量数据库
- `sentence-transformers` - 文本向量化
- `pypdf` - PDF文件处理
- `langchain-text-splitters` - 智能文本切块

---

## 💡 学习建议

1. **每天1-2小时**，不要着急
2. **理解每个概念**后再进入下一步
3. **所有代码都要自己运行**
4. **尝试修改参数**，观察效果变化
5. **用自己的文档测试**最终项目

---

## 🎓 预期成果

完成后，你将拥有：
- ✅ 扎实的RAG理论基础
- ✅ 完整的文档问答系统
- ✅ 可以应用到实际项目的能力
- ✅ 为学习Fine-tuning打下基础

---

## 📍 当前位置

**你现在在这里：** Step 1 - 理解RAG原理

**下一步：**
```bash
cd step1_understanding
cat README.md
```

**开始学习吧！** 🚀

