# 🚀 快速开始 - RAG学习

> 5分钟开始你的RAG学习之旅

---

## ✅ 前置检查

确保已完成阶段0和阶段1（基础准备）：

```bash
# 1. 激活环境
llm

# 2. 检查Python版本
python --version  # 应显示 Python 3.10.x

# 3. 测试模型
cd ~/code/MyLLM/00_basics
python test_model.py  # 应该能看到流式输出
```

如果上面都OK，继续下面的步骤 ✨

---

## 📦 安装RAG依赖

```bash
# 确保在项目根目录
cd ~/code/MyLLM

# 激活环境
llm

# 安装RAG相关库（大约1-2分钟）
pip install chromadb sentence-transformers pypdf langchain-text-splitters
```

**依赖说明：**
- `chromadb` - 轻量级向量数据库
- `sentence-transformers` - 文本向量化模型
- `pypdf` - PDF文件处理
- `langchain-text-splitters` - 智能文本切块

---

## 🎯 开始Step 1：理解RAG

### 1. 阅读说明文档
```bash
cd ~/code/MyLLM/02_rag/step1_understanding
cat README.md
```

### 2. 运行第一个实验：普通LLM
```bash
python 01_without_rag.py
```

**观察：** LLM无法准确回答关于本项目的问题

### 3. 运行第二个实验：使用RAG
```bash
python 02_with_rag.py
```

**观察：** 通过提供文档，LLM能准确回答

### 4. 运行对比实验
```bash
python 03_comparison.py
```

**观察：** 并排对比两种方式的差异

---

## 📚 学习路径

```
Step 1: 理解RAG原理          ⬅️ 你在这里
   ↓
Step 2: 文本切块（Chunking）
   ↓
Step 3: 向量化（Embedding）
   ↓
Step 4: 向量数据库
   ↓
Step 5: 检索与生成
   ↓
Final Project: 文档问答系统
```

---

## 💡 每个Step的学习流程

1. **阅读 README.md** - 理解概念和原理
2. **运行示例代码** - 看效果，建立直观认知
3. **修改参数实验** - 加深理解
4. **完成思考题** - 巩固知识
5. **进入下一步** - 循序渐进

---

## ⏱️ 时间安排建议

| 步骤 | 内容 | 建议时间 |
|------|------|----------|
| Step 1 | 理解RAG原理 | 1天 |
| Step 2 | 文本切块 | 1天 |
| Step 3 | 向量化 | 1-2天 |
| Step 4 | 向量数据库 | 1-2天 |
| Step 5 | 检索与生成 | 2天 |
| Final Project | 文档问答系统 | 2-3天 |

**总计：7-10天**（每天1-2小时）

---

## 📞 遇到问题？

### 常见问题

**Q1: 模型加载很慢？**
- A: 第一次加载需要几秒，之后会快很多

**Q2: 生成速度慢？**
- A: 我们使用的是CPU模式，3B模型生成速度约5-10 tokens/秒

**Q3: 想用GPU加速？**
- A: 修改代码中的 `n_gpu_layers=0` 为 `n_gpu_layers=32`

**Q4: 安装依赖失败？**
- A: 确保激活了 llm-learning 环境，检查网络连接

---

## 🎓 学习建议

1. **不要跳过步骤** - 每一步都有其价值
2. **动手实践** - 光看不练等于白学
3. **理解原理** - 知其然更要知其所以然
4. **做好笔记** - 记录疑惑和收获
5. **循序渐进** - 不着急，稳扎稳打

---

## ✅ Step 1 完成标志

完成以下内容，即可进入Step 2：

- [ ] 理解RAG是什么
- [ ] 知道为什么需要RAG
- [ ] 运行了3个对比实验
- [ ] 理解RAG的基本工作流程
- [ ] 能够说出RAG的应用场景

---

## 📍 下一步

**Step 2: 文本切块（Chunking）**

```bash
cd ../step2_chunking
cat README.md
```

---

**开始学习吧！加油！** 🚀

如有问题，随时提问！

