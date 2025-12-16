# Step 5: 检索与生成（RAG完整流程）

> 整合所有技术，构建端到端的RAG系统

---

## 🎯 本节目标

- 理解RAG的完整工作流程
- 实现检索（Retrieval）逻辑
- 整合LLM生成（Generation）
- 构建完整的问答系统
- 优化检索和生成效果

---

## 🔄 RAG完整流程图

```
用户提问
    ↓
【1. 向量化问题】
    Query → Embedding Model → Query Vector
    ↓
【2. 检索相关文档】
    Query Vector → ChromaDB → Top-K相似文档
    ↓
【3. 构建上下文】
    相关文档 + 问题 → 组装Prompt
    ↓
【4. LLM生成答案】
    Prompt → LLM → 答案
    ↓
返回给用户
```

---

## 📋 之前学习的技术

你已经掌握了RAG的所有组成部分：

| Step | 技术 | 对应RAG环节 |
|------|------|------------|
| Step 1 | RAG概念 | 理解原理 |
| Step 2 | 文本分块 | **离线准备** |
| Step 3 | 向量化 | **离线准备** + **在线检索** |
| Step 4 | 向量数据库 | **在线检索** |
| Step 5 | 整合LLM | **在线生成** |

---

## 🎨 RAG系统架构

### 离线阶段（只需运行一次）

```python
# 1. 准备文档
document = load_document("traffic_law.md")

# 2. 分块
chunks = text_splitter.split(document)

# 3. 向量化
vectors = embedding_model.encode(chunks)

# 4. 存入数据库
vectorstore.add(chunks, vectors)
```

**✅ 你已经完成！** 数据在 `data/chroma_traffic_law/`

---

### 在线阶段（每次问答都会执行）

```python
# 1. 用户提问
question = "酒驾会受到什么处罚？"

# 2. 向量化问题
question_vector = embedding_model.encode(question)

# 3. 检索相关文档
results = vectorstore.query(question_vector, top_k=3)

# 4. 构建Prompt
context = "\n".join(results['documents'])
prompt = f"""
根据以下内容回答问题：

{context}

问题：{question}
答案：
"""

# 5. LLM生成
answer = llm.generate(prompt)

# 6. 返回答案
return answer
```

---

## 🔍 核心概念：检索策略

### 1. Top-K选择

**问题：应该检索多少个文档？**

| Top-K | 优点 | 缺点 |
|-------|------|------|
| K=1 | 速度快，上下文短 | 可能遗漏信息 |
| K=3 | 平衡 ⭐ | 适合大多数场景 |
| K=5-10 | 信息全面 | 上下文长，可能引入噪音 |

**经验值：K=3-5** 对大多数应用效果最好。

---

### 2. 相似度阈值

**问题：检索到的文档相似度太低怎么办？**

```python
# 设置阈值
threshold = 0.7  # 70%相似度

# 过滤低质量结果
filtered_results = [
    doc for doc, score in results
    if score > threshold
]

# 如果没有高质量结果
if len(filtered_results) == 0:
    return "抱歉，我在文档中没有找到相关信息。"
```

---

### 3. 重排序（Reranking）

**问题：向量检索不够准确？**

```python
# 第一步：向量检索（快速，召回Top-20）
candidates = vectorstore.query(question, top_k=20)

# 第二步：精排（慢速，精确计算相似度）
reranked = reranker.rank(question, candidates)

# 第三步：取Top-3
final_docs = reranked[:3]
```

**本教程不涉及重排序（高级技巧），但你应该知道有这个方法。**

---

## 💡 核心概念：Prompt工程

### 1. 基础Prompt模板

```python
prompt = f"""
根据以下参考内容回答问题：

{context}

问题：{question}
答案：
"""
```

**问题：LLM可能不遵守指令，胡乱回答。**

---

### 2. 改进版Prompt（推荐）

```python
prompt = f"""
你是一个交通法规专家助手。请根据以下参考资料回答用户的问题。

【参考资料】
{context}

【要求】
1. 只根据参考资料回答，不要编造信息
2. 如果参考资料中没有答案，请明确说明
3. 回答要简洁准确，突出重点

【问题】
{question}

【回答】
"""
```

**改进点：**
- ✅ 角色定位（专家助手）
- ✅ 明确指令（只用参考资料）
- ✅ 格式清晰（分块标注）
- ✅ 约束回答（不编造）

---

### 3. 带引用的Prompt（最佳实践）

```python
prompt = f"""
你是一个交通法规助手。请根据参考资料回答问题，并标注引用来源。

【参考资料】
[文档1] {doc1}
[文档2] {doc2}
[文档3] {doc3}

【回答要求】
1. 只使用参考资料中的信息
2. 在回答中用[文档X]标注来源
3. 如果没有相关信息，说"参考资料中未提及"

问题：{question}
"""
```

---

## ⚙️ 核心概念：生成参数

### LLM生成参数详解

```python
answer = llm.generate(
    prompt=prompt,
    max_tokens=512,      # 最大生成长度
    temperature=0.3,     # 随机性（0=确定，1=随机）
    top_p=0.9,           # 累积概率采样
    stop=["\n\n", "问题："]  # 停止标记
)
```

---

### Temperature对RAG的影响

| Temperature | 效果 | 适用场景 |
|-------------|------|---------|
| 0.0-0.3 | 确定性强，严格依赖上下文 | **RAG首选** ⭐ |
| 0.4-0.7 | 平衡，稍有创造性 | 需要总结概括时 |
| 0.8-1.0 | 创造性强，可能编造 | ❌ 不适合RAG |

**RAG最佳实践：temperature=0.1-0.3**，确保答案来自文档，不瞎编。

---

## 🛠️ 实践练习

### 练习1：基础RAG问答
```bash
python 01_basic_rag.py
```

**内容：**
- 加载向量数据库
- 实现检索逻辑
- 整合LLM生成
- 回答交通法问题

---

### 练习2：优化检索效果
```bash
python 02_improve_retrieval.py
```

**内容：**
- 调整Top-K
- 相似度阈值过滤
- 元数据过滤
- 结果去重

---

### 练习3：优化Prompt
```bash
python 03_improve_prompt.py
```

**内容：**
- 对比不同Prompt效果
- 添加角色和指令
- 防止幻觉（编造信息）
- 带引用的回答

---

### 练习4：完整RAG系统
```bash
python 04_complete_rag_system.py
```

**内容：**
- 整合所有优化
- 异常处理
- 日志记录
- 可复用的RAG类

---

### 练习5：交互式问答
```bash
python 05_interactive_qa.py
```

**内容：**
- 命令行交互界面
- 连续对话
- 显示检索到的文档
- 保存对话历史

---

## 📊 评估RAG效果

### 1. 定性评估（主观）

**问5-10个测试问题，检查：**
- ✅ 答案是否准确？
- ✅ 是否来自文档？
- ✅ 是否简洁明了？
- ❌ 是否有幻觉（编造）？

---

### 2. 定量评估（客观）

```python
# 准备测试集
test_cases = [
    {
        "question": "酒驾的处罚是什么？",
        "expected_keywords": ["暂扣", "罚款", "吊销", "六个月"]
    },
    # ... 更多测试
]

# 运行测试
for case in test_cases:
    answer = rag_system.query(case['question'])
    
    # 检查关键词是否在答案中
    score = sum(
        kw in answer 
        for kw in case['expected_keywords']
    ) / len(case['expected_keywords'])
    
    print(f"问题: {case['question']}")
    print(f"得分: {score:.0%}")
```

---

### 3. 检索质量评估

```python
# 检索评估指标
def evaluate_retrieval(question, expected_doc_ids, retrieved_doc_ids):
    # 精确率：检索到的文档中有多少是相关的
    precision = len(set(retrieved_doc_ids) & set(expected_doc_ids)) / len(retrieved_doc_ids)
    
    # 召回率：相关文档中有多少被检索到
    recall = len(set(retrieved_doc_ids) & set(expected_doc_ids)) / len(expected_doc_ids)
    
    # F1分数：精确率和召回率的调和平均
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1
```

---

## 🚀 常见问题和解决方案

### Q1: RAG回答不准确？

**可能原因：**
1. 检索不到相关文档 → 检查分块策略
2. 检索到但不相关 → 调整Top-K和阈值
3. LLM不遵守指令 → 优化Prompt
4. Temperature太高 → 降低到0.1-0.3

---

### Q2: RAG回答太长/太短？

**解决方案：**
```python
# 在Prompt中明确要求
prompt = f"""
...
【回答要求】
- 控制在100字以内
- 分3点回答，每点不超过30字
...
"""
```

---

### Q3: RAG产生幻觉（编造信息）？

**解决方案：**
1. **降低Temperature** → 0.1-0.3
2. **强化Prompt指令**：
   ```python
   "如果参考资料中没有答案，请回答：'参考资料中未提及此内容。'"
   ```
3. **添加验证**：检查答案中的关键词是否在检索文档中

---

### Q4: 检索速度太慢？

**优化方案：**
1. 减少Top-K（5→3）
2. 使用批量检索
3. 缓存常见问题的结果
4. 考虑更快的向量数据库（Milvus）

---

### Q5: 如何处理多跳问题？

**示例：** "酒驾处罚和闯红灯处罚哪个更严重？"

**方法1：拆解问题**
```python
# 第一次检索：酒驾处罚
docs1 = retrieve("酒驾处罚")

# 第二次检索：闯红灯处罚
docs2 = retrieve("闯红灯处罚")

# 合并上下文
context = docs1 + docs2
```

**方法2：让LLM自己拆解（高级）**
- 需要更强的LLM
- 本教程不涉及

---

## 💡 最佳实践总结

### ✅ 检索阶段

1. **Top-K = 3-5**（平衡速度和效果）
2. **相似度阈值 > 0.7**（过滤低质量）
3. **使用元数据过滤**（缩小范围）
4. **去重**（避免重复内容）

---

### ✅ Prompt阶段

1. **角色定位**（你是XX助手）
2. **明确指令**（只用参考资料）
3. **格式清晰**（分块标注）
4. **约束回答**（不编造，简洁）

---

### ✅ 生成阶段

1. **Temperature = 0.1-0.3**（确定性）
2. **max_tokens = 256-512**（控制长度）
3. **添加stop标记**（避免冗余）
4. **流式输出**（提升体验）

---

## 📈 RAG进阶方向

完成本节后，你可以探索：

1. **多模态RAG**
   - 图片 + 文本检索
   - PDF表格提取

2. **对话式RAG**
   - 多轮对话
   - 上下文记忆

3. **混合检索**
   - 向量检索 + 关键词检索
   - BM25 + Embedding

4. **Agent RAG**
   - 自动拆解问题
   - 多次检索

**但这些都是后话，先掌握基础！**

---

## ✅ 完成标志

掌握了以下内容，即可完成RAG阶段：

- [ ] 理解RAG完整流程
- [ ] 实现基础检索逻辑
- [ ] 整合LLM生成答案
- [ ] 优化检索和Prompt
- [ ] 构建完整RAG系统
- [ ] 运行交互式问答

---

## 📍 下一步

**完成RAG阶段后：**

进入 **Stage 3: Fine-tuning（微调）**

```bash
cd ../../03_finetuning
cat README.md
```

学习如何训练专属的交通法LLM！

---

**开始实践吧！** 🚀

```bash
cd ~/code/MyLLM/02_rag/step5_retrieval
python 01_basic_rag.py
```

---

**恭喜你即将完成RAG学习！这是最激动人心的部分！** 🎉

