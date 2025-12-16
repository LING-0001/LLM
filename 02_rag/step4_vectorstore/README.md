# Step 4: 向量数据库（ChromaDB）

> 专业的向量存储和检索方案

---

## 🎯 本节目标

- 理解为什么需要向量数据库
- 学会使用ChromaDB
- 掌握向量的增删改查
- 优化检索性能

---

## 🤔 为什么需要向量数据库？

### 场景对比

**用numpy数组存储（Step 3的做法）：**

```python
# 10万个向量
vectors = np.load('vectors.npy')  # 形状：(100000, 768)

# 查询：找最相似的5个
query_vec = model.encode("问题")
similarities = cosine_similarity([query_vec], vectors)[0]
top_5 = similarities.argsort()[-5:][::-1]

# 问题：
# ❌ 需要计算10万次相似度（慢！）
# ❌ 内存占用大（300MB+）
# ❌ 没有元数据管理
# ❌ 不支持增量更新
```

---

**用向量数据库（ChromaDB）：**

```python
# 存储
collection.add(
    embeddings=vectors,
    documents=texts,
    metadatas=metadata,
    ids=ids
)

# 查询：自动优化，秒级返回
results = collection.query(
    query_embeddings=[query_vec],
    n_results=5
)

# 优势：
# ✅ 使用ANN算法（近似最近邻），速度快
# ✅ 自动管理元数据
# ✅ 支持增删改查
# ✅ 持久化存储
```

---

## 📊 性能对比

| 向量数量 | NumPy数组 | ChromaDB |
|---------|----------|----------|
| 1,000   | 10ms     | 5ms      |
| 10,000  | 100ms    | 8ms      |
| 100,000 | 1000ms   | 15ms     |
| 1,000,000 | 10s+   | 50ms     |

**结论：数据量越大，向量数据库优势越明显！**

---

## 🔍 什么是ANN算法？

### 精确搜索 vs 近似搜索

**精确搜索（Exact Search）：**
```
计算问题向量与所有向量的相似度
→ 找到真正最相似的Top-K
→ 100%准确，但慢
```

**近似搜索（ANN - Approximate Nearest Neighbor）：**
```
使用聪明的索引结构（如HNSW）
→ 只计算部分向量
→ 找到"几乎最相似"的Top-K
→ 95%+准确，但超快！
```

**类比：**
- 精确搜索 = 问遍全班同学找最高的5个人
- 近似搜索 = 先按身高分组，只问最高的几组

---

## 🛠️ ChromaDB简介

### 为什么选择ChromaDB？

✅ **轻量级**：单文件即可运行，无需安装服务器  
✅ **易用**：Python API简单直观  
✅ **功能全**：支持过滤、元数据查询  
✅ **本地优先**：数据存在本地，隐私安全  
✅ **适合学习**：完美的RAG学习工具

### 核心概念

```python
Client          # 客户端（连接数据库）
  └── Collection  # 集合（类似MySQL的表）
        ├── Embeddings  # 向量
        ├── Documents   # 原始文本
        ├── Metadatas   # 元数据
        └── IDs         # 唯一标识
```

---

## 📚 ChromaDB基本操作

### 1. 创建/连接数据库

```python
import chromadb

# 方式1：内存模式（重启丢失）
client = chromadb.Client()

# 方式2：持久化模式（推荐）
client = chromadb.PersistentClient(path="./chroma_db")
```

---

### 2. 创建/获取集合

```python
# 创建新集合
collection = client.create_collection(
    name="my_documents",
    metadata={"description": "我的文档库"}
)

# 获取已存在的集合
collection = client.get_collection(name="my_documents")

# 获取或创建
collection = client.get_or_create_collection(name="my_documents")
```

---

### 3. 添加向量

```python
collection.add(
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],  # 向量列表
    documents=["文本1", "文本2"],                    # 原文
    metadatas=[{"source": "doc1"}, {"source": "doc2"}],  # 元数据
    ids=["id1", "id2"]                              # ID
)
```

---

### 4. 查询（最重要！）

```python
results = collection.query(
    query_embeddings=[[0.15, 0.25, ...]],  # 问题向量
    n_results=5,                           # 返回Top-5
    where={"source": "doc1"},              # 元数据过滤（可选）
    include=["documents", "metadatas", "distances"]  # 返回内容
)
```

**返回结果：**
```python
{
    'ids': [['id2', 'id1', ...]],
    'documents': [['文本2', '文本1', ...]],
    'metadatas': [[{'source': 'doc2'}, ...]],
    'distances': [[0.23, 0.45, ...]]  # 距离（越小越相似）
}
```

---

### 5. 更新和删除

```python
# 更新
collection.update(
    ids=["id1"],
    documents=["新文本"],
    metadatas=[{"updated": True}]
)

# 删除
collection.delete(ids=["id1"])

# 删除所有
collection.delete(where={"source": "doc1"})
```

---

## 🎨 高级功能

### 1. 元数据过滤

```python
# 只搜索特定来源的文档
results = collection.query(
    query_embeddings=[query_vec],
    n_results=5,
    where={"chapter": "第五章"}  # 只在第五章中搜索
)

# 复杂条件
results = collection.query(
    query_embeddings=[query_vec],
    where={
        "$and": [
            {"chapter": "第五章"},
            {"length": {"$gt": 100}}  # 长度>100
        ]
    }
)
```

**支持的操作符：**
- `$eq`, `$ne` - 等于/不等于
- `$gt`, `$gte`, `$lt`, `$lte` - 大于/小于
- `$in`, `$nin` - 在列表中/不在
- `$and`, `$or` - 逻辑与/或

---

### 2. 距离度量

```python
collection = client.create_collection(
    name="my_docs",
    metadata={
        "hnsw:space": "cosine"  # 余弦距离（默认）
        # "hnsw:space": "l2"    # 欧氏距离
        # "hnsw:space": "ip"    # 内积
    }
)
```

---

### 3. 批量操作

```python
# 批量添加（高效）
collection.add(
    embeddings=vectors_list,  # 1000个向量
    documents=texts_list,
    ids=ids_list
)

# 分批添加（避免内存溢出）
batch_size = 100
for i in range(0, len(vectors), batch_size):
    batch_vectors = vectors[i:i+batch_size]
    batch_texts = texts[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]
    
    collection.add(
        embeddings=batch_vectors,
        documents=batch_texts,
        ids=batch_ids
    )
```

---

## 🚀 实践练习

### 练习1：ChromaDB基础操作
```bash
python 01_chromadb_basics.py
```

**内容：**
- 创建和连接数据库
- 增删改查操作
- 查看集合信息

---

### 练习2：导入交通法数据
```bash
python 02_import_traffic_law.py
```

**内容：**
- 读取准备好的数据
- 导入到ChromaDB
- 测试检索效果

---

### 练习3：高级检索
```bash
python 03_advanced_query.py
```

**内容：**
- 元数据过滤
- 多条件查询
- 结果排序和分析

---

### 练习4：性能优化
```bash
python 04_performance.py
```

**内容：**
- 批量导入优化
- 检索速度测试
- 内存管理

---

## 💡 最佳实践

### 1. ID命名规范

```python
# ✅ 好的做法
ids = [f"doc_{i:05d}" for i in range(100)]
# ['doc_00000', 'doc_00001', ...]

# ❌ 避免
ids = ["1", "2", "3"]  # 太简单
ids = ["随机字符串"]    # 难以管理
```

---

### 2. 元数据设计

```python
# ✅ 结构化元数据
metadata = {
    "source": "traffic_law_document.md",
    "chapter": "第五章",
    "chunk_id": 15,
    "length": 380,
    "created_at": "2024-01-01"
}

# ❌ 避免
metadata = {"info": "一些信息"}  # 太模糊
```

---

### 3. 持久化路径

```python
# ✅ 使用项目目录
client = chromadb.PersistentClient(
    path="./data/chroma_db"
)

# ❌ 避免
client = chromadb.PersistentClient(path="/tmp/chroma")
# 临时目录可能被清理
```

---

### 4. 集合管理

```python
# 列出所有集合
collections = client.list_collections()

# 删除集合
client.delete_collection(name="old_collection")

# 重命名（先复制再删除）
# ChromaDB不支持直接重命名
```

---

## 🔧 常见问题

### Q1: ChromaDB存储在哪里？

```python
client = chromadb.PersistentClient(path="./chroma_db")
# 数据存储在：./chroma_db/ 目录
# 可以直接删除目录来清空数据
```

---

### Q2: 如何备份数据？

```bash
# 方法1：复制目录
cp -r ./chroma_db ./chroma_db_backup

# 方法2：导出向量
vectors = collection.get(include=["embeddings"])
np.save("backup.npy", vectors)
```

---

### Q3: 多大数据适合ChromaDB？

- ✅ **1千-100万向量**：完美
- ⚠️ **100万-1000万**：可以，但可能慢
- ❌ **1000万+**：考虑专业方案（Milvus, Qdrant）

---

### Q4: 距离 vs 相似度？

```python
# ChromaDB返回的是距离（distance）
distance = 0.2   # 越小越相似

# 转换为相似度
similarity = 1 - distance  # cosine距离
# 或
similarity = 1 / (1 + distance)  # 通用转换
```

---

## ✅ 完成标志

掌握了以下内容，即可进入Step 5：

- [ ] 理解向量数据库的作用
- [ ] 会使用ChromaDB的基本操作
- [ ] 成功导入交通法数据
- [ ] 能够进行高级查询
- [ ] 运行了所有4个练习

---

## 📍 下一步

**Step 5: 检索与生成（RAG完整流程）**

```bash
cd ../step5_retrieval
cat README.md
```

整合检索和生成，构建完整的RAG系统！

---

**开始实践吧！** 🚀

```bash
cd ~/code/MyLLM/02_rag/step4_vectorstore
python 01_chromadb_basics.py
```

