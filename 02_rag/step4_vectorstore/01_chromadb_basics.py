#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4.1: ChromaDB基础操作
学习目标：掌握向量数据库的增删改查
"""

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

print("=" * 60)
print("📚 ChromaDB基础操作教程")
print("=" * 60)

# ============================================================
# 第一部分：创建和连接数据库
# ============================================================

print("\n【第一部分：创建数据库】")
print("-" * 60)

# 方式1：内存模式（关闭程序后数据丢失）
print("\n1️⃣ 创建内存数据库（用于测试）...")
memory_client = chromadb.Client()
print("   ✅ 内存数据库创建成功")

# 方式2：持久化模式（数据保存到硬盘）
print("\n2️⃣ 创建持久化数据库（用于生产）...")
import os
db_path = "./data/chroma_basics_demo"
os.makedirs(db_path, exist_ok=True)

persistent_client = chromadb.PersistentClient(path=db_path)
print(f"   ✅ 持久化数据库创建成功")
print(f"   📁 存储路径: {db_path}")

# 后续使用持久化客户端
client = persistent_client

# ============================================================
# 第二部分：创建集合（Collection）
# ============================================================

print("\n【第二部分：创建集合】")
print("-" * 60)

# 删除旧集合（如果存在）
try:
    client.delete_collection(name="demo_collection")
    print("🗑️  删除旧集合: demo_collection")
except:
    pass

# 创建新集合
collection = client.create_collection(
    name="demo_collection",
    metadata={
        "description": "学习ChromaDB的演示集合",
        "hnsw:space": "cosine"  # 使用余弦距离
    }
)
print("✅ 创建新集合: demo_collection")

# ============================================================
# 第三部分：添加数据（Add）
# ============================================================

print("\n【第三部分：添加数据】")
print("-" * 60)

# 准备数据
documents = [
    "Python是一种高级编程语言",
    "机器学习是人工智能的一个分支",
    "深度学习使用神经网络进行学习",
    "自然语言处理研究计算机如何理解人类语言",
    "RAG技术结合了检索和生成"
]

# 加载embedding模型
print("\n📦 加载embedding模型...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')
print("   ✅ 模型加载完成")

# 生成向量
print("\n🔄 生成向量...")
embeddings = model.encode(documents, show_progress_bar=False)
print(f"   ✅ 生成了 {len(embeddings)} 个向量，维度: {embeddings.shape[1]}")

# 准备元数据
metadatas = [
    {"category": "编程", "difficulty": "入门"},
    {"category": "AI", "difficulty": "中级"},
    {"category": "AI", "difficulty": "高级"},
    {"category": "AI", "difficulty": "中级"},
    {"category": "AI", "difficulty": "高级"}
]

# 准备ID
ids = [f"doc_{i}" for i in range(len(documents))]

# 添加到集合
print("\n💾 添加数据到ChromaDB...")
collection.add(
    embeddings=embeddings.tolist(),
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
print("   ✅ 数据添加成功")

# 查看集合信息
count = collection.count()
print(f"\n📊 集合统计:")
print(f"   • 文档数量: {count}")

# ============================================================
# 第四部分：查询（Query）
# ============================================================

print("\n【第四部分：查询数据】")
print("-" * 60)

# 测试问题
query = "什么是RAG？"
print(f"\n❓ 问题: {query}")

# 生成问题向量
query_embedding = model.encode([query], show_progress_bar=False)

# 查询最相似的3个结果
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

print("\n🔍 检索结果（Top-3）:")
print("-" * 60)
for i in range(len(results['ids'][0])):
    doc_id = results['ids'][0][i]
    document = results['documents'][0][i]
    metadata = results['metadatas'][0][i]
    distance = results['distances'][0][i]
    similarity = 1 - distance  # 转换为相似度
    
    print(f"\n[{i+1}] ID: {doc_id}")
    print(f"    文档: {document}")
    print(f"    元数据: {metadata}")
    print(f"    距离: {distance:.4f}")
    print(f"    相似度: {similarity:.4f} ({similarity*100:.1f}%)")

# ============================================================
# 第五部分：元数据过滤
# ============================================================

print("\n【第五部分：元数据过滤】")
print("-" * 60)

query2 = "人工智能相关的技术"
print(f"\n❓ 问题: {query2}")
print("🔧 过滤条件: category='AI'")

query_embedding2 = model.encode([query2], show_progress_bar=False)

results2 = collection.query(
    query_embeddings=query_embedding2.tolist(),
    n_results=3,
    where={"category": "AI"},  # 只搜索AI类别
    include=["documents", "metadatas", "distances"]
)

print("\n🔍 过滤后的结果:")
print("-" * 60)
for i in range(len(results2['ids'][0])):
    document = results2['documents'][0][i]
    metadata = results2['metadatas'][0][i]
    distance = results2['distances'][0][i]
    
    print(f"\n[{i+1}] {document}")
    print(f"    分类: {metadata['category']}, 难度: {metadata['difficulty']}")
    print(f"    相似度: {(1-distance)*100:.1f}%")

# ============================================================
# 第六部分：更新数据（Update）
# ============================================================

print("\n【第六部分：更新数据】")
print("-" * 60)

print("\n📝 更新 doc_0 的内容...")
new_document = "Python是一种简单易学的高级编程语言"
new_embedding = model.encode([new_document], show_progress_bar=False)

collection.update(
    ids=["doc_0"],
    embeddings=new_embedding.tolist(),
    documents=[new_document],
    metadatas=[{"category": "编程", "difficulty": "入门", "updated": True}]
)
print("   ✅ 更新成功")

# 验证更新
result = collection.get(ids=["doc_0"], include=["documents", "metadatas"])
print(f"\n🔍 更新后的内容:")
print(f"   文档: {result['documents'][0]}")
print(f"   元数据: {result['metadatas'][0]}")

# ============================================================
# 第七部分：删除数据（Delete）
# ============================================================

print("\n【第七部分：删除数据】")
print("-" * 60)

print(f"\n🗑️  删除前的数量: {collection.count()}")

# 删除单个文档
collection.delete(ids=["doc_4"])
print("   ✅ 删除 doc_4")

print(f"📊 删除后的数量: {collection.count()}")

# ============================================================
# 第八部分：获取所有数据（Get）
# ============================================================

print("\n【第八部分：获取所有数据】")
print("-" * 60)

all_data = collection.get(include=["documents", "metadatas"])

print(f"\n📋 集合中的所有文档 (共{len(all_data['ids'])}个):")
for i, (doc_id, doc, meta) in enumerate(zip(
    all_data['ids'], 
    all_data['documents'], 
    all_data['metadatas']
), 1):
    print(f"\n[{i}] {doc_id}")
    print(f"    {doc}")
    print(f"    {meta}")

# ============================================================
# 第九部分：集合管理
# ============================================================

print("\n【第九部分：集合管理】")
print("-" * 60)

# 列出所有集合
all_collections = client.list_collections()
print(f"\n📚 所有集合 (共{len(all_collections)}个):")
for coll in all_collections:
    print(f"   • {coll.name}: {coll.count()} 个文档")

# 获取集合元数据
metadata = collection.metadata
print(f"\n📋 集合元数据:")
for key, value in metadata.items():
    print(f"   • {key}: {value}")

# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 60)
print("🎉 ChromaDB基础操作完成！")
print("=" * 60)

print("\n✅ 已掌握的技能:")
print("   1. 创建持久化数据库")
print("   2. 创建和管理集合")
print("   3. 添加向量数据")
print("   4. 语义搜索查询")
print("   5. 元数据过滤")
print("   6. 更新和删除数据")
print("   7. 获取所有数据")
print("   8. 集合管理")

print("\n📁 数据已保存到:")
print(f"   {os.path.abspath(db_path)}")

print("\n🎯 下一步:")
print("   运行: python 02_import_traffic_law.py")
print("   学习如何导入真实数据集")

print("\n" + "=" * 60)

