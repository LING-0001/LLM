#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4.3: 高级检索技巧
学习目标：掌握复杂查询和结果优化
"""

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import json

print("=" * 60)
print("🔍 ChromaDB高级检索技巧")
print("=" * 60)

# ============================================================
# 第一部分：连接数据库
# ============================================================

print("\n【第一部分：连接数据库】")
print("-" * 60)

# 连接到已有的数据库
db_path = "./data/chroma_traffic_law"
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="traffic_law")

count = collection.count()
print(f"✅ 连接成功")
print(f"📊 数据库包含 {count} 个文档")

# 加载模型
print(f"\n📦 加载embedding模型...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')
print("   ✅ 模型加载完成")

# ============================================================
# 第二部分：基础检索回顾
# ============================================================

print("\n【第二部分：基础检索】")
print("-" * 60)

question = "酒驾的处罚是什么？"
print(f"\n❓ 问题: {question}")

# 生成向量
query_embedding = model.encode([question], show_progress_bar=False)

# 基础检索（Top-3）
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

print("\n🔍 基础检索结果:")
for i in range(len(results['ids'][0])):
    document = results['documents'][0][i]
    distance = results['distances'][0][i]
    similarity = 1 - distance
    
    print(f"\n[Top-{i+1}] 相似度: {similarity*100:.1f}%")
    preview = document[:80] + "..." if len(document) > 80 else document
    print(f"   {preview}")

# ============================================================
# 第三部分：元数据过滤（Where子句）
# ============================================================

print("\n【第三部分：元数据过滤】")
print("-" * 60)

# 场景1：按章节过滤
print("\n🔹 场景1: 只搜索「第五章」")
question1 = "交通事故如何处理？"
print(f"   问题: {question1}")

query_embedding1 = model.encode([question1], show_progress_bar=False)

results1 = collection.query(
    query_embeddings=query_embedding1.tolist(),
    n_results=2,
    where={"chapter": "第五章"},  # 只搜索第五章
    include=["documents", "metadatas", "distances"]
)

print(f"\n   结果:")
for i in range(len(results1['ids'][0])):
    chapter = results1['metadatas'][0][i]['chapter']
    document = results1['documents'][0][i]
    similarity = 1 - results1['distances'][0][i]
    
    print(f"   [{i+1}] {chapter} | 相似度: {similarity*100:.1f}%")
    preview = document[:60] + "..." if len(document) > 60 else document
    print(f"       {preview}")

# 场景2：按长度过滤（找长文本）
print("\n🔹 场景2: 只搜索长度>350的文本块")
question2 = "驾驶证扣分制度"
print(f"   问题: {question2}")

query_embedding2 = model.encode([question2], show_progress_bar=False)

results2 = collection.query(
    query_embeddings=query_embedding2.tolist(),
    n_results=2,
    where={"length": {"$gt": 350}},  # 长度大于350
    include=["documents", "metadatas", "distances"]
)

print(f"\n   结果:")
for i in range(len(results2['ids'][0])):
    length = results2['metadatas'][0][i]['length']
    document = results2['documents'][0][i]
    similarity = 1 - results2['distances'][0][i]
    
    print(f"   [{i+1}] 长度: {length} | 相似度: {similarity*100:.1f}%")
    preview = document[:60] + "..." if len(document) > 60 else document
    print(f"       {preview}")

# 场景3：复合条件（AND逻辑）
print("\n🔹 场景3: 第三章 AND 长度>300")
question3 = "扣分规则"
print(f"   问题: {question3}")

query_embedding3 = model.encode([question3], show_progress_bar=False)

results3 = collection.query(
    query_embeddings=query_embedding3.tolist(),
    n_results=2,
    where={
        "$and": [
            {"chapter": "第三章"},
            {"length": {"$gt": 300}}
        ]
    },
    include=["documents", "metadatas", "distances"]
)

if len(results3['ids'][0]) > 0:
    print(f"\n   结果:")
    for i in range(len(results3['ids'][0])):
        chapter = results3['metadatas'][0][i]['chapter']
        length = results3['metadatas'][0][i]['length']
        document = results3['documents'][0][i]
        similarity = 1 - results3['distances'][0][i]
        
        print(f"   [{i+1}] {chapter}, {length}字 | 相似度: {similarity*100:.1f}%")
        preview = document[:60] + "..." if len(document) > 60 else document
        print(f"       {preview}")
else:
    print("   ⚠️  没有找到符合条件的文档")

# ============================================================
# 第四部分：调整返回数量（Top-K）
# ============================================================

print("\n【第四部分：调整Top-K】")
print("-" * 60)

question = "超速怎么处罚？"
print(f"\n❓ 问题: {question}")

query_embedding = model.encode([question], show_progress_bar=False)

# 测试不同的K值
k_values = [1, 3, 5]

for k in k_values:
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k,
        include=["documents", "distances"]
    )
    
    print(f"\n📊 Top-{k} 结果:")
    for i in range(len(results['ids'][0])):
        similarity = 1 - results['distances'][0][i]
        print(f"   [{i+1}] 相似度: {similarity*100:.1f}%")

# ============================================================
# 第五部分：相似度阈值过滤
# ============================================================

print("\n【第五部分：相似度阈值过滤】")
print("-" * 60)

question = "停车规定"
print(f"\n❓ 问题: {question}")

query_embedding = model.encode([question], show_progress_bar=False)

# 获取Top-10
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=10,
    include=["documents", "distances"]
)

# 只保留相似度>70%的结果
threshold = 0.70
print(f"\n🎯 过滤条件: 相似度 > {threshold*100}%")

filtered_results = []
for i in range(len(results['ids'][0])):
    distance = results['distances'][0][i]
    similarity = 1 - distance
    
    if similarity > threshold:
        filtered_results.append({
            'document': results['documents'][0][i],
            'similarity': similarity
        })

print(f"\n✅ 找到 {len(filtered_results)} 个高质量结果:")
for i, result in enumerate(filtered_results, 1):
    print(f"\n[{i}] 相似度: {result['similarity']*100:.1f}%")
    preview = result['document'][:80] + "..." if len(result['document']) > 80 else result['document']
    print(f"    {preview}")

if len(filtered_results) == 0:
    print("   ⚠️  没有找到相似度超过阈值的结果")
    print("   💡 建议：降低阈值或改写问题")

# ============================================================
# 第六部分：多问题批量检索
# ============================================================

print("\n【第六部分：批量检索】")
print("-" * 60)

questions = [
    "闯红灯扣分",
    "酒驾处罚",
    "超速罚款"
]

print(f"\n📝 批量检索 {len(questions)} 个问题...")

# 批量生成向量
query_embeddings = model.encode(questions, show_progress_bar=False)

# 批量查询
results = collection.query(
    query_embeddings=query_embeddings.tolist(),
    n_results=2,
    include=["documents", "distances"]
)

# 显示结果
for i, question in enumerate(questions):
    print(f"\n【问题 {i+1}】{question}")
    
    for j in range(len(results['ids'][i])):
        similarity = 1 - results['distances'][i][j]
        document = results['documents'][i][j]
        
        print(f"  [Top-{j+1}] 相似度: {similarity*100:.1f}%")
        preview = document[:60] + "..." if len(document) > 60 else document
        print(f"          {preview}")

# ============================================================
# 第七部分：结果去重
# ============================================================

print("\n【第七部分：结果去重】")
print("-" * 60)

# 相似问题可能检索到相同的文档
questions = [
    "酒后驾驶的处罚",
    "醉驾会受到什么惩罚"
]

print(f"\n📝 检索相似问题:")
for q in questions:
    print(f"   • {q}")

all_doc_ids = set()
all_results = []

for question in questions:
    query_embedding = model.encode([question], show_progress_bar=False)
    
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3,
        include=["documents", "distances"]
    )
    
    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        
        # 去重
        if doc_id not in all_doc_ids:
            all_doc_ids.add(doc_id)
            all_results.append({
                'id': doc_id,
                'document': results['documents'][0][i],
                'similarity': 1 - results['distances'][0][i]
            })

print(f"\n✅ 去重后的结果 (共{len(all_results)}个):")
for i, result in enumerate(all_results, 1):
    print(f"\n[{i}] ID: {result['id']} | 相似度: {result['similarity']*100:.1f}%")
    preview = result['document'][:70] + "..." if len(result['document']) > 70 else result['document']
    print(f"    {preview}")

# ============================================================
# 第八部分：检索性能测试
# ============================================================

print("\n【第八部分：性能测试】")
print("-" * 60)

import time

# 测试不同Top-K的速度
print(f"\n⏱️  测试检索速度（10次平均）...")

question = "交通违法处罚标准"
query_embedding = model.encode([question], show_progress_bar=False)

for k in [1, 5, 10, 20]:
    times = []
    
    for _ in range(10):
        start = time.time()
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            include=["distances"]
        )
        end = time.time()
        times.append((end - start) * 1000)  # 转为毫秒
    
    avg_time = np.mean(times)
    print(f"   Top-{k:2d}: {avg_time:.2f}ms")

# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 60)
print("🎉 高级检索技巧学习完成！")
print("=" * 60)

print("\n✅ 掌握的技能:")
print("   1. 基础语义检索")
print("   2. 元数据过滤（where子句）")
print("   3. 复合条件查询（$and, $gt等）")
print("   4. Top-K调整")
print("   5. 相似度阈值过滤")
print("   6. 批量检索")
print("   7. 结果去重")
print("   8. 性能测试")

print("\n💡 实用技巧:")
print("   • 先用高Top-K检索，再用阈值过滤")
print("   • 结合元数据缩小搜索范围")
print("   • 批量检索提高效率")
print("   • 去重避免重复内容")

print("\n🎯 下一步:")
print("   运行: python 04_performance.py")
print("   学习性能优化技巧")

print("\n" + "=" * 60)

