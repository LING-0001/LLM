#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4.2: 导入交通法数据到ChromaDB
学习目标：将真实数据导入向量数据库
"""

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

print("=" * 60)
print("🚗 导入交通法数据到ChromaDB")
print("=" * 60)

# ============================================================
# 第一部分：加载预处理的数据
# ============================================================

print("\n【第一部分：加载数据】")
print("-" * 60)

# 检查数据文件
data_dir = "../../data"
vectors_path = os.path.join(data_dir, "traffic_law_vectors.npy")
json_path = os.path.join(data_dir, "traffic_law_data.json")

if not os.path.exists(vectors_path):
    print("❌ 错误：找不到向量文件！")
    print(f"   请先运行: python ../../prepare_traffic_law_data.py")
    exit(1)

if not os.path.exists(json_path):
    print("❌ 错误：找不到数据文件！")
    print(f"   请先运行: python ../../prepare_traffic_law_data.py")
    exit(1)

# 加载向量
print(f"\n📦 加载向量数据...")
vectors = np.load(vectors_path)
print(f"   ✅ 加载成功: {vectors.shape[0]} 个向量，维度: {vectors.shape[1]}")

# 加载JSON数据
print(f"\n📦 加载文本数据...")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

chunks = data['chunks']

print(f"   ✅ 加载成功: {len(chunks)} 个文本块")
print(f"\n📊 数据集信息:")
print(f"   • 文档来源: {data['source_file']}")
print(f"   • 模型名称: {data['model_name']}")
print(f"   • 分块数量: {data['num_chunks']} 个")
print(f"   • 分块大小: {data['chunk_size']} 字符")
print(f"   • 重叠大小: {data['chunk_overlap']} 字符")
print(f"   • 向量维度: {data['vector_dim']}")

# ============================================================
# 第二部分：创建ChromaDB数据库
# ============================================================

print("\n【第二部分：创建向量数据库】")
print("-" * 60)

# 创建持久化数据库
db_path = "./data/chroma_traffic_law"
os.makedirs(db_path, exist_ok=True)

client = chromadb.PersistentClient(path=db_path)
print(f"✅ 数据库创建成功")
print(f"📁 存储路径: {db_path}")

# 删除旧集合（如果存在）
try:
    client.delete_collection(name="traffic_law")
    print("🗑️  删除旧集合: traffic_law")
except:
    pass

# 创建新集合
collection = client.create_collection(
    name="traffic_law",
    metadata={
        "description": "中国交通法规知识库",
        "hnsw:space": "cosine",  # 余弦相似度
        "source": data['source_file'],
        "model": data['model_name']
    }
)
print("✅ 创建新集合: traffic_law")

# ============================================================
# 第三部分：批量导入数据
# ============================================================

print("\n【第三部分：批量导入数据】")
print("-" * 60)

# 准备数据
documents = [chunk['content'] for chunk in chunks]
ids = [chunk['id'] for chunk in chunks]
metadatas = [{
    'chapter': chunk['chapter'],
    'length': chunk['length'],
    'chunk_id': chunk['index']
} for chunk in chunks]

print(f"\n📝 准备导入 {len(documents)} 个文本块...")

# 批量添加
batch_size = 50  # 每次添加50个
total = len(documents)

for i in range(0, total, batch_size):
    end_idx = min(i + batch_size, total)
    batch_vectors = vectors[i:end_idx].tolist()
    batch_documents = documents[i:end_idx]
    batch_ids = ids[i:end_idx]
    batch_metadatas = metadatas[i:end_idx]
    
    collection.add(
        embeddings=batch_vectors,
        documents=batch_documents,
        ids=batch_ids,
        metadatas=batch_metadatas
    )
    
    print(f"   ✅ 批次 {i//batch_size + 1}: 已导入 {end_idx}/{total} 个文本块")

print(f"\n🎉 所有数据导入完成！")

# 验证导入
count = collection.count()
print(f"\n📊 验证结果:")
print(f"   • 集合中的文档数: {count}")
print(f"   • 预期文档数: {total}")
print(f"   • 导入状态: {'✅ 成功' if count == total else '❌ 失败'}")

# ============================================================
# 第四部分：测试检索效果
# ============================================================

print("\n【第四部分：测试检索效果】")
print("-" * 60)

# 加载embedding模型
print(f"\n📦 加载embedding模型...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')
print("   ✅ 模型加载完成")

# 测试问题列表
test_questions = [
    "酒驾会受到什么处罚？",
    "闯红灯要扣几分？",
    "新手司机实习期有什么规定？",
    "超速行驶会被怎么处理？",
    "交通事故后应该怎么办？"
]

print(f"\n🧪 测试 {len(test_questions)} 个问题...")
print("=" * 60)

for idx, question in enumerate(test_questions, 1):
    print(f"\n【问题 {idx}】{question}")
    print("-" * 60)
    
    # 生成问题向量
    query_embedding = model.encode([question], show_progress_bar=False)
    
    # 检索Top-3
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    # 显示结果
    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        document = results['documents'][0][i]
        distance = results['distances'][0][i]
        similarity = 1 - distance
        chunk_id = results['metadatas'][0][i]['chunk_id']
        length = results['metadatas'][0][i]['length']
        
        print(f"\n[Top-{i+1}] 相似度: {similarity*100:.1f}%")
        print(f"   块ID: {chunk_id} | 长度: {length}字符")
        
        # 显示前100个字符
        preview = document[:100] + "..." if len(document) > 100 else document
        print(f"   内容: {preview}")

# ============================================================
# 第五部分：数据统计分析
# ============================================================

print("\n" + "=" * 60)
print("【第五部分：数据统计分析】")
print("-" * 60)

# 获取所有数据
all_data = collection.get(include=["metadatas"])

# 统计长度分布
lengths = [meta['length'] for meta in all_data['metadatas']]
avg_length = np.mean(lengths)
min_length = np.min(lengths)
max_length = np.max(lengths)

print(f"\n📊 文本块长度统计:")
print(f"   • 平均长度: {avg_length:.1f} 字符")
print(f"   • 最短: {min_length} 字符")
print(f"   • 最长: {max_length} 字符")

# 统计章节分布
chapters = [meta['chapter'] for meta in all_data['metadatas']]
chapter_counts = {}
for chapter in chapters:
    chapter_counts[chapter] = chapter_counts.get(chapter, 0) + 1

print(f"\n📖 章节分布:")
for chapter, count in sorted(chapter_counts.items()):
    bar = "█" * (count // 2)
    print(f"   {chapter}: {count:2d} 个 {bar}")

# ============================================================
# 第六部分：元数据过滤测试
# ============================================================

print("\n【第六部分：元数据过滤测试】")
print("-" * 60)

# 只搜索特定章节
question = "驾驶证扣分有什么规定？"
print(f"\n❓ 问题: {question}")
print("🔧 过滤: 只在「第三章」中搜索")

query_embedding = model.encode([question], show_progress_bar=False)

results_filtered = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=2,
    where={"chapter": "第三章"},
    include=["documents", "metadatas", "distances"]
)

print(f"\n🔍 过滤后的结果:")
for i in range(len(results_filtered['ids'][0])):
    document = results_filtered['documents'][0][i]
    distance = results_filtered['distances'][0][i]
    similarity = 1 - distance
    chapter = results_filtered['metadatas'][0][i]['chapter']
    
    print(f"\n[{i+1}] {chapter} | 相似度: {similarity*100:.1f}%")
    preview = document[:80] + "..." if len(document) > 80 else document
    print(f"    {preview}")

# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 60)
print("🎉 交通法数据导入完成！")
print("=" * 60)

print("\n✅ 完成的工作:")
print("   1. 加载预处理的向量数据")
print("   2. 创建ChromaDB持久化数据库")
print("   3. 批量导入文本和向量")
print("   4. 测试语义检索效果")
print("   5. 数据统计分析")
print("   6. 元数据过滤测试")

print(f"\n📁 数据库位置:")
print(f"   {os.path.abspath(db_path)}")

print(f"\n📊 数据库规模:")
print(f"   • 文档数: {count}")
print(f"   • 总字符数: {sum(chunk['length'] for chunk in chunks)}")
print(f"   • 向量维度: {vectors.shape[1]}")

print("\n💡 使用方法:")
print("   1. 其他脚本可以直接连接这个数据库")
print("   2. 路径: ./data/chroma_traffic_law")
print("   3. 集合名: traffic_law")

print("\n🎯 下一步:")
print("   运行: python 03_advanced_query.py")
print("   学习高级检索技巧")

print("\n" + "=" * 60)

