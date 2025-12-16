#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5.2: 优化检索效果
学习目标：通过调整检索策略提升RAG质量
"""

import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import os

print("=" * 60)
print("🔍 优化RAG检索效果")
print("=" * 60)

# ============================================================
# 加载组件
# ============================================================

print("\n📦 加载系统组件...")

# 向量数据库
db_path = "../step4_vectorstore/data/chroma_traffic_law"
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="traffic_law")

# Embedding模型
embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')

# LLM
llm = Llama(
    model_path="/Users/a58/llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False
)

print("✅ 所有组件加载完成\n")

# ============================================================
# 第一部分：Top-K对比测试
# ============================================================

print("【第一部分：Top-K对比测试】")
print("=" * 60)

def retrieve_with_topk(question, top_k):
    """检索文档"""
    q_vec = embedding_model.encode([question], show_progress_bar=False)
    results = collection.query(
        query_embeddings=q_vec.tolist(),
        n_results=top_k,
        include=["documents", "distances"]
    )
    
    docs = []
    for i in range(len(results['ids'][0])):
        docs.append({
            'content': results['documents'][0][i],
            'similarity': 1 - results['distances'][0][i]
        })
    return docs

question = "酒驾的处罚是什么？"
print(f"\n❓ 测试问题: {question}\n")

# 测试不同的Top-K
for k in [1, 3, 5]:
    print(f"\n{'='*60}")
    print(f"Top-{k} 检索结果")
    print(f"{'='*60}")
    
    docs = retrieve_with_topk(question, k)
    
    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}] 相似度: {doc['similarity']*100:.1f}%")
        preview = doc['content'][:80] + "..." if len(doc['content']) > 80 else doc['content']
        print(f"    {preview}")

print(f"\n💡 观察:")
print("   • Top-1: 速度快，但可能信息不全")
print("   • Top-3: 平衡，适合大多数情况 ⭐")
print("   • Top-5: 信息全，但可能有噪音")

# ============================================================
# 第二部分：相似度阈值过滤
# ============================================================

print("\n\n【第二部分：相似度阈值过滤】")
print("=" * 60)

def retrieve_with_threshold(question, top_k=10, threshold=0.7):
    """
    检索并过滤低相似度文档
    
    Args:
        question: 问题
        top_k: 初始检索数量
        threshold: 相似度阈值(0-1)
    
    Returns:
        过滤后的文档列表
    """
    # 先检索Top-K
    docs = retrieve_with_topk(question, top_k)
    
    # 过滤低相似度
    filtered_docs = [
        doc for doc in docs
        if doc['similarity'] >= threshold
    ]
    
    return filtered_docs, docs

test_cases = [
    ("酒驾处罚标准", 0.75),  # 精确问题
    ("路上开车要注意什么", 0.60),  # 模糊问题
]

for question, threshold in test_cases:
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"阈值: {threshold}")
    print(f"{'='*60}")
    
    filtered, all_docs = retrieve_with_threshold(question, top_k=5, threshold=threshold)
    
    print(f"\n📊 检索结果:")
    print(f"   • 初始检索: {len(all_docs)} 个文档")
    print(f"   • 过滤后: {len(filtered)} 个文档")
    
    if filtered:
        print(f"\n✅ 高质量文档:")
        for i, doc in enumerate(filtered, 1):
            print(f"   [{i}] 相似度: {doc['similarity']*100:.1f}%")
    else:
        print(f"\n⚠️ 没有文档超过阈值，建议：")
        print(f"   1. 降低阈值")
        print(f"   2. 改写问题")
        print(f"   3. 回答「文档中未找到相关信息」")

print(f"\n💡 阈值选择建议:")
print("   • 高阈值(0.8+): 严格，适合专业场景")
print("   • 中阈值(0.7): 平衡，推荐 ⭐")
print("   • 低阈值(0.6): 宽松，适合模糊问题")

# ============================================================
# 第三部分：元数据过滤
# ============================================================

print("\n\n【第三部分：元数据过滤】")
print("=" * 60)

def retrieve_with_metadata(question, chapter=None, min_length=None, top_k=5):
    """
    使用元数据过滤检索
    
    Args:
        question: 问题
        chapter: 指定章节
        min_length: 最小文档长度
        top_k: 返回数量
    """
    q_vec = embedding_model.encode([question], show_progress_bar=False)
    
    # 构建where条件
    where_clause = None
    if chapter and min_length:
        where_clause = {
            "$and": [
                {"chapter": chapter},
                {"length": {"$gte": min_length}}
            ]
        }
    elif chapter:
        where_clause = {"chapter": chapter}
    elif min_length:
        where_clause = {"length": {"$gte": min_length}}
    
    # 检索
    results = collection.query(
        query_embeddings=q_vec.tolist(),
        n_results=top_k,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )
    
    docs = []
    for i in range(len(results['ids'][0])):
        docs.append({
            'content': results['documents'][0][i],
            'chapter': results['metadatas'][0][i]['chapter'],
            'length': results['metadatas'][0][i]['length'],
            'similarity': 1 - results['distances'][0][i]
        })
    
    return docs

# 测试场景
print(f"\n场景1: 只在「第三章」中搜索驾驶证相关问题")
q1 = "驾驶证扣分规定"
docs1 = retrieve_with_metadata(q1, chapter="第三章：机动车驾驶证管理", top_k=2)

print(f"\n❓ 问题: {q1}")
print(f"🔧 过滤: chapter='第三章：机动车驾驶证管理'")
print(f"\n结果:")
for i, doc in enumerate(docs1, 1):
    print(f"\n[{i}] {doc['chapter']} | 长度:{doc['length']} | 相似度:{doc['similarity']:.2%}")
    preview = doc['content'][:60] + "..." if len(doc['content']) > 60 else doc['content']
    print(f"    {preview}")

print(f"\n{'='*60}")
print(f"场景2: 只搜索详细文档（长度>250）")
q2 = "交通违法处罚"
docs2 = retrieve_with_metadata(q2, min_length=250, top_k=3)

print(f"\n❓ 问题: {q2}")
print(f"🔧 过滤: length >= 250")
print(f"\n结果:")
for i, doc in enumerate(docs2, 1):
    print(f"\n[{i}] 长度:{doc['length']} | 相似度:{doc['similarity']:.2%}")
    print(f"    {doc['chapter']}")

print(f"\n💡 元数据过滤的优势:")
print("   • 缩小搜索范围，提高准确性")
print("   • 避免检索到不相关章节")
print("   • 可以过滤太短的碎片文档")

# ============================================================
# 第四部分：结果去重
# ============================================================

print("\n\n【第四部分：结果去重】")
print("=" * 60)

def retrieve_and_deduplicate(questions, top_k=3):
    """
    多个相似问题检索并去重
    
    Args:
        questions: 问题列表
        top_k: 每个问题检索数量
    
    Returns:
        去重后的文档
    """
    all_docs = {}  # 用dict自动去重
    
    for question in questions:
        q_vec = embedding_model.encode([question], show_progress_bar=False)
        results = collection.query(
            query_embeddings=q_vec.tolist(),
            n_results=top_k,
            include=["documents", "distances"]
        )
        
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    'content': results['documents'][0][i],
                    'similarity': 1 - results['distances'][0][i],
                    'from_question': question
                }
    
    return list(all_docs.values())

# 测试
similar_questions = [
    "酒后驾驶的处罚",
    "醉驾会受到什么惩罚",
    "喝酒开车怎么处理"
]

print(f"\n🔄 测试：多个相似问题")
for i, q in enumerate(similar_questions, 1):
    print(f"   {i}. {q}")

# 不去重
total_without_dedup = 0
for q in similar_questions:
    docs = retrieve_with_topk(q, 3)
    total_without_dedup += len(docs)

# 去重
deduplicated_docs = retrieve_and_deduplicate(similar_questions, top_k=3)

print(f"\n📊 统计:")
print(f"   • 不去重: {total_without_dedup} 个文档")
print(f"   • 去重后: {len(deduplicated_docs)} 个文档")
print(f"   • 减少: {total_without_dedup - len(deduplicated_docs)} 个重复")

print(f"\n去重后的文档:")
for i, doc in enumerate(deduplicated_docs[:3], 1):
    print(f"\n[{i}] 相似度: {doc['similarity']:.2%}")
    print(f"    来自问题: {doc['from_question']}")
    preview = doc['content'][:60] + "..." if len(doc['content']) > 60 else doc['content']
    print(f"    {preview}")

print(f"\n💡 去重的好处:")
print("   • 避免给LLM重复信息")
print("   • 节省token和时间")
print("   • 提升答案质量")

# ============================================================
# 第五部分：组合优化策略
# ============================================================

print("\n\n【第五部分：组合优化策略】")
print("=" * 60)

def optimized_retrieve(question, top_k=10, threshold=0.7, max_results=3):
    """
    组合多种优化策略的检索函数
    
    策略：
    1. 先检索较多文档(top_k=10)
    2. 相似度过滤(threshold=0.7)
    3. 限制最终结果数量(max_results=3)
    """
    # 检索
    q_vec = embedding_model.encode([question], show_progress_bar=False)
    results = collection.query(
        query_embeddings=q_vec.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # 处理和过滤
    filtered_docs = []
    for i in range(len(results['ids'][0])):
        similarity = 1 - results['distances'][0][i]
        
        if similarity >= threshold:
            filtered_docs.append({
                'content': results['documents'][0][i],
                'chapter': results['metadatas'][0][i]['chapter'],
                'similarity': similarity
            })
    
    # 返回Top-N
    return filtered_docs[:max_results]

# 测试
test_q = "交通事故逃逸的后果"

print(f"\n❓ 问题: {test_q}")
print(f"🔧 优化策略:")
print(f"   1. 初始检索 Top-10")
print(f"   2. 过滤相似度 < 0.7")
print(f"   3. 返回最终 Top-3")

optimized_docs = optimized_retrieve(test_q, top_k=10, threshold=0.7, max_results=3)

print(f"\n✅ 最终结果 ({len(optimized_docs)}个):")
for i, doc in enumerate(optimized_docs, 1):
    print(f"\n[{i}] 相似度: {doc['similarity']:.2%} | {doc['chapter']}")
    preview = doc['content'][:70] + "..." if len(doc['content']) > 70 else doc['content']
    print(f"    {preview}")

if len(optimized_docs) == 0:
    print("\n⚠️  没有高质量文档，建议回答：")
    print("    「抱歉，我在文档中没有找到相关信息。」")

# ============================================================
# 总结
# ============================================================

print("\n\n" + "=" * 60)
print("🎉 检索优化学习完成！")
print("=" * 60)

print("\n✅ 掌握的优化技巧:")
print("   1. Top-K调整（1 vs 3 vs 5）")
print("   2. 相似度阈值过滤（去除低质量）")
print("   3. 元数据过滤（缩小范围）")
print("   4. 结果去重（避免重复）")
print("   5. 组合策略（多重优化）")

print("\n💡 最佳实践:")
print("   • 先检索Top-10，再过滤到Top-3")
print("   • 设置阈值0.7，过滤低质量")
print("   • 使用元数据缩小范围")
print("   • 多问题检索要去重")

print("\n🎯 下一步:")
print("   运行: python 03_improve_prompt.py")
print("   学习如何优化Prompt提升生成质量")

print("\n" + "=" * 60)

