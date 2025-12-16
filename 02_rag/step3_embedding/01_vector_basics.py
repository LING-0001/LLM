"""
练习1：理解向量基础
从零开始理解什么是向量，以及如何计算相似度
"""

import numpy as np
import math

print("="*70)
print(" "*20 + "向量基础知识")
print("="*70)
print()

# 1. 什么是向量？
print("="*70)
print("1. 什么是向量？")
print("="*70)
print()

print("向量就是一组数字：")
print()

# 2维向量（可以画在平面上）
vec_2d = [3, 4]
print(f"2维向量: {vec_2d}")
print("   可以表示平面上的一个点或方向")
print()

# 3维向量（可以画在空间中）
vec_3d = [1, 2, 3]
print(f"3维向量: {vec_3d}")
print("   可以表示空间中的一个点或方向")
print()

# 高维向量（无法直观可视化，但数学上一样）
vec_high = np.random.rand(768)
print(f"768维向量: [{vec_high[0]:.3f}, {vec_high[1]:.3f}, {vec_high[2]:.3f}, ..., {vec_high[-1]:.3f}]")
print("   Embedding模型生成的向量就是这样的高维向量")
print()

# 2. 向量的运算
print("="*70)
print("2. 向量的基本运算")
print("="*70)
print()

A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

print(f"向量A: {A}")
print(f"向量B: {B}")
print()

# 加法
print(f"A + B = {A + B}  ← 向量加法")

# 减法
print(f"A - B = {A - B}  ← 向量减法")

# 数乘
print(f"2 × A = {2 * A}  ← 数乘")
print()

# 点积（内积）
dot_product = np.dot(A, B)
print(f"A · B = {dot_product}  ← 点积（用于计算相似度）")
print(f"计算过程: {A[0]}×{B[0]} + {A[1]}×{B[1]} + {A[2]}×{B[2]} = {dot_product}")
print()

# 3. 向量的长度
print("="*70)
print("3. 向量的长度（模）")
print("="*70)
print()

def vector_length(vec):
    """计算向量长度"""
    return math.sqrt(sum(x**2 for x in vec))

length_A = vector_length(A)
print(f"向量A的长度: √({A[0]}² + {A[1]}² + {A[2]}²) = {length_A:.3f}")
print()

# 单位化（归一化）
A_normalized = A / length_A
print(f"A单位化后: {A_normalized}")
print(f"长度变为: {vector_length(A_normalized):.3f}  ← 变成1了！")
print()

# 4. 余弦相似度
print("="*70)
print("4. 余弦相似度（最重要！）")
print("="*70)
print()

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2)

print("余弦相似度公式：")
print("   相似度 = (A·B) / (|A| × |B|)")
print("   范围：-1 到 1")
print("   越接近1 = 越相似")
print()

# 测试不同的向量对
test_pairs = [
    ([1, 2, 3], [2, 4, 6], "成比例（相同方向）"),
    ([1, 0, 0], [1, 0, 0], "完全相同"),
    ([1, 0, 0], [0, 1, 0], "垂直（无关）"),
    ([1, 2, 3], [3, 2, 1], "有些相似"),
    ([1, 0, 0], [-1, 0, 0], "完全相反"),
]

for vec1, vec2, desc in test_pairs:
    sim = cosine_similarity(np.array(vec1), np.array(vec2))
    print(f"{str(vec1):20} vs {str(vec2):20} → {sim:6.3f}  ({desc})")

print()

# 5. 文本向量化的意义
print("="*70)
print("5. 为什么要把文本转成向量？")
print("="*70)
print()

print("假设我们有这些句子的向量（简化为3维）：")
print()

# 模拟的句子向量
sentences = {
    "Python很好用": np.array([0.8, 0.6, 0.1]),
    "Python非常实用": np.array([0.75, 0.65, 0.15]),
    "Java很强大": np.array([0.3, 0.7, 0.8]),
    "天气很好": np.array([0.1, 0.2, 0.9]),
}

query = "Python怎么样"
query_vec = np.array([0.7, 0.55, 0.2])

print(f"问题：'{query}'")
print(f"问题向量: {query_vec}")
print()
print("计算与各句子的相似度：")
print()

similarities = {}
for sentence, vec in sentences.items():
    sim = cosine_similarity(query_vec, vec)
    similarities[sentence] = sim
    print(f"  '{sentence}': {sim:.3f}")

print()
print("排序后（相似度从高到低）：")
for sentence, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(sim * 50)
    print(f"  {sentence:20} {bar} {sim:.3f}")

print()
print("💡 观察：")
print("   • 意思相近的句子，向量也相近")
print("   • 可以通过计算相似度找到最相关的内容")
print("   • 这就是RAG中检索的原理！")
print()

# 6. 实战：简单的文本检索
print("="*70)
print("6. 实战：简单的文本检索")
print("="*70)
print()

# 知识库（模拟）
knowledge_base = {
    "Python是一种编程语言": np.array([0.9, 0.1, 0.2, 0.1]),
    "Python适合数据科学": np.array([0.8, 0.3, 0.7, 0.2]),
    "Java是面向对象的": np.array([0.3, 0.9, 0.1, 0.1]),
    "机器学习很有趣": np.array([0.2, 0.3, 0.9, 0.8]),
    "深度学习需要GPU": np.array([0.1, 0.2, 0.8, 0.9]),
}

def search(query_text, query_vector, knowledge_base, top_k=3):
    """简单的向量检索"""
    results = []
    for text, vec in knowledge_base.items():
        sim = cosine_similarity(query_vector, vec)
        results.append((text, sim))
    
    # 按相似度排序，返回Top-K
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# 测试不同的问题
queries = [
    ("Python有什么用？", np.array([0.85, 0.2, 0.5, 0.1])),
    ("如何学习机器学习？", np.array([0.2, 0.1, 0.9, 0.7])),
]

for query_text, query_vec in queries:
    print(f"问题：{query_text}")
    print(f"Top 3 相关文档：")
    
    results = search(query_text, query_vec, knowledge_base, top_k=3)
    for i, (text, sim) in enumerate(results, 1):
        print(f"  {i}. [{sim:.3f}] {text}")
    print()

print("💡 这就是向量检索的核心原理！")
print("   真实的RAG系统就是这样工作的，只是：")
print("   • 使用更复杂的Embedding模型（768维）")
print("   • 使用向量数据库（支持百万级检索）")
print("   • 相似度计算更优化")
print()

print("="*70)
print("✅ 练习1完成！")
print()
print("💡 关键收获：")
print("   • 向量是一组数字，可以表示文本的语义")
print("   • 余弦相似度用于计算向量的相似程度")
print("   • 通过向量检索可以找到最相关的内容")
print("   • 这是RAG的数学基础！")
print()
print("📍 下一步：python 02_embedding_model.py")
print("   学习使用真实的Embedding模型！")
print("="*70)

