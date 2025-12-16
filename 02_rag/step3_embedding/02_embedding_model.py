"""
练习2：使用Embedding模型
学习如何使用真实的中文Embedding模型把文本转成向量
"""

from sentence_transformers import SentenceTransformer
import numpy as np

print("="*70)
print(" "*20 + "使用Embedding模型")
print("="*70)
print()

# 1. 加载模型
print("="*70)
print("1. 加载中文Embedding模型")
print("="*70)
print()

print("正在加载模型：text2vec-base-chinese")
print("（第一次运行会自动下载，约400MB，需要几分钟）")
print()

try:
    model = SentenceTransformer('shibing624/text2vec-base-chinese')
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败：{e}")
    print()
    print("解决方案：")
    print("1. 确保已安装：pip install sentence-transformers")
    print("2. 检查网络连接")
    print("3. 或使用其他模型：paraphrase-multilingual-MiniLM-L12-v2")
    exit(1)

print()
print("模型信息：")
print(f"  • 模型名称：text2vec-base-chinese")
print(f"  • 向量维度：{model.get_sentence_embedding_dimension()}")
print(f"  • 支持语言：中文")
print()

# 2. 文本转向量
print("="*70)
print("2. 把文本转换成向量")
print("="*70)
print()

text = "Python是一种编程语言"
print(f"原始文本：{text}")
print()

# 转换成向量
vector = model.encode(text)

print(f"向量维度：{len(vector)}")
print(f"向量类型：{type(vector)}")
print()

print("向量内容（前10个元素）：")
print(vector[:10])
print()

print("向量内容（后10个元素）：")
print(vector[-10:])
print()

# 向量的统计信息
print("向量统计：")
print(f"  • 最小值：{vector.min():.4f}")
print(f"  • 最大值：{vector.max():.4f}")
print(f"  • 平均值：{vector.mean():.4f}")
print(f"  • 标准差：{vector.std():.4f}")
print()

# 3. 批量转换
print("="*70)
print("3. 批量转换多个文本")
print("="*70)
print()

texts = [
    "Python是一种编程语言",
    "Java也是编程语言",
    "我喜欢吃苹果",
    "机器学习很有趣",
]

print("要转换的文本：")
for i, text in enumerate(texts, 1):
    print(f"  {i}. {text}")
print()

print("正在批量转换...")
vectors = model.encode(texts)
print(f"✅ 完成！得到 {len(vectors)} 个向量")
print(f"   向量形状：{vectors.shape}")
print()

# 4. 计算相似度
print("="*70)
print("4. 计算文本相似度")
print("="*70)
print()

from sklearn.metrics.pairwise import cosine_similarity

# 计算所有文本两两之间的相似度
similarity_matrix = cosine_similarity(vectors)

print("相似度矩阵：")
print()
print(f"{'':25}", end="")
for i, text in enumerate(texts):
    print(f"{i+1:8}", end="")
print()

for i, text in enumerate(texts):
    print(f"{i+1}. {text:20}", end="")
    for j in range(len(texts)):
        print(f"{similarity_matrix[i][j]:8.3f}", end="")
    print()

print()
print("💡 观察：")
print("   • 对角线都是1.000（自己和自己完全相同）")
print("   • 句子1和2相似度高（都是编程语言）")
print("   • 句子3和其他差异大（不同主题）")
print()

# 5. 找最相似的文本
print("="*70)
print("5. 实战：找到最相似的文本")
print("="*70)
print()

query = "学习编程"
print(f"问题：{query}")
print()

# 把问题也转成向量
query_vector = model.encode(query)

# 计算问题和每个文本的相似度
similarities = cosine_similarity([query_vector], vectors)[0]

print("与各文本的相似度：")
for text, sim in zip(texts, similarities):
    bar = "█" * int(sim * 50)
    print(f"  [{sim:.3f}] {bar} {text}")

print()

# 找到最相似的
best_idx = similarities.argmax()
print(f"🎯 最相似的文本：{texts[best_idx]}")
print(f"   相似度：{similarities[best_idx]:.3f}")
print()

# 6. 不同文本的对比
print("="*70)
print("6. 测试：语义理解能力")
print("="*70)
print()

test_cases = [
    {
        "query": "如何学Python",
        "candidates": [
            "Python学习指南",
            "Java开发教程",
            "Python入门方法",
            "天气预报",
        ]
    },
    {
        "query": "天气怎么样",
        "candidates": [
            "今天天气很好",
            "Python很强大",
            "气候变化研究",
            "明天会下雨",
        ]
    }
]

for case in test_cases:
    query = case["query"]
    candidates = case["candidates"]
    
    print(f"问题：{query}")
    print()
    
    # 编码
    query_vec = model.encode(query)
    cand_vecs = model.encode(candidates)
    
    # 计算相似度
    sims = cosine_similarity([query_vec], cand_vecs)[0]
    
    # 排序
    ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)
    
    print("排序结果（按相似度）：")
    for i, (text, sim) in enumerate(ranked, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"  {emoji} [{sim:.3f}] {text}")
    print()

print("💡 观察：")
print("   • Embedding模型能理解语义，不只是关键词匹配")
print("   • '如何学Python'和'Python学习指南'相似度高")
print("   • '天气怎么样'能匹配到'今天天气很好'")
print()

# 7. 保存和加载向量
print("="*70)
print("7. 保存向量（避免重复计算）")
print("="*70)
print()

# 保存
vectors_file = "sample_vectors.npy"
np.save(vectors_file, vectors)
print(f"✅ 向量已保存到：{vectors_file}")

# 加载
loaded_vectors = np.load(vectors_file)
print(f"✅ 向量已加载，形状：{loaded_vectors.shape}")
print()

print("💡 实际应用中：")
print("   • 文档向量只需计算一次，保存起来")
print("   • 查询时只需把问题转成向量")
print("   • 大大节省计算时间！")
print()

# 清理临时文件
import os
os.remove(vectors_file)
print(f"（已删除临时文件：{vectors_file}）")
print()

# 8. 性能测试
print("="*70)
print("8. 性能测试")
print("="*70)
print()

import time

# 单个文本
text = "这是一个测试句子"
start = time.time()
vec = model.encode(text)
time_single = time.time() - start

print(f"单个文本编码：{time_single*1000:.2f}ms")

# 批量
texts_batch = [f"这是第{i}个测试句子" for i in range(100)]
start = time.time()
vecs = model.encode(texts_batch)
time_batch = time.time() - start

print(f"100个文本批量编码：{time_batch*1000:.2f}ms")
print(f"平均每个：{time_batch/100*1000:.2f}ms")
print()

print(f"💡 批量处理快 {time_single*100/time_batch:.1f} 倍！")
print()

print("="*70)
print("✅ 练习2完成！")
print()
print("💡 关键收获：")
print("   • 学会使用SentenceTransformer加载Embedding模型")
print("   • 文本 → 向量转换（encode方法）")
print("   • 批量处理比单个处理快很多")
print("   • 向量可以保存，避免重复计算")
print()
print("📍 下一步：python 03_text_similarity.py")
print("   深入学习相似度计算和应用！")
print("="*70)

