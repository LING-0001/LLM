"""
练习4：批量向量化和优化
学习如何高效处理大量文本，为向量数据库做准备
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import time
import json

print("="*70)
print(" "*20 + "批量向量化和优化")
print("="*70)
print()

# 加载模型
print("正在加载模型...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')
print("✅ 模型加载完成\n")

# 1. 生成测试数据
print("="*70)
print("1. 生成模拟文档数据")
print("="*70)
print()

# 模拟从Step 2切块后得到的文档块
document_chunks = []
for i in range(100):
    chunk = {
        "id": f"chunk_{i:03d}",
        "content": f"这是第{i}个文档块，包含关于Python、机器学习、数据科学的内容。" * 3,
        "source": f"document_{i//10}.txt",
        "chunk_index": i % 10
    }
    document_chunks.append(chunk)

print(f"✅ 生成了 {len(document_chunks)} 个文档块")
print()
print("示例块：")
print(json.dumps(document_chunks[0], ensure_ascii=False, indent=2))
print()

# 2. 单个 vs 批量编码
print("="*70)
print("2. 性能对比：单个 vs 批量编码")
print("="*70)
print()

# 准备数据
test_texts = [chunk["content"] for chunk in document_chunks[:50]]

# 方法1：逐个编码
print("方法1：逐个编码（不推荐）")
start = time.time()
vectors_individual = []
for text in test_texts:
    vec = model.encode(text)
    vectors_individual.append(vec)
time_individual = time.time() - start
print(f"  耗时：{time_individual:.2f}秒")
print()

# 方法2：批量编码
print("方法2：批量编码（推荐）")
start = time.time()
vectors_batch = model.encode(test_texts, batch_size=32)
time_batch = time.time() - start
print(f"  耗时：{time_batch:.2f}秒")
print()

print(f"💡 批量编码快 {time_individual/time_batch:.1f}倍！")
print()

# 3. 测试不同的batch_size
print("="*70)
print("3. 优化batch_size")
print("="*70)
print()

batch_sizes = [8, 16, 32, 64]
print("测试不同的batch_size：")
print()

for bs in batch_sizes:
    start = time.time()
    vectors = model.encode(test_texts, batch_size=bs)
    elapsed = time.time() - start
    
    throughput = len(test_texts) / elapsed
    print(f"  batch_size={bs:2d}: {elapsed:.2f}秒, {throughput:.1f} texts/sec")

print()
print("💡 建议：")
print("   • CPU：batch_size=16-32")
print("   • GPU：batch_size=64-128")
print("   • 内存充足可以更大")
print()

# 4. 显示进度条
print("="*70)
print("4. 处理大量数据时显示进度")
print("="*70)
print()

from tqdm import tqdm

all_texts = [chunk["content"] for chunk in document_chunks]

print("编码所有文档块（带进度条）：")
vectors = model.encode(
    all_texts, 
    batch_size=32,
    show_progress_bar=True
)

print(f"✅ 完成！生成了 {len(vectors)} 个向量")
print(f"   向量形状：{vectors.shape}")
print()

# 5. 保存向量和元数据
print("="*70)
print("5. 保存向量和元数据")
print("="*70)
print()

# 方式1：只保存向量
vectors_file = "document_vectors.npy"
np.save(vectors_file, vectors)
print(f"✅ 向量已保存：{vectors_file}")

# 方式2：保存向量+元数据
data_package = {
    "vectors": vectors.tolist(),  # 转成list才能JSON序列化
    "metadata": document_chunks,
    "model_name": "shibing624/text2vec-base-chinese",
    "vector_dim": vectors.shape[1],
    "count": len(vectors)
}

package_file = "vectors_with_metadata.json"
with open(package_file, 'w', encoding='utf-8') as f:
    json.dump(data_package, f, ensure_ascii=False, indent=2)
print(f"✅ 向量+元数据已保存：{package_file}")
print()

# 6. 加载和验证
print("="*70)
print("6. 加载和验证")
print("="*70)
print()

# 加载向量
loaded_vectors = np.load(vectors_file)
print(f"✅ 从 {vectors_file} 加载了 {len(loaded_vectors)} 个向量")

# 加载完整数据
with open(package_file, 'r', encoding='utf-8') as f:
    loaded_package = json.load(f)

print(f"✅ 从 {package_file} 加载了完整数据")
print(f"   向量数量：{loaded_package['count']}")
print(f"   向量维度：{loaded_package['vector_dim']}")
print(f"   使用模型：{loaded_package['model_name']}")
print()

# 验证
reconstructed_vectors = np.array(loaded_package["vectors"])
print("验证向量一致性...")
if np.allclose(vectors, reconstructed_vectors):
    print("  ✅ 向量完全一致")
else:
    print("  ❌ 向量不一致")
print()

# 7. 增量更新
print("="*70)
print("7. 增量更新向量库")
print("="*70)
print()

print("场景：新增3个文档块")

# 新文档
new_chunks = [
    {"id": "chunk_100", "content": "新增的第1个文档块"},
    {"id": "chunk_101", "content": "新增的第2个文档块"},
    {"id": "chunk_102", "content": "新增的第3个文档块"},
]

# 生成新向量
new_texts = [c["content"] for c in new_chunks]
new_vectors = model.encode(new_texts)

# 合并
all_vectors = np.vstack([vectors, new_vectors])
all_chunks = document_chunks + new_chunks

print(f"✅ 更新完成")
print(f"   原向量数：{len(vectors)}")
print(f"   新向量数：{len(new_vectors)}")
print(f"   总向量数：{len(all_vectors)}")
print()

# 8. 内存和存储分析
print("="*70)
print("8. 内存和存储分析")
print("="*70)
print()

import os

# 计算内存占用
vector_memory = all_vectors.nbytes / 1024 / 1024  # MB

print(f"向量内存占用：")
print(f"  • 向量数量：{len(all_vectors)}")
print(f"  • 每个向量：{all_vectors.shape[1]} 维")
print(f"  • 数据类型：{all_vectors.dtype}")
print(f"  • 总内存：{vector_memory:.2f} MB")
print()

# 文件大小
file_size = os.path.getsize(vectors_file) / 1024 / 1024
json_size = os.path.getsize(package_file) / 1024 / 1024

print(f"文件大小：")
print(f"  • {vectors_file}: {file_size:.2f} MB")
print(f"  • {package_file}: {json_size:.2f} MB")
print()

# 预估
print("💡 预估（100万文档块）：")
scale_factor = 1000000 / len(all_vectors)
estimated_memory = vector_memory * scale_factor
print(f"  • 内存需求：约 {estimated_memory:.0f} MB ({estimated_memory/1024:.1f} GB)")
print(f"  • 推荐使用向量数据库（ChromaDB）优化存储和检索")
print()

# 9. 实战：简单的向量检索
print("="*70)
print("9. 实战：向量检索")
print("="*70)
print()

from sklearn.metrics.pairwise import cosine_similarity

def vector_search(query, vectors, chunks, top_k=3):
    """简单的向量检索"""
    # 问题向量化
    query_vec = model.encode(query)
    
    # 计算相似度
    similarities = cosine_similarity([query_vec], vectors)[0]
    
    # 排序
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "chunk_id": chunks[idx]["id"],
            "content": chunks[idx]["content"][:50] + "...",
            "score": similarities[idx],
            "source": chunks[idx]["source"]
        })
    return results

# 测试检索
query = "Python和机器学习"
print(f"查询：{query}")
print()

results = vector_search(query, all_vectors, all_chunks, top_k=5)

print("检索结果：")
for i, res in enumerate(results, 1):
    print(f"{i}. [{res['score']:.3f}] {res['chunk_id']}")
    print(f"   {res['content']}")
    print(f"   来源：{res['source']}")
    print()

# 清理临时文件
print("清理临时文件...")
os.remove(vectors_file)
os.remove(package_file)
print("✅ 清理完成")
print()

print("="*70)
print("✅ 练习4完成！")
print()
print("💡 关键收获：")
print("   • 批量编码比单个快10-20倍")
print("   • batch_size要根据硬件调整")
print("   • 向量可以保存和增量更新")
print("   • 100万向量约需要3GB内存")
print("   • 简单的numpy数组就能实现基本检索")
print()
print("🎉 Step 3 全部完成！")
print()
print("📊 Step 3 总结：")
print("   ✅ 理解了向量和向量化的原理")
print("   ✅ 学会使用Embedding模型")
print("   ✅ 掌握了相似度计算和应用")
print("   ✅ 能够高效处理大量文本")
print()
print("📍 下一步：Step 4 - 向量数据库（ChromaDB）")
print("   命令：cd ../step4_vectorstore && cat README.md")
print()
print("💡 为什么需要向量数据库？")
print("   • numpy数组：百万向量检索慢")
print("   • 向量数据库：优化了检索速度（ANN算法）")
print("   • ChromaDB：轻量级，易用，完美适合学习")
print()
print("="*70)

