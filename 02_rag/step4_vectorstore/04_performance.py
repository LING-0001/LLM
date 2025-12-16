#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4.4: 性能优化
学习目标：优化向量数据库的导入和检索性能
"""

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import os

print("=" * 60)
print("⚡ ChromaDB性能优化")
print("=" * 60)

# ============================================================
# 第一部分：批量导入性能测试
# ============================================================

print("\n【第一部分：批量导入性能测试】")
print("-" * 60)

# 加载模型
print(f"\n📦 加载embedding模型...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')
print("   ✅ 模型加载完成")

# 准备测试数据（100个文档）
num_docs = 100
test_docs = [f"这是第{i}个测试文档，内容关于交通法规和驾驶安全" for i in range(num_docs)]

print(f"\n🔄 生成 {num_docs} 个测试向量...")
test_embeddings = model.encode(test_docs, show_progress_bar=False)
print(f"   ✅ 向量生成完成: {test_embeddings.shape}")

# 创建临时数据库
db_path = "./data/chroma_performance_test"
os.makedirs(db_path, exist_ok=True)
client = chromadb.PersistentClient(path=db_path)

# 测试不同的批量大小
batch_sizes = [1, 10, 25, 50, 100]

print(f"\n⏱️  测试不同批量大小的导入速度:")
print("-" * 60)

results = []

for batch_size in batch_sizes:
    # 删除旧集合
    try:
        client.delete_collection(name=f"test_batch_{batch_size}")
    except:
        pass
    
    # 创建新集合
    collection = client.create_collection(name=f"test_batch_{batch_size}")
    
    # 开始计时
    start_time = time.time()
    
    # 批量导入
    for i in range(0, num_docs, batch_size):
        end_idx = min(i + batch_size, num_docs)
        batch_embeddings = test_embeddings[i:end_idx].tolist()
        batch_docs = test_docs[i:end_idx]
        batch_ids = [f"doc_{j}" for j in range(i, end_idx)]
        
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_docs,
            ids=batch_ids
        )
    
    # 结束计时
    end_time = time.time()
    elapsed = (end_time - start_time) * 1000  # 转为毫秒
    
    results.append({
        'batch_size': batch_size,
        'time': elapsed,
        'docs_per_sec': num_docs / (elapsed / 1000)
    })
    
    print(f"   批量大小 {batch_size:3d}: {elapsed:6.2f}ms ({results[-1]['docs_per_sec']:.1f} docs/s)")

# 最佳批量大小
best = min(results, key=lambda x: x['time'])
print(f"\n✅ 最佳批量大小: {best['batch_size']} (速度: {best['docs_per_sec']:.1f} docs/s)")

# ============================================================
# 第二部分：检索性能测试
# ============================================================

print("\n【第二部分：检索性能测试】")
print("-" * 60)

# 使用真实数据库
real_db_path = "./data/chroma_traffic_law"
if not os.path.exists(real_db_path):
    print("\n⚠️  请先运行 02_import_traffic_law.py 创建数据库")
else:
    real_client = chromadb.PersistentClient(path=real_db_path)
    real_collection = real_client.get_collection(name="traffic_law")
    
    print(f"\n📊 数据库规模: {real_collection.count()} 个文档")
    
    # 测试问题
    test_questions = [
        "酒驾处罚标准",
        "闯红灯扣分",
        "超速罚款金额",
        "交通事故处理",
        "驾驶证扣分"
    ]
    
    print(f"\n⏱️  测试检索速度（每个问题运行10次）:")
    print("-" * 60)
    
    for k in [1, 5, 10, 20]:
        all_times = []
        
        for question in test_questions:
            query_embedding = model.encode([question], show_progress_bar=False)
            
            # 运行10次取平均
            times = []
            for _ in range(10):
                start = time.time()
                results = real_collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=k,
                    include=["distances"]
                )
                end = time.time()
                times.append((end - start) * 1000)
            
            all_times.extend(times)
        
        avg_time = np.mean(all_times)
        std_time = np.std(all_times)
        
        print(f"   Top-{k:2d}: {avg_time:5.2f}ms ± {std_time:4.2f}ms")
    
    print("\n💡 观察:")
    print("   • 检索速度与Top-K关系不大")
    print("   • ChromaDB的ANN算法优化了搜索")
    print("   • 即使Top-20也只需几毫秒")

# ============================================================
# 第三部分：内存使用分析
# ============================================================

print("\n【第三部分：内存使用分析】")
print("-" * 60)

import psutil
import gc

# 获取当前进程
process = psutil.Process(os.getpid())

# 初始内存
gc.collect()
mem_before = process.memory_info().rss / 1024 / 1024  # MB

print(f"\n💾 初始内存: {mem_before:.1f} MB")

# 加载大量数据
print(f"\n🔄 创建大规模测试数据...")
large_num = 1000
large_docs = [f"测试文档{i}" * 10 for i in range(large_num)]  # 每个文档约100字符
large_embeddings = model.encode(large_docs, show_progress_bar=True, batch_size=64)

# 测量内存增长
gc.collect()
mem_after = process.memory_info().rss / 1024 / 1024  # MB

print(f"\n💾 生成向量后的内存: {mem_after:.1f} MB")
print(f"📈 内存增长: {mem_after - mem_before:.1f} MB")

# 计算向量占用的理论内存
# 768维 * 4字节(float32) * 1000个向量
vector_size_mb = (768 * 4 * large_num) / 1024 / 1024
print(f"\n📊 向量理论大小: {vector_size_mb:.1f} MB")
print(f"📊 实际内存增长: {mem_after - mem_before:.1f} MB")

# 导入到ChromaDB
try:
    client.delete_collection(name="memory_test")
except:
    pass

collection = client.create_collection(name="memory_test")

print(f"\n💾 导入前内存: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# 批量导入
batch_size = 100
for i in range(0, large_num, batch_size):
    end_idx = min(i + batch_size, large_num)
    collection.add(
        embeddings=large_embeddings[i:end_idx].tolist(),
        documents=large_docs[i:end_idx],
        ids=[f"doc_{j}" for j in range(i, end_idx)]
    )

gc.collect()
mem_final = process.memory_info().rss / 1024 / 1024

print(f"💾 导入后内存: {mem_final:.1f} MB")
print(f"📈 ChromaDB额外占用: {mem_final - mem_after:.1f} MB")

# ============================================================
# 第四部分：向量化性能优化
# ============================================================

print("\n【第四部分：向量化性能优化】")
print("-" * 60)

# 测试不同batch_size对向量化速度的影响
num_texts = 100
test_texts = [f"测试文本{i}，关于交通安全和法规" for i in range(num_texts)]

batch_sizes = [1, 8, 16, 32, 64]

print(f"\n⏱️  测试不同batch_size的向量化速度:")
print("-" * 60)

for batch_size in batch_sizes:
    start = time.time()
    embeddings = model.encode(test_texts, batch_size=batch_size, show_progress_bar=False)
    end = time.time()
    
    elapsed = (end - start) * 1000
    texts_per_sec = num_texts / (elapsed / 1000)
    
    print(f"   batch_size {batch_size:2d}: {elapsed:6.2f}ms ({texts_per_sec:.1f} texts/s)")

print("\n💡 建议:")
print("   • CPU: batch_size=16-32")
print("   • GPU: batch_size=64-128")
print("   • 根据内存大小调整")

# ============================================================
# 第五部分：持久化 vs 内存模式
# ============================================================

print("\n【第五部分：持久化 vs 内存模式】")
print("-" * 60)

# 测试数据
test_num = 50
test_data_docs = [f"文档{i}" for i in range(test_num)]
test_data_embeddings = model.encode(test_data_docs, show_progress_bar=False)

# 测试1：持久化模式
print(f"\n⏱️  持久化模式:")
persistent_client = chromadb.PersistentClient(path="./data/chroma_persistent_test")
try:
    persistent_client.delete_collection(name="test")
except:
    pass

start = time.time()
persistent_collection = persistent_client.create_collection(name="test")
persistent_collection.add(
    embeddings=test_data_embeddings.tolist(),
    documents=test_data_docs,
    ids=[f"doc_{i}" for i in range(test_num)]
)
persistent_time = (time.time() - start) * 1000

print(f"   导入时间: {persistent_time:.2f}ms")

# 测试2：内存模式
print(f"\n⏱️  内存模式:")
memory_client = chromadb.Client()

start = time.time()
memory_collection = memory_client.create_collection(name="test")
memory_collection.add(
    embeddings=test_data_embeddings.tolist(),
    documents=test_data_docs,
    ids=[f"doc_{i}" for i in range(test_num)]
)
memory_time = (time.time() - start) * 1000

print(f"   导入时间: {memory_time:.2f}ms")

# 对比
print(f"\n📊 性能对比:")
print(f"   内存模式: {memory_time:.2f}ms")
print(f"   持久化模式: {persistent_time:.2f}ms")
print(f"   速度差异: {persistent_time/memory_time:.2f}x")

print(f"\n💡 选择建议:")
print(f"   • 开发测试: 用内存模式（快速）")
print(f"   • 生产环境: 用持久化模式（可靠）")

# ============================================================
# 第六部分：优化建议总结
# ============================================================

print("\n【第六部分：优化建议总结】")
print("=" * 60)

print("\n✅ 导入优化:")
print("   1. 使用批量导入（batch_size=50-100）")
print("   2. 预先生成所有向量，然后一次性导入")
print("   3. 避免频繁的小批量添加")

print("\n✅ 检索优化:")
print("   1. 合理设置Top-K（通常5-10就够）")
print("   2. 使用元数据过滤缩小范围")
print("   3. 批量检索多个问题")

print("\n✅ 向量化优化:")
print("   1. 使用合适的batch_size（16-64）")
print("   2. 如果有GPU，开启GPU加速")
print("   3. 缓存常用文本的向量")

print("\n✅ 内存优化:")
print("   1. 及时释放不用的向量（del, gc.collect()）")
print("   2. 大规模数据分批处理")
print("   3. 使用float16减少内存（如果精度允许）")

print("\n✅ 数据库选择:")
print("   • < 10万向量: ChromaDB ✅")
print("   • 10万-100万: ChromaDB 可以")
print("   • > 100万: 考虑Milvus/Qdrant")

# ============================================================
# 清理
# ============================================================

print("\n【清理测试数据】")
print("-" * 60)

# 删除测试集合
test_collections = ["memory_test", "test"]
for coll_name in test_collections:
    try:
        client.delete_collection(name=coll_name)
        print(f"🗑️  删除集合: {coll_name}")
    except:
        pass

# 删除批量测试集合
for batch_size in batch_sizes:
    try:
        client.delete_collection(name=f"test_batch_{batch_size}")
        print(f"🗑️  删除集合: test_batch_{batch_size}")
    except:
        pass

print("\n✅ 清理完成")

# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 60)
print("🎉 性能优化学习完成！")
print("=" * 60)

print("\n✅ 掌握的知识:")
print("   1. 批量导入性能测试")
print("   2. 检索性能分析")
print("   3. 内存使用优化")
print("   4. 向量化速度优化")
print("   5. 持久化 vs 内存模式对比")

print("\n🎓 Step 4 完成！")
print("   你已经掌握了ChromaDB的:")
print("   • 基础操作（增删改查）")
print("   • 高级检索（过滤、排序）")
print("   • 性能优化（批量、内存）")

print("\n🎯 下一步:")
print("   进入 Step 5: 检索与生成（RAG完整流程）")
print("   cd ../step5_retrieval")
print("   cat README.md")

print("\n" + "=" * 60)

