"""
准备交通法文档数据
1. 读取交通法文档
2. 智能分块
3. 向量化
4. 保存结果供后续步骤使用
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import json

print("="*70)
print(" "*15 + "交通法文档数据准备")
print("="*70)
print()

# 1. 读取文档
print("步骤1：读取文档")
print("─"*70)

with open('traffic_law_document.md', 'r', encoding='utf-8') as f:
    document = f.read()

print(f"✅ 文档读取成功")
print(f"   文件：traffic_law_document.md")
print(f"   长度：{len(document)} 字符")
print(f"   约：{len(document)//500} 个段落")
print()

# 2. 智能分块
print("步骤2：智能分块")
print("─"*70)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " ", ""]
)

chunks = text_splitter.split_text(document)

print(f"✅ 分块完成")
print(f"   块数：{len(chunks)} 块")
print(f"   平均大小：{sum(len(c) for c in chunks) / len(chunks):.0f} 字符")
print()

print("前3个块预览：")
for i, chunk in enumerate(chunks[:3], 1):
    preview = chunk[:80].replace('\n', ' ')
    print(f"   块{i}: {preview}...")
print()

# 3. 向量化
print("步骤3：向量化（Embedding）")
print("─"*70)

print("正在加载Embedding模型...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')
print("✅ 模型加载完成")
print()

print("正在批量生成向量...")
vectors = model.encode(chunks, batch_size=32, show_progress_bar=True)
print(f"✅ 向量生成完成")
print(f"   向量数量：{len(vectors)}")
print(f"   向量维度：{vectors.shape[1]}")
print()

# 4. 准备元数据
print("步骤4：准备元数据")
print("─"*70)

chunks_with_metadata = []
for i, chunk in enumerate(chunks):
    # 确定来源章节
    if "第一章" in chunk or "通行规则" in chunk:
        chapter = "第一章：基本通行规则"
    elif "第二章" in chunk or "交通信号" in chunk:
        chapter = "第二章：交通信号和标志"
    elif "第三章" in chunk or "驾驶证" in chunk:
        chapter = "第三章：机动车驾驶证管理"
    elif "第四章" in chunk or "交通事故" in chunk:
        chapter = "第四章：交通事故处理"
    elif "第五章" in chunk or "法律责任" in chunk:
        chapter = "第五章：法律责任与处罚"
    else:
        chapter = "未分类"
    
    chunks_with_metadata.append({
        "id": f"chunk_{i:03d}",
        "content": chunk,
        "chapter": chapter,
        "length": len(chunk),
        "index": i
    })

print(f"✅ 元数据准备完成")
print()

# 5. 保存结果
print("步骤5：保存结果")
print("─"*70)

# 保存向量
vectors_file = "data/traffic_law_vectors.npy"
np.save(vectors_file, vectors)
print(f"✅ 向量已保存：{vectors_file}")

# 保存完整数据包
data_package = {
    "source_file": "traffic_law_document.md",
    "model_name": "shibing624/text2vec-base-chinese",
    "chunk_size": 400,
    "chunk_overlap": 60,
    "num_chunks": len(chunks),
    "vector_dim": int(vectors.shape[1]),
    "chunks": chunks_with_metadata
}

package_file = "data/traffic_law_data.json"
with open(package_file, 'w', encoding='utf-8') as f:
    json.dump(data_package, f, ensure_ascii=False, indent=2)
print(f"✅ 数据包已保存：{package_file}")
print()

# 6. 数据统计
print("步骤6：数据统计")
print("─"*70)

from collections import Counter
chapter_counts = Counter(item["chapter"] for item in chunks_with_metadata)

print("各章节块数分布：")
for chapter, count in sorted(chapter_counts.items()):
    bar = "█" * (count * 2)
    print(f"   {chapter:25} {bar} ({count}块)")
print()

# 块大小分布
sizes = [item["length"] for item in chunks_with_metadata]
print("块大小统计：")
print(f"   最小：{min(sizes)} 字符")
print(f"   最大：{max(sizes)} 字符")
print(f"   平均：{np.mean(sizes):.0f} 字符")
print(f"   中位数：{np.median(sizes):.0f} 字符")
print()

# 向量内存占用
vector_memory = vectors.nbytes / 1024 / 1024
print(f"向量内存占用：{vector_memory:.2f} MB")
print()

# 7. 简单测试
print("步骤7：快速测试检索")
print("─"*70)

from sklearn.metrics.pairwise import cosine_similarity

test_query = "酒驾怎么处罚"
print(f"测试问题：{test_query}")
print()

query_vector = model.encode(test_query)
similarities = cosine_similarity([query_vector], vectors)[0]

top_3_indices = similarities.argsort()[-3:][::-1]

print("最相关的3个块：")
for rank, idx in enumerate(top_3_indices, 1):
    chunk_info = chunks_with_metadata[idx]
    score = similarities[idx]
    preview = chunk_info["content"][:60].replace('\n', ' ')
    
    print(f"{rank}. [{score:.3f}] {chunk_info['chapter']}")
    print(f"   {preview}...")
    print()

print("="*70)
print("✅ 数据准备完成！")
print()
print("📊 总结：")
print(f"   • 原始文档：1篇 (约1600字)")
print(f"   • 分块数量：{len(chunks)} 块")
print(f"   • 向量维度：{vectors.shape[1]}")
print(f"   • 数据文件：")
print(f"     - {vectors_file}")
print(f"     - {package_file}")
print()
print("📍 这些数据将在Step 4和Step 5中使用")
print("="*70)

