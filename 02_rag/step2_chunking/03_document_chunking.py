"""
练习3：处理真实文档
读取TXT/Markdown文件，智能切块并保存结果
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json

print("="*70)
print(" "*20 + "真实文档切块")
print("="*70)
print()

# 创建示例文档
sample_doc = """# MyLLM 项目学习指南

## 项目简介

MyLLM是一个从零开始学习大语言模型的项目，包含RAG和Fine-tuning两大核心内容。

## 学习路线

### 阶段1：RAG学习

RAG（检索增强生成）是一种让LLM能够访问外部知识的技术。

#### Step 1: 理解RAG原理

学习RAG的基本概念和工作流程。通过对比实验，直观感受RAG的价值。

#### Step 2: 文本切块

将长文档切分成合适的小块。学习三种切块方法：固定长度、按分隔符、智能语义切块。

#### Step 3: 向量化

将文本转换为向量表示。学习Embedding模型的使用，计算文本相似度。

#### Step 4: 向量数据库

使用ChromaDB存储和检索向量。学习如何高效管理大量文档。

#### Step 5: 检索与生成

整合检索和生成流程，构建完整的RAG系统。

### 阶段2：Fine-tuning学习

Fine-tuning是用自己的数据训练模型的过程。

#### Step 1: 理解微调原理

学习什么是微调，LoRA和QLoRA的区别。

#### Step 2: 数据准备

构造高质量的训练数据集。

#### Step 3: 训练过程

使用Unsloth或LLaMA-Factory进行模型微调。

## 最佳实践

### RAG最佳实践

1. 选择合适的chunk_size（300-500字）
2. 设置15-20%的overlap
3. 使用高质量的Embedding模型
4. 优化检索策略

### Fine-tuning最佳实践

1. 准备高质量数据（至少1000条）
2. 使用LoRA降低训练成本
3. 仔细选择训练参数
4. 评估模型效果

## 总结

通过系统学习RAG和Fine-tuning，你将掌握大语言模型的核心应用技术。
"""

# 保存示例文档
doc_path = "sample_document.md"
with open(doc_path, 'w', encoding='utf-8') as f:
    f.write(sample_doc)

print(f"📄 已创建示例文档：{doc_path}")
print(f"   文件大小：{len(sample_doc)} 字符")
print()

# 读取文档
print("="*70)
print("步骤1：读取文档")
print("="*70)
print()

with open(doc_path, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"✅ 成功读取文档")
print(f"   总长度：{len(content)} 字符")
print(f"   行数：{content.count(chr(10)) + 1} 行")
print()

# 智能切块
print("="*70)
print("步骤2：智能切块")
print("="*70)
print()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", "。", " ", ""]
)

chunks = text_splitter.split_text(content)

print(f"✅ 切块完成")
print(f"   块数：{len(chunks)} 块")
print(f"   平均大小：{sum(len(c) for c in chunks) / len(chunks):.0f} 字符")
print()

# 展示切块结果
print("="*70)
print("步骤3：查看切块结果")
print("="*70)
print()

for i, chunk in enumerate(chunks[:5], 1):  # 只显示前5块
    print(f"【块 {i}】({len(chunk)} 字符)")
    print(chunk[:150] + ("..." if len(chunk) > 150 else ""))
    print(f"{'─'*70}")
    print()

if len(chunks) > 5:
    print(f"... 还有 {len(chunks) - 5} 块\n")

# 添加元数据
print("="*70)
print("步骤4：添加元数据")
print("="*70)
print()

chunks_with_metadata = []
for i, chunk in enumerate(chunks):
    chunk_data = {
        "chunk_id": i,
        "content": chunk,
        "length": len(chunk),
        "source": doc_path,
        "metadata": {
            "has_heading": chunk.strip().startswith("#"),
            "is_code": "```" in chunk,
            "has_list": any(chunk.strip().startswith(marker) for marker in ["- ", "1. ", "* "]),
            "char_start": sum(len(chunks[j]) for j in range(i)),
            "char_end": sum(len(chunks[j]) for j in range(i+1))
        }
    }
    chunks_with_metadata.append(chunk_data)

print("✅ 已为每个块添加元数据")
print()
print("示例（块1的元数据）：")
print(json.dumps(chunks_with_metadata[0]["metadata"], indent=2, ensure_ascii=False))
print()

# 保存结果
print("="*70)
print("步骤5：保存切块结果")
print("="*70)
print()

output_file = "chunks_output.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)

print(f"✅ 切块结果已保存到：{output_file}")
print()

# 统计分析
print("="*70)
print("📊 切块统计分析")
print("="*70)
print()

# 块大小分布
sizes = [len(c) for c in chunks]
print("块大小分布：")
print(f"   最小：{min(sizes)} 字符")
print(f"   最大：{max(sizes)} 字符")
print(f"   平均：{sum(sizes)/len(sizes):.1f} 字符")
print(f"   中位数：{sorted(sizes)[len(sizes)//2]} 字符")
print()

# 内容类型统计
has_heading = sum(1 for c in chunks_with_metadata if c["metadata"]["has_heading"])
has_code = sum(1 for c in chunks_with_metadata if c["metadata"]["is_code"])
has_list = sum(1 for c in chunks_with_metadata if c["metadata"]["has_list"])

print("内容类型统计：")
print(f"   包含标题的块：{has_heading} ({has_heading/len(chunks)*100:.1f}%)")
print(f"   包含代码的块：{has_code} ({has_code/len(chunks)*100:.1f}%)")
print(f"   包含列表的块：{has_list} ({has_list/len(chunks)*100:.1f}%)")
print()

# 可视化块大小
print("块大小可视化：")
for i, size in enumerate(sizes, 1):
    bar = "█" * (size // 15)
    print(f"   块{i:2d}: {bar} ({size})")
print()

# 实用函数：搜索块
print("="*70)
print("实用功能：搜索特定内容的块")
print("="*70)
print()

def search_chunks(chunks_data, keyword):
    """搜索包含关键词的块"""
    results = []
    for chunk_data in chunks_data:
        if keyword.lower() in chunk_data["content"].lower():
            results.append({
                "chunk_id": chunk_data["chunk_id"],
                "preview": chunk_data["content"][:100] + "...",
                "length": chunk_data["length"]
            })
    return results

# 搜索示例
keywords = ["RAG", "Fine-tuning", "最佳实践"]
for keyword in keywords:
    results = search_chunks(chunks_with_metadata, keyword)
    print(f"🔍 搜索 '{keyword}'：找到 {len(results)} 个块")
    if results:
        print(f"   示例：块{results[0]['chunk_id']} - {results[0]['preview']}")
    print()

# 质量检查
print("="*70)
print("🔍 切块质量检查")
print("="*70)
print()

def check_chunk_quality(chunks):
    """检查切块质量"""
    issues = []
    
    for i, chunk in enumerate(chunks):
        # 检查是否太短
        if len(chunk) < 50:
            issues.append(f"块{i}: 太短（{len(chunk)}字符）")
        
        # 检查是否太长
        if len(chunk) > 600:
            issues.append(f"块{i}: 太长（{len(chunk)}字符）")
        
        # 检查是否以不完整句子结尾
        if not chunk.rstrip().endswith(('。', '！', '？', '\n', '#')):
            if len(chunk) > 100:  # 只检查较长的块
                issues.append(f"块{i}: 可能在句子中间切断")
    
    return issues

issues = check_chunk_quality(chunks)

if issues:
    print("⚠️  发现以下问题：")
    for issue in issues[:5]:  # 只显示前5个问题
        print(f"   {issue}")
    if len(issues) > 5:
        print(f"   ... 还有 {len(issues)-5} 个问题")
else:
    print("✅ 切块质量良好，未发现明显问题")

print()

# 清理临时文件（可选）
cleanup = input("是否删除示例文件？(y/n): ").strip().lower()
if cleanup == 'y':
    os.remove(doc_path)
    os.remove(output_file)
    print(f"✅ 已删除 {doc_path} 和 {output_file}")
else:
    print(f"📁 文件保留在当前目录")

print()
print("="*70)
print("✅ 练习3完成！")
print()
print("💡 关键收获：")
print("   • 学会读取和处理真实文档")
print("   • 为切块添加有用的元数据")
print("   • 掌握切块质量分析方法")
print("   • 能够搜索和管理切块结果")
print()
print("📍 下一步：python 04_chunk_optimization.py")
print("   学习如何优化切块参数！")
print("="*70)

