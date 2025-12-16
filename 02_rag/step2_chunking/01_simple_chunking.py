"""
练习1：固定长度切块
最简单的切块方法，按字符数固定切分
"""

def simple_chunk(text, chunk_size=200, overlap=50):
    """
    固定长度切块函数
    
    参数：
        text: 要切分的文本
        chunk_size: 每块的大小（字符数）
        overlap: 重叠部分的大小
    
    返回：
        chunks: 切分后的文本块列表
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # 切出一块
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # 移动到下一块的起始位置（考虑重叠）
        start += (chunk_size - overlap)
    
    return chunks


# 测试文本
test_text = """
Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。Python的设计哲学强调代码的可读性和简洁的语法，尤其是使用空格缩进来表示代码块，而不是使用括号或关键字。

Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。它拥有动态类型系统和自动内存管理功能，还有一个庞大而广泛的标准库。

Python的应用领域非常广泛，包括Web开发、数据分析、人工智能、科学计算、自动化运维等。Django和Flask是最流行的Python Web框架。在数据科学领域，NumPy、Pandas、Matplotlib等库被广泛使用。

在人工智能和机器学习方面，Python有TensorFlow、PyTorch、scikit-learn等强大的库。这些工具使得Python成为AI开发的首选语言。许多大型科技公司如Google、Facebook、Netflix都在大规模使用Python。

Python社区非常活跃，有大量的第三方库和工具。Python Package Index（PyPI）上有超过40万个项目。Python的易学性使它成为初学者学习编程的理想选择，同时它的强大功能也满足了专业开发者的需求。
"""

print("="*70)
print(" "*20 + "固定长度切块演示")
print("="*70)
print()

print("📄 原始文本：")
print(f"   长度：{len(test_text)} 字符")
print(f"   内容预览：{test_text[:100]}...")
print()

# 实验1：不同的chunk_size
print("="*70)
print("实验1：测试不同的 chunk_size（不重叠）")
print("="*70)
print()

for chunk_size in [100, 200, 300]:
    chunks = simple_chunk(test_text, chunk_size=chunk_size, overlap=0)
    print(f"📊 chunk_size={chunk_size}, overlap=0")
    print(f"   切分成 {len(chunks)} 块")
    print(f"   第1块: {chunks[0][:80]}...")
    print(f"   第2块: {chunks[1][:80]}..." if len(chunks) > 1 else "")
    print()

# 实验2：重叠的效果
print("="*70)
print("实验2：测试重叠（overlap）的效果")
print("="*70)
print()

chunk_size = 200
for overlap in [0, 30, 60]:
    chunks = simple_chunk(test_text, chunk_size=chunk_size, overlap=overlap)
    print(f"📊 chunk_size={chunk_size}, overlap={overlap}")
    print(f"   切分成 {len(chunks)} 块")
    
    if len(chunks) >= 2:
        # 显示第1块和第2块的重叠部分
        chunk1_end = chunks[0][-50:]
        chunk2_start = chunks[1][:50]
        
        print(f"   第1块结尾: ...{chunk1_end}")
        print(f"   第2块开头: {chunk2_start}...")
        
        # 计算重叠内容
        if overlap > 0:
            print(f"   💡 注意：两块之间有 {overlap} 字符的重叠")
    print()

# 实验3：观察切块边界问题
print("="*70)
print("实验3：观察固定长度切块的问题")
print("="*70)
print()

chunks = simple_chunk(test_text, chunk_size=150, overlap=20)

print(f"切分成 {len(chunks)} 块，来看看边界处的问题：\n")

for i, chunk in enumerate(chunks[:3], 1):  # 只显示前3块
    print(f"【块 {i}】({len(chunk)} 字符)")
    print(chunk)
    print(f"{'─'*70}")
    print()

print("❌ 观察到的问题：")
print("   1. 可能在句子中间切断")
print("   2. 可能在词语中间切断")
print("   3. 语义不完整")
print()
print("✅ 解决方案：")
print("   使用智能切块（按句子、段落等自然边界切分）")
print("   → 运行 02_smart_chunking.py 查看改进方法")
print()

# 练习：计算一些统计信息
print("="*70)
print("📊 切块统计信息")
print("="*70)
print()

chunk_size = 200
overlap = 30
chunks = simple_chunk(test_text, chunk_size=chunk_size, overlap=overlap)

print(f"配置：chunk_size={chunk_size}, overlap={overlap}")
print(f"原文长度：{len(test_text)} 字符")
print(f"切分块数：{len(chunks)} 块")
print(f"平均块大小：{sum(len(c) for c in chunks) / len(chunks):.1f} 字符")
print(f"最小块：{min(len(c) for c in chunks)} 字符")
print(f"最大块：{max(len(c) for c in chunks)} 字符")
print()

# 显示每块的大小分布
print("各块大小：")
for i, chunk in enumerate(chunks, 1):
    bar = "█" * (len(chunk) // 10)
    print(f"   块{i:2d}: {bar} ({len(chunk)} 字符)")
print()

print("="*70)
print("✅ 练习1完成！")
print()
print("💡 关键收获：")
print("   • 理解了固定长度切块的原理")
print("   • 看到了chunk_size和overlap的作用")
print("   • 发现了固定切块的局限性（切断句子）")
print()
print("📍 下一步：python 02_smart_chunking.py")
print("   学习更好的切块方法！")
print("="*70)

