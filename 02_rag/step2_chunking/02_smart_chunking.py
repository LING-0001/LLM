"""
练习2：智能切块
使用LangChain的RecursiveCharacterTextSplitter
按自然边界（段落、句子）切分，效果更好
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 测试文本（同样的Python介绍文本）
test_text = """
Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。Python的设计哲学强调代码的可读性和简洁的语法，尤其是使用空格缩进来表示代码块，而不是使用括号或关键字。

Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。它拥有动态类型系统和自动内存管理功能，还有一个庞大而广泛的标准库。

Python的应用领域非常广泛，包括Web开发、数据分析、人工智能、科学计算、自动化运维等。Django和Flask是最流行的Python Web框架。在数据科学领域，NumPy、Pandas、Matplotlib等库被广泛使用。

在人工智能和机器学习方面，Python有TensorFlow、PyTorch、scikit-learn等强大的库。这些工具使得Python成为AI开发的首选语言。许多大型科技公司如Google、Facebook、Netflix都在大规模使用Python。

Python社区非常活跃，有大量的第三方库和工具。Python Package Index（PyPI）上有超过40万个项目。Python的易学性使它成为初学者学习编程的理想选择，同时它的强大功能也满足了专业开发者的需求。
"""

print("="*70)
print(" "*20 + "智能切块演示")
print("="*70)
print()

print("📚 RecursiveCharacterTextSplitter 工作原理：")
print()
print("   1. 优先按段落分隔符（\\n\\n）切分")
print("   2. 如果段落太大，按句子分隔符（。！？）切分")
print("   3. 如果句子还太大，按逗号（，）切分")
print("   4. 最后才按字符数硬切")
print()
print("   这样可以保持语义的完整性！")
print()

# 创建智能切块器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # 目标块大小
    chunk_overlap=30,    # 重叠大小
    length_function=len,  # 用字符数计算长度
    separators=["\n\n", "\n", "。", "，", " ", ""]  # 分隔符优先级
)

print("="*70)
print("实验1：智能切块效果")
print("="*70)
print()

chunks = text_splitter.split_text(test_text)

print(f"✅ 切分成 {len(chunks)} 块\n")

for i, chunk in enumerate(chunks, 1):
    print(f"【块 {i}】({len(chunk)} 字符)")
    print(chunk)
    print(f"{'─'*70}")
    print()

print("💡 观察：")
print("   • 每块都在完整的句子边界切分")
print("   • 没有切断句子或词语")
print("   • 语义完整，适合检索")
print()

# 对比固定长度切块
print("="*70)
print("实验2：对比固定切块 vs 智能切块")
print("="*70)
print()

# 固定长度切块（简单实现）
def simple_chunk(text, size):
    return [text[i:i+size] for i in range(0, len(text), size-30)]

simple_chunks = simple_chunk(test_text, 200)

print("📊 统计对比：\n")
print(f"{'':20} 固定切块    智能切块")
print(f"{'─'*50}")
print(f"{'块数量':20} {len(simple_chunks):^12} {len(chunks):^12}")
print(f"{'平均块大小':20} {sum(len(c) for c in simple_chunks)/len(simple_chunks):^12.0f} {sum(len(c) for c in chunks)/len(chunks):^12.0f}")
print()

# 检查边界质量
def check_boundary_quality(chunks):
    """检查切块边界的质量"""
    score = 0
    for chunk in chunks:
        # 检查是否以完整句子结尾
        if chunk.rstrip().endswith(('。', '！', '？', '\n')):
            score += 1
    return score / len(chunks) * 100

simple_quality = check_boundary_quality(simple_chunks)
smart_quality = check_boundary_quality(chunks)

print(f"{'边界质量（%）':20} {simple_quality:^12.0f} {smart_quality:^12.0f}")
print()
print("💡 智能切块的边界质量明显更高！")
print()

# 实验3：测试不同参数
print("="*70)
print("实验3：调整chunk_size参数")
print("="*70)
print()

for size in [150, 250, 350]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=int(size * 0.15),  # 15%重叠
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    
    chunks_test = splitter.split_text(test_text)
    avg_size = sum(len(c) for c in chunks_test) / len(chunks_test)
    
    print(f"chunk_size={size:3d} → {len(chunks_test)} 块, 平均大小: {avg_size:.0f} 字符")

print()
print("💡 观察：chunk_size越大，块数越少，但每块信息更完整")
print()

# 实验4：测试overlap的影响
print("="*70)
print("实验4：调整overlap参数")
print("="*70)
print()

chunk_size = 200
for overlap_pct in [0, 10, 20, 30]:
    overlap = int(chunk_size * overlap_pct / 100)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    
    chunks_test = splitter.split_text(test_text)
    
    print(f"overlap={overlap_pct:2d}% ({overlap:2d}字符) → {len(chunks_test)} 块")
    
    if len(chunks_test) >= 2:
        # 检查实际重叠
        chunk1_end = chunks_test[0][-50:]
        chunk2_start = chunks_test[1][:50]
        
        # 简单检查是否有共同部分
        has_overlap = any(word in chunk2_start for word in chunk1_end.split() if len(word) > 2)
        print(f"   块1与块2有重叠: {'✅' if has_overlap else '❌'}")
    
    print()

print("💡 重叠的作用：")
print("   • 保持上下文连续性")
print("   • 避免关键信息在边界丢失")
print("   • 一般设置为chunk_size的10-20%")
print()

# 实用示例：不同类型的文本
print("="*70)
print("实验5：处理不同类型的文本")
print("="*70)
print()

# 示例1：列表型文本
list_text = """
Python的主要特点：

1. 简单易学：语法简洁明了，适合初学者
2. 跨平台：可在Windows、Linux、macOS上运行
3. 丰富的库：拥有海量第三方库
4. 动态类型：无需声明变量类型
5. 解释型语言：开发调试方便

Python的应用场景：

- Web开发：Django、Flask
- 数据科学：Pandas、NumPy
- 人工智能：TensorFlow、PyTorch
- 自动化运维：Ansible、Fabric
- 爬虫：Scrapy、BeautifulSoup
"""

print("📝 列表型文本：")
splitter_list = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    separators=["\n\n", "\n", "：", " ", ""]
)
chunks_list = splitter_list.split_text(list_text)
print(f"   切分成 {len(chunks_list)} 块")
print(f"   块1: {chunks_list[0][:80]}...")
print()

# 示例2：对话型文本
dialog_text = """
用户：Python和Java有什么区别？

助手：主要区别有几点。首先，Python是动态类型语言，Java是静态类型。其次，Python语法更简洁，学习曲线较平缓。

用户：哪个性能更好？

助手：Java在执行速度上通常更快，因为它是编译型语言。但Python开发效率更高，适合快速原型开发。

用户：我应该学哪个？

助手：取决于你的目标。如果做数据科学或AI，选Python。如果做企业级应用，Java更常见。
"""

print("💬 对话型文本：")
splitter_dialog = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", "。", "？", "，"]
)
chunks_dialog = splitter_dialog.split_text(dialog_text)
print(f"   切分成 {len(chunks_dialog)} 块")
print(f"   块1: {chunks_dialog[0][:80]}...")
print()

print("="*70)
print("✅ 练习2完成！")
print()
print("💡 关键收获：")
print("   • RecursiveCharacterTextSplitter按自然边界切分")
print("   • chunk_size控制块大小，overlap保持连续性")
print("   • 可以自定义分隔符优先级")
print("   • 不同类型文本需要不同的切分策略")
print()
print("📍 下一步：python 03_document_chunking.py")
print("   处理真实的文档文件！")
print("="*70)

