#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5.1.1: 手动对比RAG效果
学习目标：让用户自己输入问题，直观对比有无RAG的差异
"""

import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import os

print("=" * 60)
print("🔬 手动对比：直接问LLM vs 使用RAG")
print("=" * 60)

# ============================================================
# 加载组件
# ============================================================

print("\n📦 加载系统组件...")

# 1. 向量数据库
print("   [1/3] 加载向量数据库...")
db_path = "../step4_vectorstore/data/chroma_traffic_law"

if not os.path.exists(db_path):
    print("❌ 错误：向量数据库不存在！")
    print(f"   请先运行: cd ../step4_vectorstore && python 02_import_traffic_law.py")
    exit(1)

client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="traffic_law")
print(f"         ✅ 数据库包含 {collection.count()} 个文档")

# 2. Embedding模型
print("   [2/3] 加载Embedding模型...")
embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')
print("         ✅ Embedding模型加载完成")

# 3. LLM
print("   [3/3] 加载LLM...")
llm_path = "/Users/a58/llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf"

if not os.path.exists(llm_path):
    print("❌ 错误：LLM模型不存在！")
    print(f"   请检查路径: {llm_path}")
    exit(1)

llm = Llama(
    model_path=llm_path,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False
)
print("         ✅ LLM加载完成")

print("\n✅ 所有组件加载完成！\n")

# ============================================================
# 定义函数
# ============================================================

def ask_llm_directly(question):
    """
    直接问LLM（不使用RAG）
    
    Args:
        question: 用户问题
    
    Returns:
        LLM的直接回答
    """
    prompt = f"{question}\n\n请回答："
    
    output = llm(
        prompt,
        max_tokens=200,
        temperature=0.7,  # 较高温度
        stop=["\n\n"],
        echo=False,
        stream=False
    )
    
    return output['choices'][0]['text'].strip()


def ask_with_rag(question):
    """
    使用RAG回答问题
    
    Args:
        question: 用户问题
    
    Returns:
        tuple: (RAG答案, 检索到的文档)
    """
    # Step 1: 向量化问题
    question_vector = embedding_model.encode([question], show_progress_bar=False)
    
    # Step 2: 检索相关文档（降低阈值，允许更多结果）
    results = collection.query(
        query_embeddings=question_vector.tolist(),
        n_results=5,  # 增加检索数量
        include=["documents", "metadatas", "distances"]
    )
    
    # 格式化检索结果，过滤低相似度（阈值0.5）
    retrieved_docs = []
    for i in range(len(results['ids'][0])):
        similarity = 1 - results['distances'][0][i]
        if similarity >= 0.5:  # 阈值0.5，允许更宽松的匹配
            retrieved_docs.append({
                'content': results['documents'][0][i],
                'chapter': results['metadatas'][0][i]['chapter'],
                'similarity': similarity
            })
    
    # 最多保留3个最相似的
    retrieved_docs = retrieved_docs[:3]
    
    # Step 3: 构建上下文
    context = "\n\n".join([doc['content'] for doc in retrieved_docs])
    
    # Step 4: 生成答案
    prompt = f"""你是一个专业的交通法规助手，专门解答中国道路交通安全法相关问题。

【参考资料】
{context}

【回答规则】
1. 严格依据参考资料回答，不编造信息
2. 如果参考资料中没有相关内容，明确回答「参考资料中未提及此内容」
3. 回答要准确、简洁、分点列出
4. 保持客观中立的语气

【用户问题】
{question}

【你的回答】
"""
    
    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.3,  # 低温度，更确定
        stop=["【", "\n\n\n"],
        echo=False,
        stream=False
    )
    
    answer = output['choices'][0]['text'].strip()
    
    return answer, retrieved_docs, prompt  # 返回prompt用于显示


# ============================================================
# 交互式对比
# ============================================================

print("=" * 60)
print("🎯 使用说明")
print("=" * 60)
print("\n现在你可以输入任何关于交通法的问题，系统会展示两种回答：")
print("   1️⃣  直接问LLM（没有参考资料）")
print("   2️⃣  使用RAG（基于交通法文档）")
print("\n然后你可以对比两者的差异！")
print("\n💡 输入 'exit' 或 'quit' 退出")
print("💡 输入 'examples' 查看示例问题")
print("\n" + "=" * 60)

# 示例问题
example_questions = [
    "酒驾的处罚是什么？",
    "闯红灯要扣几分？",
    "驾驶证扣满12分怎么办？",
    "交通事故后应该怎么处理？",
    "超速50%以上会被怎么处罚？",
]

# 主循环
while True:
    try:
        # 获取用户输入
        print("\n" + "─" * 60)
        user_question = input("\n💬 请输入你的问题: ").strip()
        
        # 处理退出
        if user_question.lower() in ['exit', 'quit', 'bye']:
            print("\n👋 再见！\n")
            break
        
        # 显示示例
        if user_question.lower() == 'examples':
            print("\n📝 示例问题:")
            for i, q in enumerate(example_questions, 1):
                print(f"   {i}. {q}")
            continue
        
        # 处理空输入
        if not user_question:
            print("⚠️  请输入问题")
            continue
        
        print("\n" + "=" * 60)
        print(f"❓ 你的问题: {user_question}")
        print("=" * 60)
        
        # ============================================================
        # 方式1：直接问LLM
        # ============================================================
        
        print("\n" + "━" * 60)
        print("方式1️⃣ : 直接问LLM（不使用RAG）")
        print("━" * 60)
        print("\n🤖 正在生成...")
        
        answer_direct = ask_llm_directly(user_question)
        
        print(f"\n💬 LLM直接回答:")
        print("┌" + "─" * 58 + "┐")
        for line in answer_direct.split('\n'):
            print(f"│ {line:<56} │")
        print("└" + "─" * 58 + "┘")
        
        print("\n📌 特点:")
        print("   • 基于模型自身的训练数据")
        print("   • 可能包含模型的「记忆」或「猜测」")
        print("   • 无法验证答案来源")
        print("   • 信息可能过时或不准确")
        
        # ============================================================
        # 方式2：使用RAG
        # ============================================================
        
        print("\n" + "━" * 60)
        print("方式2️⃣ : 使用RAG（基于交通法文档）")
        print("━" * 60)
        
        print("\n🔍 Step 1: 检索相关文档...")
        answer_rag, retrieved_docs, rag_prompt = ask_with_rag(user_question)
        
        print(f"\n📚 检索到 {len(retrieved_docs)} 个相关文档:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"\n   [{i}] 相似度: {doc['similarity']*100:.1f}% | {doc['chapter']}")
            preview = doc['content'][:80].replace('\n', ' ') + "..." if len(doc['content']) > 80 else doc['content']
            print(f"       预览: {preview}")
        
        print("\n📝 Step 2: 构建Prompt（把检索到的片段塞进去）...")
        print("┌" + "─" * 58 + "┐")
        print("│ 🔍 Prompt内容预览（前500字符）:                         │")
        print("├" + "─" * 58 + "┤")
        prompt_preview = rag_prompt[:500].replace('\n', '\n│ ')
        for line in prompt_preview.split('\n'):
            print(f"│ {line:<56} │")
        if len(rag_prompt) > 500:
            print(f"│ ... (还有 {len(rag_prompt)-500} 字符)                               │")
        print("└" + "─" * 58 + "┘")
        
        print(f"\n   • Prompt总长度: {len(rag_prompt)} 字符")
        print(f"   • 包含的文档片段数: {len(retrieved_docs)} 个")
        
        print("\n🤖 Step 3: LLM基于这个Prompt生成答案...")
        
        print(f"\n💡 RAG回答:")
        print("┌" + "─" * 58 + "┐")
        for line in answer_rag.split('\n'):
            print(f"│ {line:<56} │")
        print("└" + "─" * 58 + "┘")
        
        print("\n📌 特点:")
        print("   • 基于最新的交通法文档")
        print("   • 答案有据可查")
        print("   • 可以追溯到具体章节")
        print("   • 更准确、更可信")
        
        # ============================================================
        # 对比分析
        # ============================================================
        
        print("\n" + "=" * 60)
        print("📊 对比分析")
        print("=" * 60)
        
        print(f"\n直接问LLM:")
        print(f"   长度: {len(answer_direct)} 字符")
        print(f"   温度: 0.7（较随机）")
        print(f"   来源: 模型记忆")
        
        print(f"\n使用RAG:")
        print(f"   长度: {len(answer_rag)} 字符")
        print(f"   温度: 0.3（更确定）")
        print(f"   来源: {len(retrieved_docs)} 个文档片段")
        
        print(f"\n💡 你觉得哪个答案更好？")
        print(f"   • 直接问LLM: 可能流畅但不一定准确")
        print(f"   • 使用RAG: 基于权威文档，更可信")
        
    except KeyboardInterrupt:
        print("\n\n👋 检测到Ctrl+C，退出\n")
        break
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("请重试或输入 'exit' 退出\n")
        continue

# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 60)
print("🎉 对比体验完成！")
print("=" * 60)

print("\n✅ 通过对比，你应该发现了:")
print("   1. LLM直接回答可能「听起来对」，但不一定准确")
print("   2. RAG提供有依据的回答，可以追溯来源")
print("   3. RAG的temperature更低，更忠实于文档")
print("   4. RAG能防止模型「编造」信息")

print("\n💡 RAG的核心价值:")
print("   • 让LLM回答「有据可查」")
print("   • 防止模型幻觉（编造）")
print("   • 可以使用最新的、领域专属的文档")
print("   • 提升答案的准确性和可信度")

print("\n🎯 下一步:")
print("   运行: python 01_basic_rag.py")
print("   或者: python 02_improve_retrieval.py")
print("   继续深入学习RAG优化技巧")

print("\n" + "=" * 60)

