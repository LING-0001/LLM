#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5.1: 基础RAG问答系统
学习目标：整合检索和生成，构建完整的RAG流程
"""

import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import os

print("=" * 60)
print("🤖 基础RAG问答系统")
print("=" * 60)

# ============================================================
# 第一部分：加载组件
# ============================================================

print("\n【第一部分：加载系统组件】")
print("-" * 60)

# 1. 加载向量数据库
print("\n1️⃣ 加载向量数据库...")
db_path = "../step4_vectorstore/data/chroma_traffic_law"

if not os.path.exists(db_path):
    print("❌ 错误：向量数据库不存在！")
    print(f"   请先运行: cd ../step4_vectorstore && python 02_import_traffic_law.py")
    exit(1)

client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="traffic_law")
print(f"   ✅ 数据库加载成功，包含 {collection.count()} 个文档")

# 2. 加载Embedding模型
print("\n2️⃣ 加载Embedding模型...")
embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')
print("   ✅ Embedding模型加载完成")

# 3. 加载LLM
print("\n3️⃣ 加载LLM...")
llm_path = "/Users/a58/llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf"

if not os.path.exists(llm_path):
    print("❌ 错误：LLM模型不存在！")
    print(f"   请检查路径: {llm_path}")
    exit(1)

llm = Llama(
    model_path=llm_path,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,  # CPU模式
    verbose=False
)
print("   ✅ LLM加载完成")

# ============================================================
# 第二部分：实现RAG核心函数
# ============================================================

print("\n【第二部分：RAG核心函数】")
print("-" * 60)

def retrieve_documents(question, top_k=3):
    """
    检索相关文档
    
    Args:
        question: 用户问题
        top_k: 返回Top-K个文档
    
    Returns:
        检索到的文档列表
    """
    # 1. 向量化问题
    question_vector = embedding_model.encode([question], show_progress_bar=False)
    
    # 2. 检索
    results = collection.query(
        query_embeddings=question_vector.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # 3. 格式化结果
    retrieved_docs = []
    for i in range(len(results['ids'][0])):
        retrieved_docs.append({
            'id': results['ids'][0][i],
            'content': results['documents'][0][i],
            'chapter': results['metadatas'][0][i]['chapter'],
            'similarity': 1 - results['distances'][0][i]
        })
    
    return retrieved_docs


def generate_answer(question, context):
    """
    基于上下文生成答案
    
    Args:
        question: 用户问题
        context: 检索到的上下文
    
    Returns:
        LLM生成的答案
    """
    # 构建Prompt
    prompt = f"""根据以下参考资料回答问题：

【参考资料】
{context}

【问题】
{question}

【回答】
"""
    
    # LLM生成
    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.3,  # 低温度，更确定
        stop=["【", "\n\n"],
        echo=False,
        stream=False
    )
    
    answer = output['choices'][0]['text'].strip()
    return answer


def rag_query(question, top_k=3, show_retrieval=True):
    """
    完整的RAG查询流程
    
    Args:
        question: 用户问题
        top_k: 检索文档数量
        show_retrieval: 是否显示检索结果
    
    Returns:
        答案
    """
    print(f"\n{'='*60}")
    print(f"❓ 问题: {question}")
    print(f"{'='*60}")
    
    # Step 1: 检索
    print(f"\n🔍 检索相关文档（Top-{top_k}）...")
    retrieved_docs = retrieve_documents(question, top_k)
    
    if show_retrieval:
        print(f"\n📚 检索到的文档:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"\n[文档{i}] 相似度: {doc['similarity']*100:.1f}% | {doc['chapter']}")
            preview = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
            print(f"   {preview}")
    
    # Step 2: 构建上下文
    context = "\n\n".join([doc['content'] for doc in retrieved_docs])
    
    # Step 3: 生成答案
    print(f"\n🤖 生成答案...")
    answer = generate_answer(question, context)
    
    print(f"\n💡 答案:")
    print(f"{'='*60}")
    print(answer)
    print(f"{'='*60}")
    
    return answer

print("✅ RAG函数定义完成")

# ============================================================
# 第三部分：测试RAG系统
# ============================================================

print("\n【第三部分：测试RAG系统】")
print("-" * 60)

# 测试问题列表
test_questions = [
    "酒驾会受到什么处罚？",
    "闯红灯要扣多少分？",
    "新手司机实习期有什么规定？",
    "交通事故后应该怎么处理？",
    "超速行驶如何处罚？"
]

print(f"\n将测试 {len(test_questions)} 个问题...")
print("\n" + "🔔 提示：首次生成可能较慢，请耐心等待")

# 只测试前2个问题（完整测试太慢）
for idx, question in enumerate(test_questions[:2], 1):
    print(f"\n\n{'#'*60}")
    print(f"# 测试 {idx}/{len(test_questions[:2])}")
    print(f"{'#'*60}")
    
    answer = rag_query(question, top_k=3, show_retrieval=True)

# ============================================================
# 第四部分：对比有无RAG的效果
# ============================================================

print("\n\n" + "=" * 60)
print("【第四部分：对比有无RAG的效果】")
print("=" * 60)

comparison_question = "醉酒驾驶会被判刑吗？"

# 无RAG：直接问LLM
print(f"\n❓ 问题: {comparison_question}")
print(f"\n{'='*60}")
print("方式1：不使用RAG（LLM直接回答）")
print(f"{'='*60}")

prompt_no_rag = f"{comparison_question}\n回答："

output_no_rag = llm(
    prompt_no_rag,
    max_tokens=128,
    temperature=0.7,
    stop=["\n\n"],
    echo=False,
    stream=False
)

answer_no_rag = output_no_rag['choices'][0]['text'].strip()
print(f"\n💬 {answer_no_rag}")

# 使用RAG
print(f"\n{'='*60}")
print("方式2：使用RAG（基于交通法文档）")
print(f"{'='*60}")

retrieved_docs = retrieve_documents(comparison_question, top_k=2)
context = "\n\n".join([doc['content'] for doc in retrieved_docs])
answer_with_rag = generate_answer(comparison_question, context)

print(f"\n💡 {answer_with_rag}")

# 对比分析
print(f"\n{'='*60}")
print("📊 对比分析")
print(f"{'='*60}")
print("\n无RAG:")
print("  • 可能基于模型记忆回答")
print("  • 信息可能过时或不准确")
print("  • 缺乏依据")

print("\n使用RAG:")
print("  • 基于最新的交通法文档")
print("  • 答案有据可查")
print("  • 更准确、更可信")

# ============================================================
# 第五部分：分析RAG流程
# ============================================================

print("\n" + "=" * 60)
print("【第五部分：RAG流程分析】")
print("=" * 60)

test_q = "驾驶证扣满12分怎么办？"

print(f"\n📋 详细流程演示")
print(f"问题: {test_q}")
print("-" * 60)

# Step 1
print("\n[Step 1] 向量化问题")
q_vec = embedding_model.encode([test_q], show_progress_bar=False)
print(f"   • 问题: {test_q}")
print(f"   • 向量维度: {q_vec.shape[1]}")
print(f"   • 向量示例: [{q_vec[0][:5].tolist()}...]")

# Step 2
print("\n[Step 2] 检索相关文档")
docs = retrieve_documents(test_q, top_k=2)
print(f"   • 检索到 {len(docs)} 个文档")
for i, doc in enumerate(docs, 1):
    print(f"   • 文档{i}: 相似度={doc['similarity']:.2%}, 章节={doc['chapter']}")

# Step 3
print("\n[Step 3] 构建Prompt")
context = "\n\n".join([doc['content'] for doc in docs])
prompt = f"""根据以下参考资料回答问题：

【参考资料】
{context[:200]}...（省略）

【问题】
{test_q}

【回答】
"""
print(f"   • Prompt长度: {len(prompt)} 字符")
print(f"   • 上下文长度: {len(context)} 字符")

# Step 4
print("\n[Step 4] LLM生成答案")
answer = generate_answer(test_q, context)
print(f"   • 答案长度: {len(answer)} 字符")
print(f"   • 答案: {answer[:100]}...")

# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 60)
print("🎉 基础RAG系统演示完成！")
print("=" * 60)

print("\n✅ 你已经学会:")
print("   1. 加载向量数据库和LLM")
print("   2. 实现检索函数")
print("   3. 实现生成函数")
print("   4. 整合完整RAG流程")
print("   5. 对比有无RAG的效果")

print("\n💡 RAG核心流程:")
print("   问题 → 向量化 → 检索 → 构建Prompt → LLM生成 → 答案")

print("\n📊 观察结果:")
print("   • RAG能提供准确的、有依据的答案")
print("   • 检索质量直接影响答案质量")
print("   • Prompt设计很重要")

print("\n🔧 可优化的地方:")
print("   • Top-K选择（3 vs 5 vs 10）")
print("   • 相似度阈值过滤")
print("   • Prompt模板优化")
print("   • Temperature调整")

print("\n🎯 下一步:")
print("   运行: python 02_improve_retrieval.py")
print("   学习如何优化检索效果")

print("\n" + "=" * 60)

