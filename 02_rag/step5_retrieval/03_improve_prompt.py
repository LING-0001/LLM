#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5.3: 优化Prompt提升生成质量
学习目标：通过Prompt工程防止幻觉、提升答案质量
"""

import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import os

print("=" * 60)
print("✨ Prompt优化提升RAG质量")
print("=" * 60)

# ============================================================
# 加载组件
# ============================================================

print("\n📦 加载系统组件...")

db_path = "../step4_vectorstore/data/chroma_traffic_law"
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="traffic_law")

embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')

llm = Llama(
    model_path="/Users/a58/llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False
)

print("✅ 所有组件加载完成\n")

# ============================================================
# 检索函数
# ============================================================

def retrieve_documents(question, top_k=3):
    """检索文档"""
    q_vec = embedding_model.encode([question], show_progress_bar=False)
    results = collection.query(
        query_embeddings=q_vec.tolist(),
        n_results=top_k,
        include=["documents"]
    )
    return results['documents'][0]

# ============================================================
# 第一部分：对比不同Prompt模板
# ============================================================

print("【第一部分：Prompt模板对比】")
print("=" * 60)

question = "酒驾的处罚是什么？"
docs = retrieve_documents(question, top_k=2)
context = "\n\n".join(docs)

print(f"\n❓ 测试问题: {question}\n")

# Prompt 1: 极简版（容易出问题）
prompt1 = f"""{context}

问题：{question}
回答："""

print("="*60)
print("Prompt 1: 极简版")
print("="*60)
print(prompt1[:200] + "...")

print("\n🤖 生成答案...")
output1 = llm(prompt1, max_tokens=150, temperature=0.3, stop=["\n\n"], echo=False, stream=False)
answer1 = output1['choices'][0]['text'].strip()
print(f"\n💬 {answer1}")

print("\n⚠️  问题:")
print("   • 没有角色定位")
print("   • 没有明确指令")
print("   • 可能偏离参考资料")

# Prompt 2: 基础版
prompt2 = f"""根据以下参考资料回答问题：

【参考资料】
{context}

【问题】
{question}

【回答】
"""

print("\n" + "="*60)
print("Prompt 2: 基础版")
print("="*60)
print(prompt2[:200] + "...")

print("\n🤖 生成答案...")
output2 = llm(prompt2, max_tokens=150, temperature=0.3, stop=["【"], echo=False, stream=False)
answer2 = output2['choices'][0]['text'].strip()
print(f"\n💬 {answer2}")

print("\n✅ 改进:")
print("   • 明确了「参考资料」")
print("   • 结构清晰")

# Prompt 3: 专业版（推荐）
prompt3 = f"""你是一个交通法规助手，专门解答中国道路交通安全法相关问题。

【参考资料】
{context}

【回答要求】
1. 仅根据参考资料回答，不要编造信息
2. 如果参考资料中没有答案，请明确说明
3. 回答要简洁准确，分点列出要点

【用户问题】
{question}

【你的回答】
"""

print("\n" + "="*60)
print("Prompt 3: 专业版（推荐）")
print("="*60)
print(prompt3[:250] + "...")

print("\n🤖 生成答案...")
output3 = llm(prompt3, max_tokens=200, temperature=0.3, stop=["【"], echo=False, stream=False)
answer3 = output3['choices'][0]['text'].strip()
print(f"\n💬 {answer3}")

print("\n✅ 优势:")
print("   • 明确角色定位（交通法规助手）")
print("   • 详细的回答要求")
print("   • 防止编造信息")
print("   • 格式规范")

# ============================================================
# 第二部分：防止幻觉（编造信息）
# ============================================================

print("\n\n【第二部分：防止LLM幻觉】")
print("=" * 60)

# 用一个文档中没有的问题测试
tricky_question = "高速公路最低限速是多少？"
print(f"\n❓ 测试问题: {tricky_question}")
print("（注意：这个信息可能不在我们的文档中）\n")

docs_tricky = retrieve_documents(tricky_question, top_k=2)
context_tricky = "\n\n".join(docs_tricky)

# 不防幻觉的Prompt
prompt_no_guard = f"""根据参考资料回答：

{context_tricky}

问题：{tricky_question}
回答："""

print("="*60)
print("不防幻觉的Prompt")
print("="*60)

output_no_guard = llm(prompt_no_guard, max_tokens=100, temperature=0.5, stop=["\n\n"], echo=False, stream=False)
answer_no_guard = output_no_guard['choices'][0]['text'].strip()
print(f"\n💬 {answer_no_guard}")

print("\n⚠️  风险: LLM可能基于记忆回答，而不是参考资料")

# 防幻觉的Prompt
prompt_with_guard = f"""你是一个严谨的交通法规助手。

【参考资料】
{context_tricky}

【重要规则】
- 仅根据参考资料回答
- 如果参考资料中没有相关信息，请回答：「参考资料中未提及此内容」
- 绝对不要编造或猜测答案

【问题】
{tricky_question}

【回答】
"""

print("\n" + "="*60)
print("防幻觉的Prompt")
print("="*60)

output_with_guard = llm(prompt_with_guard, max_tokens=100, temperature=0.2, stop=["【"], echo=False, stream=False)
answer_with_guard = output_with_guard['choices'][0]['text'].strip()
print(f"\n💬 {answer_with_guard}")

print("\n✅ 防幻觉策略:")
print("   1. 明确说明「仅根据参考资料」")
print("   2. 提供「未提及」的标准回答")
print("   3. 降低temperature（0.1-0.3）")
print("   4. 强调「不要编造」")

# ============================================================
# 第三部分：带引用的回答
# ============================================================

print("\n\n【第三部分：带引用的回答】")
print("=" * 60)

question_cite = "驾驶证扣12分后怎么办？"
print(f"\n❓ 问题: {question_cite}\n")

docs_cite = retrieve_documents(question_cite, top_k=3)

# 构建带编号的参考资料
context_with_numbers = ""
for i, doc in enumerate(docs_cite, 1):
    context_with_numbers += f"[文档{i}]\n{doc}\n\n"

prompt_with_citation = f"""你是一个交通法规助手。请根据参考资料回答问题，并用[文档X]标注引用来源。

【参考资料】
{context_with_numbers}

【回答要求】
1. 在回答中用[文档1]、[文档2]等标注引用来源
2. 只使用参考资料中的信息
3. 分点回答，每点后面标注来源

【问题】
{question_cite}

【回答】（请在每个要点后标注[文档X]）
"""

print("🤖 生成带引用的答案...")
output_cite = llm(prompt_with_citation, max_tokens=300, temperature=0.3, stop=["【"], echo=False, stream=False)
answer_cite = output_cite['choices'][0]['text'].strip()

print(f"\n💬 {answer_cite}")

print("\n✅ 带引用的优势:")
print("   • 用户可以验证答案来源")
print("   • 提高可信度")
print("   • 便于追溯原文")
print("   • 专业性强")

# ============================================================
# 第四部分：控制回答长度和格式
# ============================================================

print("\n\n【第四部分：控制回答格式】")
print("=" * 60)

question_format = "闯红灯的处罚是什么？"
docs_format = retrieve_documents(question_format, top_k=2)
context_format = "\n\n".join(docs_format)

# 格式1：简洁版
prompt_short = f"""你是交通法规助手，用最简洁的方式回答。

参考资料：
{context_format}

要求：一句话回答，不超过30字。

问题：{question_format}
回答："""

print(f"\n❓ 问题: {question_format}")
print("\n" + "="*60)
print("格式1: 超简洁（一句话，30字内）")
print("="*60)

output_short = llm(prompt_short, max_tokens=50, temperature=0.2, stop=["\n"], echo=False, stream=False)
answer_short = output_short['choices'][0]['text'].strip()
print(f"\n💬 {answer_short}")
print(f"   （{len(answer_short)}字）")

# 格式2：分点列出
prompt_points = f"""你是交通法规助手，请分点回答。

参考资料：
{context_format}

要求：分3点回答，每点一句话。

问题：{question_format}
回答："""

print("\n" + "="*60)
print("格式2: 分点列出（3点）")
print("="*60)

output_points = llm(prompt_points, max_tokens=150, temperature=0.2, stop=["\n\n"], echo=False, stream=False)
answer_points = output_points['choices'][0]['text'].strip()
print(f"\n💬 {answer_points}")

print("\n💡 格式控制技巧:")
print("   • 在Prompt中明确要求长度")
print("   • 指定回答格式（分点/表格/一句话）")
print("   • 使用max_tokens限制")
print("   • 调整stop标记")

# ============================================================
# 第五部分：最佳Prompt模板总结
# ============================================================

print("\n\n【第五部分：最佳Prompt模板】")
print("=" * 60)

best_prompt_template = """你是一个专业的交通法规助手，专门解答中国道路交通安全法相关问题。

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

print("\n✅ 最佳模板要素:")
print("   1. 🎭 角色定位: 「你是XX助手」")
print("   2. 📚 资料标注: 清晰的【参考资料】标签")
print("   3. 📋 回答规则: 详细的约束条件")
print("   4. 🚫 防幻觉: 「不编造」「未提及」")
print("   5. 📝 格式要求: 「分点」「简洁」")

print("\n💡 关键参数:")
print("   • temperature = 0.1-0.3（低温度，更确定）")
print("   • max_tokens = 200-400（控制长度）")
print("   • stop = [\"【\", \"\\n\\n\"]（停止标记）")

# ============================================================
# 总结
# ============================================================

print("\n\n" + "=" * 60)
print("🎉 Prompt优化学习完成！")
print("=" * 60)

print("\n✅ 掌握的技巧:")
print("   1. 对比不同Prompt效果")
print("   2. 防止LLM幻觉")
print("   3. 带引用的回答")
print("   4. 控制回答格式和长度")
print("   5. 最佳Prompt模板")

print("\n💡 核心原则:")
print("   • 明确角色和任务")
print("   • 详细的回答规则")
print("   • 防止编造信息")
print("   • 格式化输出")

print("\n🎯 下一步:")
print("   运行: python 04_complete_rag_system.py")
print("   整合所有优化，构建完整RAG系统")

print("\n" + "=" * 60)

