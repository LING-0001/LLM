"""
实验3：直观对比RAG和普通LLM
并排展示两种方式的回答差异
"""

from llama_cpp import Llama

print("="*70)
print(" "*20 + "RAG vs 普通LLM 对比实验")
print("="*70)
print()

# 加载模型
print("正在加载模型...")
llm = Llama(
    model_path="/Users/a58/llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False
)
print("模型加载完成！\n")

# 项目文档（模拟从向量数据库检索）
knowledge_base = """
MyLLM项目包含3个学习阶段：

阶段2：RAG（检索增强生成）
- 学习如何让LLM访问外部文档
- 包含5个步骤：理解原理、文本切块、向量化、向量数据库、检索生成
- 最终项目：构建文档问答系统
- 预计时间：7-10天

阶段3：Fine-tuning（模型微调）
- 学习如何用自己的数据训练模型
- 包含3个步骤：理解原理、数据准备、训练过程
- 最终项目：微调领域模型
- 预计时间：10-15天
"""

# 测试问题
question = "RAG学习需要多长时间？包含哪些内容？"

print("📌 测试问题：")
print(f"   {question}\n")

# 方式1：普通LLM
print("┌" + "─"*66 + "┐")
print("│" + " "*15 + "方式1：普通LLM（没有文档）" + " "*16 + "│")
print("└" + "─"*66 + "┘")
print()
print("回答：", end="", flush=True)

answer_normal = ""
for output in llm(
    question,
    max_tokens=150,
    temperature=0.7,
    stream=True
):
    text = output['choices'][0]['text']
    print(text, end="", flush=True)
    answer_normal += text

print("\n\n")

# 方式2：RAG
print("┌" + "─"*66 + "┐")
print("│" + " "*12 + "方式2：使用RAG（提供文档）" + " "*17 + "│")
print("└" + "─"*66 + "┘")
print()

prompt_rag = f"""请根据以下文档回答问题：

【文档】
{knowledge_base}

【问题】
{question}

【回答】（只根据文档内容回答，简洁明确）
"""

print("回答：", end="", flush=True)

answer_rag = ""
for output in llm(
    prompt_rag,
    max_tokens=200,
    temperature=0.3,
    stream=True
):
    text = output['choices'][0]['text']
    print(text, end="", flush=True)
    answer_rag += text

print("\n\n")

# 总结对比
print("="*70)
print(" "*28 + "对比总结")
print("="*70)
print()

print("┌─────────────┬──────────────────────────────────────────────┐")
print("│  对比维度   │              差异说明                        │")
print("├─────────────┼──────────────────────────────────────────────┤")
print("│  准确性     │ 普通LLM：可能不准确或编造                    │")
print("│             │ RAG：基于真实文档，准确可靠                  │")
print("├─────────────┼──────────────────────────────────────────────┤")
print("│  时效性     │ 普通LLM：只知道训练时的数据                  │")
print("│             │ RAG：可以使用最新文档                        │")
print("├─────────────┼──────────────────────────────────────────────┤")
print("│  私有知识   │ 普通LLM：无法访问                            │")
print("│             │ RAG：可以访问你的私有文档                    │")
print("├─────────────┼──────────────────────────────────────────────┤")
print("│  可追溯性   │ 普通LLM：无法说明信息来源                    │")
print("│             │ RAG：可以引用具体文档段落                    │")
print("├─────────────┼──────────────────────────────────────────────┤")
print("│  成本       │ 普通LLM：低（直接推理）                      │")
print("│             │ RAG：稍高（需要检索+推理）                   │")
print("└─────────────┴──────────────────────────────────────────────┘")

print()
print("💡 核心发现：")
print()
print("   ✅ RAG让LLM能够访问外部知识")
print("   ✅ 回答更准确、可靠、可追溯")
print("   ✅ 适合企业知识库、文档问答等场景")
print("   ✅ 无需重新训练模型，只需准备文档")
print()
print("📚 RAG适用场景：")
print()
print("   1️⃣  企业内部知识库（规章制度、技术文档）")
print("   2️⃣  学术论文助手（检索和总结论文）")
print("   3️⃣  法律咨询助手（查找相关法条）")
print("   4️⃣  客户支持系统（产品手册、FAQ）")
print("   5️⃣  个人知识管理（笔记、文档整理）")
print()
print("="*70)
print()
print("✅ Step 1 完成！你已经理解了RAG的核心价值")
print()
print("📍 下一步：Step 2 - 学习文本切块（Chunking）")
print("   命令：cd ../step2_chunking && cat README.md")
print()
print("="*70)

