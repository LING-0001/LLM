"""
实验1：没有RAG的普通LLM
演示：LLM只能回答训练数据中的知识，无法访问你的私有文档
"""

from llama_cpp import Llama

print("="*60)
print("实验1：普通LLM（没有RAG）")
print("="*60)
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

# 测试问题（关于这个项目的私有知识）
questions = [
    "MyLLM项目的学习路线是什么？",
    "这个项目有哪几个学习阶段？",
    "RAG学习部分包含哪些步骤？"
]

print("📌 测试场景：询问本项目的私有信息")
print("   （LLM的训练数据中不包含这些信息）\n")

for i, question in enumerate(questions, 1):
    print(f"{'─'*60}")
    print(f"问题 {i}: {question}")
    print(f"{'─'*60}")
    print("回答: ", end="", flush=True)
    
    # 直接问LLM，不提供任何文档
    for output in llm(
        question,
        max_tokens=200,
        temperature=0.7,
        stream=True
    ):
        print(output['choices'][0]['text'], end="", flush=True)
    
    print("\n")

print()
print("="*60)
print("📊 观察结果")
print("="*60)
print()
print("❌ 问题：")
print("   - LLM无法准确回答关于本项目的问题")
print("   - 回答可能含糊不清或完全错误")
print("   - 可能会编造不存在的信息（幻觉）")
print()
print("💡 原因：")
print("   - LLM只知道训练时见过的数据")
print("   - 我们的项目文档不在训练数据中")
print("   - 没有任何外部知识来源")
print()
print("✅ 解决方案：")
print("   - 使用RAG！让LLM能够检索我们的项目文档")
print("   - 运行 02_with_rag.py 查看改进效果")
print()
print("="*60)

