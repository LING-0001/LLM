"""
实验2：使用RAG的LLM
演示：通过检索相关文档，LLM可以准确回答私有知识问题
"""

from llama_cpp import Llama

print("="*60)
print("实验2：使用RAG的LLM")
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

# 模拟的知识库（项目文档片段）
# 在真实RAG中，这些会从向量数据库检索得到
knowledge_base = """
【项目文档】

MyLLM项目学习路线：

阶段0：基础准备（已完成）
- 环境配置（Conda + Python 3.10）
- 模型下载（Qwen2.5-3B）
- 基本推理测试

阶段1：推理方式对比（已完成）
- llama.cpp 推理
- Ollama 推理
- HuggingFace Transformers 推理

阶段2：RAG - 检索增强生成（当前学习）
包含5个步骤：
Step 1: 理解RAG原理
Step 2: 文本切块（Chunking）
Step 3: 向量化（Embedding）
Step 4: 向量数据库
Step 5: 检索与生成
最终项目：文档问答系统

阶段3：Fine-tuning - 模型微调（后续）
包含3个步骤：
Step 1: 理解微调原理
Step 2: 数据准备
Step 3: 训练过程
最终项目：领域模型微调
"""

# 测试问题
questions = [
    "MyLLM项目的学习路线是什么？",
    "这个项目有哪几个学习阶段？",
    "RAG学习部分包含哪些步骤？"
]

print("📌 测试场景：使用RAG回答本项目的问题")
print("   （从项目文档中检索相关信息）\n")

for i, question in enumerate(questions, 1):
    print(f"{'─'*60}")
    print(f"问题 {i}: {question}")
    print(f"{'─'*60}")
    
    # RAG的核心：把检索到的文档和问题组合
    prompt = f"""请根据以下文档回答问题。

【参考文档】
{knowledge_base}

【问题】
{question}

【要求】
- 只根据文档内容回答
- 如果文档中没有相关信息，明确说明
- 回答要简洁准确

【回答】
"""
    
    print("回答: ", end="", flush=True)
    
    for output in llm(
        prompt,
        max_tokens=300,
        temperature=0.3,  # 降低温度，让回答更贴近文档
        stream=True
    ):
        print(output['choices'][0]['text'], end="", flush=True)
    
    print("\n")

print()
print("="*60)
print("📊 观察结果")
print("="*60)
print()
print("✅ 改进：")
print("   - LLM能够准确回答关于本项目的问题")
print("   - 回答基于真实的项目文档")
print("   - 不会编造不存在的信息")
print()
print("💡 关键点：")
print("   - 我们把相关文档和问题一起给了LLM")
print("   - 这就是RAG的核心思想！")
print("   - 在真实系统中，文档是自动检索的")
print()
print("🔍 真实RAG的完整流程：")
print("   1. 把所有文档切块并向量化（准备阶段）")
print("   2. 用户提问")
print("   3. 把问题向量化")
print("   4. 在向量数据库中搜索最相关的文档")
print("   5. 把文档和问题组合成提示词")
print("   6. LLM生成答案")
print()
print("📝 对比实验1和实验2，感受RAG的价值！")
print()
print("="*60)

