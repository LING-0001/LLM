from llama_cpp import Llama

print("正在加载模型...")
llm = Llama(
    model_path="/Users/a58/llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False
)
print("模型加载完成！\n")

# 技巧1: 角色设定
print("="*60)
print("技巧1: 角色设定")
print("="*60)

prompt1 = """你是一位资深的Python程序员，擅长用简单的语言解释复杂概念。

请解释什么是装饰器？"""

print(f"提示词:\n{prompt1}\n")
print("回答: ", end="", flush=True)

for output in llm(prompt1, max_tokens=200, temperature=0.7, stream=True):
    print(output['choices'][0]['text'], end="", flush=True)

print("\n\n")

# 技巧2: 分步骤思考
print("="*60)
print("技巧2: 分步骤思考（Chain of Thought）")
print("="*60)

prompt2 = """请一步一步思考并解决这个问题：

问题：一个水池有进水管和出水管，进水管每小时注入10升水，出水管每小时排出3升水。
如果水池初始是空的，5小时后水池有多少升水？

请按照以下步骤：
1. 列出已知条件
2. 计算净增加速度
3. 计算最终结果"""

print(f"提示词:\n{prompt2}\n")
print("回答: ", end="", flush=True)

for output in llm(prompt2, max_tokens=300, temperature=0.3, stream=True):
    print(output['choices'][0]['text'], end="", flush=True)

print("\n\n")

# 技巧3: Few-shot Learning（提供示例）
print("="*60)
print("技巧3: Few-shot Learning（提供示例）")
print("="*60)

prompt3 = """请将以下句子改写成更正式的表达方式：

示例1:
原句: 这个东西真不错
正式: 该产品质量优良

示例2:
原句: 我觉得可以试试
正式: 我认为此方案值得尝试

示例3:
原句: 他挺厉害的
正式: 他的能力较为出众

现在轮到你了:
原句: 这代码写得太乱了
正式:"""

print(f"提示词:\n{prompt3}\n")
print("回答: ", end="", flush=True)

for output in llm(prompt3, max_tokens=50, temperature=0.5, stream=True):
    print(output['choices'][0]['text'], end="", flush=True)

print("\n\n")

# 技巧4: 设置输出格式
print("="*60)
print("技巧4: 设置输出格式")
print("="*60)

prompt4 = """请根据以下信息生成一份产品评测，使用JSON格式输出：

产品：MacBook Pro M1
使用体验：性能强劲，续航出色，屏幕显示效果好

输出格式：
{
  "product": "产品名称",
  "rating": "评分(1-5)",
  "pros": ["优点1", "优点2"],
  "cons": ["缺点1", "缺点2"],
  "summary": "总结"
}"""

print(f"提示词:\n{prompt4}\n")
print("回答: ", end="", flush=True)

for output in llm(prompt4, max_tokens=300, temperature=0.5, stream=True):
    print(output['choices'][0]['text'], end="", flush=True)

print("\n\n")

