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

examples = [
    {
        "title": "代码审查助手",
        "prompt": """请审查以下Python代码，指出潜在问题：

```python
def calculate_average(numbers):
    sum = 0
    for num in numbers:
        sum = sum + num
    return sum / len(numbers)
```

请指出：
1. 代码逻辑是否正确
2. 是否有潜在的错误
3. 如何改进""",
        "temp": 0.3
    },
    {
        "title": "文案生成",
        "prompt": """为一款AI学习助手App写一段吸引人的产品介绍（50字以内）：

产品特点：
- 个性化学习路径
- 实时答疑
- 智能复习提醒

要求：突出优势，语言简洁有力""",
        "temp": 0.8
    },
    {
        "title": "数据分析",
        "prompt": """请分析以下销售数据并给出建议：

1月销售额: 100万
2月销售额: 95万
3月销售额: 88万

请：
1. 分析趋势
2. 找出可能的原因
3. 提出3条改进建议""",
        "temp": 0.5
    }
]

for i, example in enumerate(examples, 1):
    print(f"\n{'='*60}")
    print(f"示例{i}: {example['title']}")
    print(f"{'='*60}\n")
    print(f"提示词:\n{example['prompt']}\n")
    print("回答: ", end="", flush=True)
    
    for output in llm(
        example['prompt'],
        max_tokens=300,
        temperature=example['temp'],
        stream=True
    ):
        print(output['choices'][0]['text'], end="", flush=True)
    
    print("\n")

