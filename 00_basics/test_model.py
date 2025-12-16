from llama_cpp import Llama

# 加载模型（纯CPU模式）
print("正在加载模型...")
llm = Llama(
    model_path="/Users/a58/llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=2048,       # 上下文长度
    n_threads=4,      # CPU线程数
    n_gpu_layers=0,   # 0 = 纯CPU模式
    verbose=False     # 不显示加载信息
)
print("模型加载完成！\n")

# 测试对话
prompt = "你好，你有什么技能。"

print(f"问题: {prompt}")
print("回答: ", end="", flush=True)

# 流式生成回复（像打字机一样逐字输出）
token_count = 0
for output in llm(
    prompt,
    max_tokens=128,
    temperature=0.7,
    stop=["</s>", "\n\n"],
    stream=True  # 开启流式输出
):
    text = output['choices'][0]['text']
    print(text, end="", flush=True)
    token_count += 1

print(f"\n\n生成了 {token_count} 个token")

