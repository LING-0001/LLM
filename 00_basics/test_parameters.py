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

prompt = "写一首关于编程的五言绝句"

print(f"提示词: {prompt}\n")

# 测试不同的温度值
temperatures = [0.1, 0.5, 0.9]

for temp in temperatures:
    print(f"{'='*50}")
    print(f"Temperature = {temp}")
    print(f"{'='*50}")
    
    response = ""
    for output in llm(
        prompt,
        max_tokens=100,
        temperature=temp,
        top_p=0.9,
        repeat_penalty=1.1,
        stream=True
    ):
        text = output['choices'][0]['text']
        print(text, end="", flush=True)
        response += text
    
    print("\n")

