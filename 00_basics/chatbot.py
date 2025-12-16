from llama_cpp import Llama
import sys

# 加载模型
print("正在加载模型...")
llm = Llama(
    model_path="/Users/a58/llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False
)
print("模型加载完成！输入 'exit' 退出\n")

# 对话循环
conversation_history = []

while True:
    # 获取用户输入
    user_input = input("你: ")
    
    if user_input.lower() in ['exit', 'quit', '退出']:
        print("再见！")
        break
    
    # 构建prompt（包含历史）
    conversation_history.append(f"用户: {user_input}")
    prompt = "\n".join(conversation_history) + "\n助手: "
    
    # 流式生成回复
    print("AI: ", end="", flush=True)
    response = ""
    for output in llm(
        prompt,
        max_tokens=256,
        temperature=0.7,
        stop=["用户:", "\n\n"],
        stream=True
    ):
        text = output['choices'][0]['text']
        print(text, end="", flush=True)
        response += text
    
    print("\n")  # 换行
    conversation_history.append(f"助手: {response.strip()}")
    
    # 限制历史长度（避免超出上下文）
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]

