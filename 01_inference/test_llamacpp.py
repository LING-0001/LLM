"""
llama.cpp Python绑定测试脚本

使用前确保：
1. 已安装llama-cpp-python: pip install llama-cpp-python
2. 已下载GGUF模型

这个脚本展示如何使用llama-cpp-python进行推理
"""

import os
import time
from pathlib import Path


def find_gguf_models():
    """查找可用的GGUF模型"""
    possible_paths = [
        Path.home() / "llama.cpp" / "models",
        Path.home() / ".ollama" / "models",
        Path("/Users/a58/code/MyLLM/models")
    ]
    
    gguf_models = []
    for path in possible_paths:
        if path.exists():
            for file in path.rglob("*.gguf"):
                gguf_models.append(file)
    
    return gguf_models


def test_llamacpp_basic(model_path, prompt="什么是大语言模型？"):
    """基础推理测试"""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("❌ llama-cpp-python未安装")
        print("\n安装方法：")
        print("  pip install llama-cpp-python")
        print("\nM1用户使用Metal加速：")
        print("  CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python --force-reinstall --no-cache-dir")
        return
    
    print(f"🤖 模型: {model_path.name}")
    print(f"📝 问题: {prompt}")
    print("-" * 60)
    
    # 加载模型
    print("\n[1/3] 加载模型...")
    start = time.time()
    
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,          # 上下文窗口
        n_threads=6,         # CPU线程数（M1建议4-8）
        n_gpu_layers=1,      # 使用Metal加速
        verbose=False
    )
    
    load_time = time.time() - start
    print(f"✅ 模型加载完成 ({load_time:.2f}秒)")
    
    # 编码输入
    print("\n[2/3] 编码输入...")
    # Qwen2聊天格式
    formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # 生成回复
    print("\n[3/3] 生成回复...")
    start = time.time()
    
    output = llm(
        formatted_prompt,
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["<|im_end|>", "<|endoftext|>"],
        echo=False
    )
    
    gen_time = time.time() - start
    
    # 提取结果
    response = output['choices'][0]['text']
    tokens_generated = output['usage']['completion_tokens']
    
    print(f"✅ 生成完成 ({gen_time:.2f}秒)")
    print(f"   生成token数: {tokens_generated}")
    print(f"   速度: {tokens_generated/gen_time:.2f} tokens/秒")
    
    print("\n💬 回答:")
    print("-" * 60)
    print(response.strip())
    print("-" * 60)


def test_llamacpp_stream(model_path, prompt="写一首关于人工智能的四行诗"):
    """流式输出测试"""
    try:
        from llama_cpp import Llama
    except ImportError:
        return
    
    print(f"\n{'='*60}")
    print("🌊 流式输出测试")
    print(f"🤖 模型: {model_path.name}")
    print(f"📝 问题: {prompt}")
    print("-" * 60)
    
    # 加载模型
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=6,
        n_gpu_layers=1,
        verbose=False
    )
    
    # 格式化输入
    formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    print("\n💬 回答: ", end="", flush=True)
    
    # 流式生成
    stream = llm(
        formatted_prompt,
        max_tokens=200,
        temperature=0.7,
        stream=True,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    
    token_count = 0
    start_time = time.time()
    
    for output in stream:
        text = output['choices'][0]['text']
        print(text, end="", flush=True)
        token_count += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n\n⏱️  耗时: {elapsed:.2f}秒")
    print(f"🚀 速度: {token_count/elapsed:.2f} tokens/秒")


def benchmark_quantization(model_paths):
    """对比不同量化方式的性能"""
    try:
        from llama_cpp import Llama
    except ImportError:
        return
    
    print("\n" + "="*60)
    print("📊 量化方式性能对比")
    print("="*60)
    
    test_prompt = "什么是机器学习？用一句话回答。"
    
    results = []
    
    for model_path in model_paths:
        print(f"\n测试: {model_path.name}")
        
        # 加载模型
        start = time.time()
        llm = Llama(
            model_path=str(model_path),
            n_ctx=512,
            n_threads=6,
            verbose=False
        )
        load_time = time.time() - start
        
        # 推理测试
        formatted_prompt = f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        start = time.time()
        output = llm(formatted_prompt, max_tokens=50, temperature=0.7)
        gen_time = time.time() - start
        
        tokens = output['usage']['completion_tokens']
        speed = tokens / gen_time
        
        # 获取模型大小
        size_gb = model_path.stat().st_size / 1e9
        
        results.append({
            'name': model_path.name,
            'size': size_gb,
            'load_time': load_time,
            'speed': speed
        })
        
        print(f"  大小: {size_gb:.2f} GB")
        print(f"  加载: {load_time:.2f}秒")
        print(f"  速度: {speed:.2f} tokens/秒")
    
    # 输出对比表格
    print("\n" + "="*60)
    print("对比总结:")
    print(f"{'模型':<40} {'大小':>8} {'加载':>8} {'速度':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<40} {r['size']:>7.2f}G {r['load_time']:>7.2f}s {r['speed']:>9.2f}t/s")


if __name__ == "__main__":
    print("🚀 llama.cpp推理测试\n")
    
    # 查找GGUF模型
    print("🔍 搜索GGUF模型...")
    models = find_gguf_models()
    
    if not models:
        print("\n❌ 未找到GGUF模型")
        print("\n请先下载模型：")
        print("\n方法1：从HuggingFace下载（推荐）")
        print("  pip install huggingface-hub")
        print("  huggingface-cli download Qwen/Qwen2-7B-Instruct-GGUF \\")
        print("    qwen2-7b-instruct-q4_k_m.gguf \\")
        print("    --local-dir ~/llama.cpp/models")
        
        print("\n方法2：如果已安装Ollama，模型在：")
        print("  ~/.ollama/models/blobs/")
        print("  （需要找到.gguf文件）")
        
        print("\n方法3：手动下载")
        print("  访问：https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF")
        print("  下载：qwen2-7b-instruct-q4_k_m.gguf")
        print("  保存到：~/llama.cpp/models/")
        
        exit(1)
    
    print(f"\n✅ 找到 {len(models)} 个模型:")
    for i, model in enumerate(models, 1):
        size_gb = model.stat().st_size / 1e9
        print(f"  {i}. {model.name} ({size_gb:.2f} GB)")
        print(f"     位置: {model.parent}")
    
    # 选择第一个模型进行测试
    selected_model = models[0]
    
    print(f"\n使用模型: {selected_model.name}\n")
    
    # 测试1：基础推理
    test_llamacpp_basic(
        selected_model,
        prompt="什么是RAG技术？用简单的话解释。"
    )
    
    # 测试2：流式输出
    test_llamacpp_stream(
        selected_model,
        prompt="列举3个大语言模型的应用场景"
    )
    
    # 如果找到多个模型，对比性能
    if len(models) > 1:
        print("\n" + "="*60)
        response = input("是否对比不同量化方式的性能？[y/N]: ")
        if response.lower() == 'y':
            benchmark_quantization(models[:3])  # 最多对比3个
    
    print("\n✅ 测试完成！")
    print("\n💡 提示：")
    print("  - llama.cpp比Transformers快5-10倍")
    print("  - Q4_K_M量化是最佳平衡选择")
    print("  - M1芯片Metal加速效果明显")
    print("  - 可以通过n_threads参数调整CPU线程数")

