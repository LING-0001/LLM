"""
Transformers原生推理测试

这个脚本展示如何使用Transformers库直接加载和运行模型
优势：完全透明，可以看到每一步的细节
劣势：速度较慢，内存占用高

适合：理解模型加载、推理的底层原理
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time


def test_transformers_inference(
    model_name="Qwen/Qwen2-7B-Instruct",
    prompt="什么是大语言模型？",
    max_length=200,
    use_4bit=True
):
    """
    使用Transformers进行推理
    
    Args:
        model_name: HuggingFace模型名称
        prompt: 输入提示
        max_length: 最大生成长度
        use_4bit: 是否使用4-bit量化（节省内存）
    """
    
    print(f"🤖 模型: {model_name}")
    print(f"📝 问题: {prompt}")
    print(f"💾 量化: {'4-bit' if use_4bit else 'FP16'}")
    print("-" * 60)
    
    # 第一步：加载tokenizer
    print("\n[1/4] 加载Tokenizer...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print(f"✅ Tokenizer加载完成 ({time.time()-start:.2f}秒)")
    
    # 第二步：加载模型
    print("\n[2/4] 加载模型...")
    print("⚠️  首次运行会从HuggingFace下载模型，可能需要10-30分钟")
    print("   模型会缓存到 ~/.cache/huggingface/")
    
    start = time.time()
    
    if use_4bit:
        # 使用4-bit量化（需要bitsandbytes库）
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            print(f"✅ 模型加载完成 (4-bit量化) ({time.time()-start:.2f}秒)")
            
        except ImportError:
            print("⚠️  bitsandbytes未安装，使用FP16模式")
            use_4bit = False
    
    if not use_4bit:
        # 使用FP16（M1上的默认精度）
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✅ 模型加载完成 (FP16) ({time.time()-start:.2f}秒)")
    
    # 查看内存占用
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e9
        print(f"📊 显存占用: {memory_used:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("📊 使用Apple Metal GPU加速")
    
    # 第三步：编码输入
    print("\n[3/4] 编码输入...")
    start = time.time()
    
    # 构建对话格式（Qwen2格式）
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print(f"✅ 输入编码完成 ({time.time()-start:.2f}秒)")
    print(f"   输入token数: {inputs.input_ids.shape[1]}")
    
    # 第四步：生成回复
    print("\n[4/4] 生成回复...")
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    end = time.time()
    
    # 解码输出
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    print(f"✅ 生成完成 ({end-start:.2f}秒)")
    print(f"   输出token数: {outputs.shape[1] - inputs.input_ids.shape[1]}")
    print(f"   速度: {(outputs.shape[1] - inputs.input_ids.shape[1]) / (end-start):.2f} tokens/秒")
    
    print("\n💬 回答:")
    print("-" * 60)
    print(generated_text)
    print("-" * 60)


def download_model_guide():
    """下载模型指南"""
    print("\n" + "="*60)
    print("📥 模型下载指南")
    print("="*60)
    
    print("\n推荐模型（7B级别，适合M1 16G）：")
    print("\n1. Qwen2-7B-Instruct (推荐)")
    print("   - HuggingFace ID: Qwen/Qwen2-7B-Instruct")
    print("   - 中文能力最强")
    print("   - 模型大小: ~15GB (原始) / ~4GB (4-bit量化)")
    
    print("\n2. DeepSeek-Coder-7B-Instruct")
    print("   - HuggingFace ID: deepseek-ai/deepseek-coder-7b-instruct-v1.5")
    print("   - 代码能力强")
    print("   - 模型大小: ~14GB (原始) / ~4GB (4-bit量化)")
    
    print("\n3. Llama-3-8B-Instruct")
    print("   - HuggingFace ID: meta-llama/Meta-Llama-3-8B-Instruct")
    print("   - 国际主流")
    print("   - 需要在HuggingFace同意许可协议")
    
    print("\n下载方法：")
    print("\n方法1：自动下载（运行脚本时自动下载）")
    print("  python 01_inference/test_transformers.py")
    
    print("\n方法2：手动预下载")
    print("  pip install huggingface-hub")
    print("  huggingface-cli download Qwen/Qwen2-7B-Instruct")
    
    print("\n方法3：使用国内镜像加速（可选）")
    print("  export HF_ENDPOINT=https://hf-mirror.com")
    print("  python 01_inference/test_transformers.py")
    
    print("\n⚠️  注意事项：")
    print("  1. 首次下载需要10-30分钟（取决于网速）")
    print("  2. 模型会缓存到 ~/.cache/huggingface/")
    print("  3. 确保至少有20GB可用磁盘空间")
    print("  4. 建议在稳定网络环境下下载")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "guide":
        download_model_guide()
        exit(0)
    
    print("🚀 Transformers推理测试\n")
    print("⚠️  这是一个教学脚本，用于理解模型推理原理")
    print("   生产环境建议使用Ollama或llama.cpp（速度更快）\n")
    
    # 检查是否要下载模型
    response = input("是否开始测试？这将自动下载模型（首次运行）[y/N]: ")
    
    if response.lower() != 'y':
        print("\n取消测试。")
        print("如需查看下载指南，运行：python 01_inference/test_transformers.py guide")
        exit(0)
    
    try:
        test_transformers_inference(
            model_name="Qwen/Qwen2-7B-Instruct",
            prompt="用三句话解释什么是RAG技术",
            max_length=200,
            use_4bit=True
        )
        
        print("\n✅ 测试完成！")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n可能的原因：")
        print("  1. 网络连接问题（无法下载模型）")
        print("  2. 内存不足")
        print("  3. 依赖库未安装")
        print("\n建议：先使用Ollama进行测试（更简单）")

