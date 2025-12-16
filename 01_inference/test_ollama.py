"""
Ollama推理测试脚本

使用前确保：
1. 已安装Ollama
2. 已下载模型：ollama pull qwen2:7b
3. Ollama服务已启动：ollama serve（通常自动启动）
"""

import requests
import json
import time


def test_ollama_api(model="qwen2:7b", prompt="用一句话介绍什么是大语言模型"):
    """测试Ollama API"""
    url = "http://localhost:11434/api/generate"
    
    print(f"🤖 模型: {model}")
    print(f"📝 问题: {prompt}")
    print("-" * 50)
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 200
        }
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=data, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 回答: {result['response']}")
            print(f"\n⏱️  耗时: {end_time - start_time:.2f}秒")
            print(f"📊 生成token数: {result.get('eval_count', 'N/A')}")
            if 'eval_count' in result and 'eval_duration' in result:
                tokens_per_sec = result['eval_count'] / (result['eval_duration'] / 1e9)
                print(f"🚀 速度: {tokens_per_sec:.2f} tokens/秒")
        else:
            print(f"❌ 错误: HTTP {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到Ollama服务")
        print("\n请检查：")
        print("1. Ollama是否已安装？运行: ollama --version")
        print("2. Ollama服务是否启动？运行: ollama serve")
        print("3. 模型是否已下载？运行: ollama list")
    except Exception as e:
        print(f"❌ 错误: {e}")


def test_ollama_stream(model="qwen2:7b", prompt="写一首关于人工智能的四行诗"):
    """测试流式输出"""
    url = "http://localhost:11434/api/generate"
    
    print(f"\n{'='*50}")
    print("🌊 流式输出测试")
    print(f"🤖 模型: {model}")
    print(f"📝 问题: {prompt}")
    print("-" * 50)
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    
    try:
        response = requests.post(url, json=data, stream=True, timeout=60)
        
        print("💬 回答: ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if 'response' in chunk:
                    print(chunk['response'], end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"❌ 错误: {e}")


def check_available_models():
    """检查可用模型"""
    url = "http://localhost:11434/api/tags"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("\n📦 已下载的模型：")
            for model in models:
                print(f"  - {model['name']}")
                print(f"    大小: {model['size'] / 1e9:.2f} GB")
                print(f"    修改时间: {model['modified_at']}")
            return [m['name'] for m in models]
        else:
            print("❌ 无法获取模型列表")
            return []
    except Exception as e:
        print(f"❌ 错误: {e}")
        return []


if __name__ == "__main__":
    print("🚀 Ollama推理测试\n")
    
    # 检查可用模型
    available_models = check_available_models()
    
    if not available_models:
        print("\n⚠️  未检测到已下载的模型")
        print("\n请先下载模型：")
        print("  ollama pull qwen2:7b        # 推荐：综合能力强")
        print("  ollama pull deepseek-coder:7b  # 代码能力强")
        exit(1)
    
    # 选择第一个可用模型进行测试
    test_model = available_models[0]
    
    # 测试1：普通问答
    test_ollama_api(
        model=test_model,
        prompt="什么是RAG技术？用简单的语言解释。"
    )
    
    # 测试2：流式输出
    test_ollama_stream(
        model=test_model,
        prompt="列举3个使用大语言模型的实际应用场景"
    )
    
    print("\n✅ 测试完成！")
    print("\n💡 提示：")
    print("  - 修改prompt参数可以测试不同的问题")
    print("  - 修改temperature参数可以调整输出的随机性（0.0-1.0）")
    print("  - 使用stream=True可以实现打字机效果")

