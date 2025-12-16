# LLM学习 - 从零开始

## 第0步：准备工作（首次配置）

### 0.1 安装Homebrew（如果还没有）
打开终端（Applications → 实用工具 → 终端），粘贴运行：
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
等待安装完成（10-20分钟）。

### 0.2 安装Miniconda
```bash
# 下载Miniconda
cd ~/Downloads
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# 安装
bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3

# 初始化
~/miniconda3/bin/conda init zsh

# 关闭并重新打开终端，让配置生效
```

### 0.3 创建Python环境
```bash
# 创建llm-learning环境
conda create -n llm-learning python=3.10 -y

# 激活环境（每次打开新终端都要运行）
conda activate llm-learning

# 验证Python版本
python --version
# 应该显示：Python 3.10.x
```

### 0.4 克隆项目
```bash
# 进入工作目录
cd ~/code  # 如果没有code文件夹，先运行：mkdir -p ~/code

# 克隆或创建项目（你应该已经在 /Users/a58/code/MyLLM 了）
cd /Users/a58/code/MyLLM

# 安装基础依赖
pip install requests numpy pandas tqdm
```

---

## 📍 第1步：安装llama.cpp（当前任务）

### 1.1 克隆llama.cpp仓库

在终端运行：
```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

### 1.2 编译（M1优化，使用CMake）

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release
```

编译需要3-5分钟，完成后会在 `build/bin/` 目录生成可执行文件。

---

## ✅ 第1步：llama.cpp编译完成！

---

## 📍 第2步：下载GGUF模型（当前任务）

推荐使用 Qwen2.5-3B（3B模型，稳定且效果好）：

```bash
cd ~/llama.cpp
mkdir -p models
cd models

# 下载Qwen2.5-3B的Q4量化版本（约2GB）
# 方案1：使用curl带续传功能（推荐）
curl -C - -L -o qwen2.5-3b-instruct-q4_k_m.gguf \
  --connect-timeout 60 --max-time 3600 \
  "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

# 如果中断了，再次运行上面的命令会自动续传

# 方案2：使用wget（支持断点续传）
# brew install wget  # 先安装wget
# wget -c "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

# 方案3：使用hf镜像站（速度更快）
# curl -L -o qwen2.5-3b-instruct-q4_k_m.gguf \
#   "https://hf-mirror.com/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
```

下载需要5-15分钟（取决于网速）。

---

## ✅ 第2步：模型下载完成！

---

## 📍 第3步：测试模型（当前任务）

### 3.1 验证模型文件

首先确认模型文件已下载完整：

```bash
cd ~/llama.cpp/models
ls -lh qwen2.5-3b-instruct-q4_k_m.gguf
```

应该看到文件大小约 1.9-2.1GB。

### 3.2 运行模型（命令行测试）

**注意：如果遇到 Metal 错误，需要调整参数**

```bash
cd ~/llama.cpp

# 方案1：减小上下文窗口（推荐）
./build/bin/llama-cli \
  -m models/qwen2.5-3b-instruct-q4_k_m.gguf \
  -p "你好，请用一句话介绍你自己。" \
  -n 128 \
  -c 2048 \
  --temp 0.7

# 方案2：禁用 Metal（使用 CPU，较慢但稳定）
./build/bin/llama-cli \
  -m models/qwen2.5-3b-instruct-q4_k_m.gguf \
  -p "你好，请用一句话介绍你自己。" \
  -n 128 \
  -ngl 0 \
  --temp 0.7
```

**参数说明：**
- `-m`：指定模型文件
- `-p`：提示词（prompt）
- `-n 128`：生成最多128个token
- `-c 2048`：上下文长度设为2048（默认4096可能太大）
- `-ngl 0`：不使用GPU层，纯CPU运行
- `--temp 0.7`：温度参数，控制创造性（0-1）

你应该能看到模型生成中文回复！

**常见问题：**
- 如果还是崩溃，试试方案2（纯CPU模式）
- 或者进一步减小上下文：`-c 1024` 或 `-c 512`

### 3.3 启动API服务器（推荐）

启动一个本地API服务器，可以通过HTTP调用模型：

```bash
cd ~/llama.cpp

# 使用较小的上下文避免内存问题
./build/bin/llama-server \
  -m models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -c 2048 \
  -ngl 32
```

**参数说明：**
- `--host 127.0.0.1`：本地访问
- `--port 8080`：端口号
- `-c 2048`：上下文长度（减小以避免 Metal 错误）
- `-ngl 32`：GPU加载层数（36层中的32层，留点给CPU）

启动后：
- 服务器会持续运行（保持终端打开）
- 在浏览器打开：http://localhost:8080
- 你会看到一个聊天界面！

**如果还是遇到错误，用纯CPU模式：**
```bash
./build/bin/llama-server \
  -m models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -c 2048 \
  -ngl 0
```

**测试API（新开终端）：**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

---

## ✅ 第3步：模型测试成功！

---

## 📍 第4步：用Python调用模型（当前任务）

现在我们用Python代码来调用本地模型，这样更灵活，可以集成到你的项目中。

### 4.1 安装Python库

```bash
cd ~/code/MyLLM
conda activate llm-learning

# 检查Python版本（确保是3.10）
python --version
which python

# 如果显示Python 2.7，重新初始化conda
source ~/miniconda3/bin/activate
conda activate llm-learning

# 再次检查
python --version  # 应该显示 Python 3.10.x

# 安装 llama-cpp-python（Python绑定）
pip install llama-cpp-python
```

**如果安装过程很慢或失败，可以尝试：**
```bash
# 使用预编译的wheel（更快）
CMAKE_ARGS="-DGGML_METAL=off" pip install llama-cpp-python --no-cache-dir
```

### 4.2 创建第一个Python脚本

创建文件 `test_model.py`：

```python
from llama_cpp import Llama

# 加载模型（纯CPU模式）
llm = Llama(
    model_path="/Users/a58/llama.cpp/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=2048,       # 上下文长度
    n_threads=4,      # CPU线程数
    n_gpu_layers=0,   # 0 = 纯CPU模式
    verbose=False     # 不显示加载信息
)

# 测试对话
prompt = "你好，请用一句话介绍你自己。"

print(f"问题: {prompt}")
print("回答: ", end="", flush=True)

# 生成回复（流式输出）
output = llm(
    prompt,
    max_tokens=128,
    temperature=0.7,
    stop=["</s>", "\n\n"],  # 停止标记
    echo=False
)

print(output['choices'][0]['text'])
print(f"\n生成了 {output['usage']['completion_tokens']} 个token")
```

### 4.3 运行测试

```bash
cd ~/code/MyLLM
python test_model.py
```

你会看到模型的中文回复！

### 4.4 创建聊天机器人

创建 `chatbot.py`，实现连续对话：

```python
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
    
    # 生成回复
    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.7,
        stop=["用户:", "\n\n"],
        echo=False
    )
    
    response = output['choices'][0]['text'].strip()
    conversation_history.append(f"助手: {response}")
    
    print(f"AI: {response}\n")
    
    # 限制历史长度（避免超出上下文）
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
```

### 4.5 运行聊天机器人

```bash
python chatbot.py
```

现在你可以和模型连续对话了！

---

## ✅ 第4步：Python调用成功！

---

## 📍 第5步：提示词工程和参数调优（当前任务）

现在你已经能运行模型了，接下来学习如何**让模型生成更好的内容**。

### 5.1 理解关键参数

创建 `test_parameters.py` 来实验不同参数：

```python
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
```

**参数说明：**
- **temperature** (0-2): 创造性
  - 0.1-0.3: 保守、确定性强（适合事实性问题）
  - 0.7-0.9: 平衡（日常对话）
  - 1.0-2.0: 创造性强（创意写作）

- **top_p** (0-1): 采样范围
  - 0.9: 从概率最高的90%的词中选择
  - 0.95: 更多样性
  
- **repeat_penalty** (1.0-1.5): 防止重复
  - 1.0: 不惩罚
  - 1.1-1.2: 轻度惩罚（推荐）
  - 1.5+: 强力惩罚

### 5.2 提示词工程技巧

创建 `prompt_engineering.py`：

```python
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
```

### 5.3 实用应用示例

创建 `practical_examples.py`：

```python
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
```

### 5.4 运行实验

```bash
cd ~/code/MyLLM

# 实验不同参数
python test_parameters.py

# 学习提示词技巧
python prompt_engineering.py

# 查看实用案例
python practical_examples.py
```

---

## 💡 核心要点总结

### 参数调优：
- **事实问答**：temperature=0.1-0.3
- **日常对话**：temperature=0.7
- **创意写作**：temperature=0.9-1.2

### 提示词技巧：
1. ✅ 明确角色定位
2. ✅ 提供具体示例
3. ✅ 分步骤引导思考
4. ✅ 指定输出格式

---

## ⏸️ 完成第5步后告诉我

尝试这些技巧后，我给你第6步：构建实用的AI应用（翻译助手、代码助手等）。

---