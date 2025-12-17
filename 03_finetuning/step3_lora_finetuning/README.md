# Step 3: LoRA 微调实战

## 🎯 本节目标

使用LoRA技术实际微调模型，理解微调过程和参数调整。

## 💡 什么是LoRA？

**LoRA** (Low-Rank Adaptation) = 低秩适应

### 传统微调 vs LoRA

```
传统微调:
  更新所有参数 (3B个参数)
  需要: 24GB+ 显存
  训练时间: 数小时
  
LoRA微调:
  只添加小的适配层 (几百万个参数)
  需要: 4-8GB 显存
  训练时间: 几十分钟
  
效果: LoRA达到传统微调的90-95%
```

### LoRA原理

```
原始模型权重 W (冻结，不更新)
    +
LoRA矩阵 A × B (训练这个)
    =
微调后的模型
```

关键参数：
- **rank (r)**: LoRA矩阵的秩，越大效果越好但内存越多（推荐：8-64）
- **alpha**: 学习率缩放因子（推荐：16-32）

## 📋 学习路径

```bash
# 1. 理解LoRA原理
python 01_what_is_lora.py

# 2. 简单微调（核心！）
python 02_simple_finetune.py

# 3. 查看微调效果
python 03_test_finetuned.py
```

## ⚙️ 微调参数说明

| 参数 | 作用 | 推荐值 | 说明 |
|------|------|--------|------|
| **learning_rate** | 学习步长 | 1e-4 ~ 5e-4 | 太大不收敛，太小太慢 |
| **num_epochs** | 训练轮数 | 3-10 | 小数据集可以多训几轮 |
| **batch_size** | 批次大小 | 1-4 | 内存小就用1 |
| **lora_rank** | LoRA秩 | 8-16 | 越大效果越好但越慢 |
| **lora_alpha** | 缩放因子 | 16-32 | 通常是rank的2倍 |

## 🚀 快速开始

最简单的微调命令：

```python
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("模型路径")

# 2. 配置LoRA
lora_config = LoraConfig(r=8, lora_alpha=16)
model = get_peft_model(model, lora_config)

# 3. 训练
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

## ⚠️ 注意事项

### 内存不足？
- 降低 `batch_size` 到 1
- 降低 `lora_rank` 到 4
- 使用 `gradient_checkpointing`

### 训练太慢？
- 减少数据量
- 减少 `num_epochs`
- 如果有GPU，启用它

### 效果不好？
- 检查数据质量
- 增加训练轮数
- 调整学习率

## 📊 预期结果

微调前：
```
用户: 醉驾会受到什么处罚？
模型: 醉驾是违法的，应该受到法律制裁... (笼统)
```

微调后：
```
用户: 醉驾会受到什么处罚？
模型: 醉酒驾驶机动车的，由公安机关交通管理部门约束至酒醒，
      吊销机动车驾驶证，依法追究刑事责任；
      五年内不得重新取得机动车驾驶证。(专业！)
```

---

准备好了就开始吧！这是最激动人心的部分！🚀

