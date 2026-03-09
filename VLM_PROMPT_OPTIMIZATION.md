# VLM Prompt 优化指南

## 问题：为什么改进后的 Prompt 反而性能下降？

### 实验结果

| Prompt 版本 | SR | MEB | 说明 |
|------------|----|----|------|
| **原始版本** | ~70%? | ~4-5? | 简单的中文 prompt |
| **复杂版本** | 66.67% | 2.0 | 过于详细的英文 prompt |
| **优化版本** | ? | ? | 简洁平衡的 prompt |

### 原因分析

#### 1. **过于保守的规则**

```python
# 问题：排除太多选项
"Bottom layers (L0-L2) are critical supports - NEVER remove"
"Top layers (L16-L17) are unstable - avoid unless necessary"

# 结果：
- 排除了 L0-L2（9 块）
- 排除了 L16-L17（6 块）
- 只剩下 L3-L15（39 块）
- 但其中很多已经被移除
- 实际可选的很少，容易选错
```

#### 2. **Prompt 过于复杂**

```python
# 问题：5 步分析框架
"1. Identify critical support layers..."
"2. Check which layers have gaps..."
"3. Evaluate each candidate block's role..."
"4. Consider proximity to tower center..."
"5. Make your recommendation..."

# 结果：
- VLM 可能在中间步骤出错
- 过多的约束导致决策困难
- 分析瘫痪（analysis paralysis）
```

#### 3. **层状态摘要的问题**

```python
# 问题：提供了详细的层信息
layer_summary = """
  Layer 0: 3 blocks remaining ([0, 1, 2])
  Layer 1: 3 blocks remaining ([3, 4, 5])
  ...
"""

# 结果：
- Token 消耗增加
- 信息过载
- VLM 可能被细节分散注意力
```

---

## Prompt 设计原则

### ✅ 好的 Prompt

1. **简洁明了**
   - 核心规则 3-5 条
   - 避免过多细节
   - 重点突出

2. **平衡约束**
   - 给出指导，但不过度限制
   - 允许 VLM 灵活决策
   - 避免"NEVER"这样的绝对词

3. **结构清晰**
   - 分点列出
   - 逻辑顺序
   - 易于理解

4. **适配模型**
   - GPT-4o：可以处理复杂 prompt
   - Claude：喜欢结构化的 prompt
   - Gemini：更适合简洁的 prompt

### ❌ 坏的 Prompt

1. **过于复杂**
   - 太多步骤
   - 太多约束
   - 信息过载

2. **过于保守**
   - 排除太多选项
   - 绝对化的规则
   - 限制灵活性

3. **模糊不清**
   - 没有明确指导
   - 自相矛盾
   - 缺少结构

---

## 三个版本对比

### 版本 1：原始（简单）

```python
SYSTEM_PROMPT = (
    "你是一个精通结构力学和积木物理的专家。"
    "你将看到 Jenga 塔的图像，每块积木上标注了数字 ID。"
    "你的任务是选出抽取后最不可能导致塔倒塌的积木。"
)

USER_PROMPT = (
    f"当前可抽取 ID: [{valid_str}]\n"
    f"已抽走 ID: [{removed_str}]\n\n"
    f"请逐步分析并以 JSON 输出推荐"
)
```

**优点：**
- ✅ 简单直接
- ✅ 给 VLM 充分自由
- ✅ Token 消耗少

**缺点：**
- ❌ 缺少物理知识指导
- ❌ 可能选择不合理的积木

**预期性能：** SR ~70%, MEB ~4-5

---

### 版本 2：复杂（过度）

```python
SYSTEM_PROMPT = (
    "You are an expert in structural mechanics...\n"
    "Key Physical Principles:\n"
    "1. Bottom layers (L0-L2) - NEVER remove\n"
    "2. Top layers (L16-L17) - avoid unless necessary\n"
    "3. Middle layers (L5-L12) - generally safer\n"
    "...(6 条规则)\n"
    "Analysis Framework:\n"
    "- Identify which layers have missing blocks\n"
    "- Avoid blocks that are sole supporters\n"
    "...(4 步分析)"
)

USER_PROMPT = (
    f"Layer Status:\n{layer_summary}\n\n"
    f"Please analyze step-by-step:\n"
    f"1. Identify critical support layers\n"
    f"2. Check which layers have gaps\n"
    f"...(5 步分析)"
)
```

**优点：**
- ✅ 提供详细的物理知识
- ✅ 结构化的分析框架

**缺点：**
- ❌ 过于复杂
- ❌ 过于保守（排除太多）
- ❌ Token 消耗大
- ❌ 可能导致分析瘫痪

**实际性能：** SR = 66.67%, MEB = 2.0 ❌

---

### 版本 3：优化（平衡）⭐ 推荐

```python
SYSTEM_PROMPT = (
    "You are an expert in Jenga tower physics.\n"
    "Your task is to select the safest block to extract.\n\n"
    "Key Principles:\n"
    "- Avoid bottom 2-3 layers (critical support)\n"
    "- Middle layers (L5-L12) are generally safer\n"
    "- Blocks near tower center are more stable\n"
    "- Complete layers (3 blocks) are safer than layers with gaps\n\n"
    "Think step-by-step, but keep it simple."
)

USER_PROMPT = (
    f"Available blocks: [{valid_str}]\n"
    f"Already removed: [{removed_str}]\n\n"
    f"Consider:\n"
    f"1. Which layers still have all 3 blocks?\n"
    f"2. Which blocks are in middle height range?\n"
    f"3. Avoid bottom layers (L0-L2)\n\n"
    f"Respond with JSON"
)
```

**优点：**
- ✅ 简洁但有指导
- ✅ 平衡的约束（不过度）
- ✅ 清晰的结构
- ✅ Token 消耗适中

**缺点：**
- ⚠️ 需要测试验证

**预期性能：** SR ~75%, MEB ~5-6

---

## 针对不同模型的优化

### GPT-4o

```python
# GPT-4o 可以处理更复杂的 prompt
SYSTEM_PROMPT = (
    "You are an expert in structural mechanics and Jenga physics.\n"
    "Analyze the tower carefully and select the safest block.\n\n"
    "Physical Principles:\n"
    "- Bottom layers provide critical support\n"
    "- Middle layers (L5-L12) are generally safer\n"
    "- Consider load distribution and center of mass\n"
    "- Layers with gaps are less stable"
)
```

### Claude 3.5

```python
# Claude 喜欢结构化的 prompt
SYSTEM_PROMPT = (
    "You are a Jenga expert. Your goal: select the safest block.\n\n"
    "Guidelines:\n"
    "1. Safety Priority: Avoid bottom 3 layers\n"
    "2. Preferred Range: Middle layers (L5-L12)\n"
    "3. Stability Factor: Blocks near center are safer\n"
    "4. Layer Integrity: Complete layers > layers with gaps"
)
```

### Gemini Pro

```python
# Gemini 更适合简洁的 prompt
SYSTEM_PROMPT = (
    "You are a Jenga expert. Select the safest block to extract.\n\n"
    "Key rules:\n"
    "- Avoid bottom layers (L0-L2)\n"
    "- Prefer middle layers (L5-L12)\n"
    "- Choose blocks near tower center\n"
    "- Complete layers are safer"
)
```

---

## 实验建议

### 方案 A：A/B 测试

```bash
# 测试原始 prompt
python vlm_jenga_baseline.py --max_episodes 10 --seed 42 \
    --model gemini-2.5-pro --prompt_version original

# 测试优化 prompt
python vlm_jenga_baseline.py --max_episodes 10 --seed 42 \
    --model gemini-2.5-pro --prompt_version optimized

# 对比结果
```

### 方案 B：多模型对比

```bash
# GPT-4o
python vlm_jenga_baseline.py --max_episodes 10 --seed 42 \
    --model gpt-4o

# Claude 3.5
python vlm_jenga_baseline.py --max_episodes 10 --seed 42 \
    --model claude-3-5-sonnet-20240620

# Gemini Pro
python vlm_jenga_baseline.py --max_episodes 10 --seed 42 \
    --model gemini-2.5-pro
```

---

## 当前状态

我已经更新了 `vlm_jenga_baseline.py` 为**版本 3（优化版）**：

- ✅ 简洁但有指导
- ✅ 平衡的约束
- ✅ 适合 Gemini

### 重新测试

```bash
python evaluate_jenga.py --model vlm --num_episodes 10 --seed 42 \
    --api_key YOUR_KEY --vlm_model gemini-2.5-pro \
    --record_decisions --output results/vlm_optimized.json
```

**预期：** SR 应该提升到 75%+, MEB 提升到 5+

---

## 总结

### 为什么复杂 Prompt 反而更差？

1. **过度约束** - 排除太多选项
2. **分析瘫痪** - 步骤太多导致困惑
3. **信息过载** - 太多细节分散注意力
4. **模型不匹配** - Gemini 更适合简洁 prompt

### 最佳实践

1. ✅ **简洁但有指导** - 3-5 条核心规则
2. ✅ **平衡约束** - 给出建议，不过度限制
3. ✅ **清晰结构** - 分点列出，易于理解
4. ✅ **适配模型** - 根据模型特性调整

### 下一步

1. 测试优化后的 prompt
2. 对比不同模型的表现
3. 选择最佳组合用于论文
