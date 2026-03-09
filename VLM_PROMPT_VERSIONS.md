# VLM Prompt 版本对比

## 版本 4：极简版（推荐测试）

```python
SYSTEM_PROMPT = (
    "You are a Jenga expert. "
    "Select the safest block to extract from the tower."
)

USER_PROMPT = (
    f"Available blocks: [{valid_str}]\n"
    f"Already removed: [{removed_str}]\n\n"
    f"Which block should I extract? Respond with JSON:\n"
    f'```json\n{{"block_id": <ID>}}\n```'
)
```

**特点：**
- 最简单的 prompt
- 没有任何约束
- 让 VLM 自由发挥
- 依赖 VLM 的内在知识

**预期：** 可能表现更好，因为没有误导性的约束

---

## 版本 5：Few-Shot 示例

```python
SYSTEM_PROMPT = (
    "You are a Jenga expert. Here are some examples:\n\n"
    "Example 1: Tower with all blocks intact\n"
    "Good choice: Block from middle layer (L8-L10)\n"
    "Bad choice: Bottom layer blocks (L0-L2)\n\n"
    "Example 2: Tower with some gaps\n"
    "Good choice: Block from complete layer\n"
    "Bad choice: Block from layer with gaps\n\n"
    "Now, select the safest block for the current tower."
)
```

**特点：**
- 通过示例教学
- 不直接给规则
- 让 VLM 学习模式

---

## 版本 6：反向提示（避免什么）

```python
SYSTEM_PROMPT = (
    "You are a Jenga expert. "
    "Avoid these risky choices:\n"
    "- Blocks that are sole supporters\n"
    "- Blocks from layers with existing gaps\n"
    "- Blocks at extreme positions\n\n"
    "Otherwise, use your judgment."
)
```

**特点：**
- 告诉 VLM 避免什么
- 而不是必须做什么
- 给予更多自由

---

## 建议的测试顺序

### 1. 极简版（最优先）

```bash
# 修改 vlm_jenga_baseline.py 为极简版
python evaluate_jenga.py --model vlm --num_episodes 10 --seed 42 \
    --api_key YOUR_KEY --vlm_model gemini-2.5-pro \
    --output results/vlm_minimal.json
```

### 2. 如果极简版好，就用它

### 3. 如果极简版不好，尝试 Few-Shot

---

## 关键洞察

**Less is More!**

VLM（特别是 Gemini）可能：
- 有内在的 Jenga 知识
- 不需要我们教它物理
- 过多的约束反而干扰它的判断

**类比：**
- 你不会告诉一个专家"记住要呼吸"
- 同样，不要告诉 VLM 太多它已经知道的东西
