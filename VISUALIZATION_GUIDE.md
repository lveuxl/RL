# 可视化模型决策序列 - 使用指南

## 功能

可视化不同模型选择积木的顺序和策略，包括：

1. **单 Episode 序列图** - 显示每步选择的积木层数
2. **塔视图** - 显示积木在塔中的位置和移除顺序
3. **层数分布** - 对比不同模型倾向选择哪些层
4. **位置热力图** - 显示每个位置被选择的频率
5. **策略对比** - 对比平均高度、范围、成功率

---

## 使用流程

### 步骤 1：运行评估并记录决策

```bash
# 评估 Heuristic Baseline（记录决策）
python evaluate_jenga.py --model heuristic --num_episodes 10 --seed 42 \
    --record_decisions --output results/heuristic.json

# 评估 Point Cloud Heuristic（记录决策）
python evaluate_jenga.py --model pointcloud_heuristic --num_episodes 10 --seed 42 \
    --record_decisions --output results/pointcloud.json

# 评估 RL Model（记录决策）
python evaluate_jenga.py --model rl --num_episodes 10 --seed 42 \
    --checkpoint runs/jenga_ppo \
    --record_decisions --output results/rl.json
```

**注意：** 必须添加 `--record_decisions` 参数才会记录决策序列！

### 步骤 2：生成可视化

```bash
python visualize_decisions.py \
    --decisions results/heuristic.json results/pointcloud.json results/rl.json \
    --names "Oracle Heuristic" "Point Cloud Heuristic" "RL Model" \
    --output visualizations/ \
    --episode 0
```
python visualize_decisions.py \
    --decisions results/pointcloud.json \
    --names "Heuristic" \
    --output visualizations/ \
    --episode 0

**参数说明：**
- `--decisions`: 决策记录文件（JSON）
- `--names`: 模型名称（用于图表标题）
- `--output`: 输出目录
- `--episode`: 可视化第几个 episode（默认 0）

---

## 输出文件

运行后会在 `visualizations/` 目录生成以下文件：

### 1. 单 Episode 可视化

**文件：** `{model_name}_ep{N}_sequence.png`

**内容：**
- X 轴：步数
- Y 轴：积木层数
- 绿点：成功抽取
- 红点：导致坍塌
- 蓝色虚线：连接顺序

**示例：**
```
Oracle Heuristic_ep0_sequence.png
Point Cloud Heuristic_ep0_sequence.png
RL Model_ep0_sequence.png
```

### 2. 塔视图

**文件：** `{model_name}_ep{N}_tower.png`

**内容：**
- 侧视图显示整个塔
- 每个积木用矩形表示
- 颜色表示移除顺序（viridis 色图）
- 标注积木 ID 和步数

### 3. 层数分布对比

**文件：** `level_distribution.png`

**内容：**
- 并排显示所有模型
- 柱状图显示每层被选择的频率
- 百分比标注

**观察：**
- Oracle Heuristic：倾向选择中间层
- Point Cloud Heuristic：分布更分散
- RL Model：学习到的策略

### 4. 位置热力图

**文件：** `position_heatmap.png`

**内容：**
- 18×3 的热力图（层 × 位置）
- 颜色深度表示选择频率
- 数值标注百分比

**观察：**
- 哪些位置最常被选择
- 不同模型的位置偏好

### 5. 策略对比

**文件：** `strategy_comparison.png`

**内容：**
- 3 个子图：
  1. 平均选择高度
  2. 层数范围（箱线图）
  3. 成功率

**观察：**
- 模型倾向选择高层还是低层
- 选择的多样性
- 整体成功率

---

## 示例输出解读

### 场景 1：Oracle Heuristic

```
层数分布：
- 主要选择 L5-L12（中间层）
- 避开 L0-L2（底层）和 L16-L17（顶层）
- 策略：排除最高两层和最底层

位置热力图：
- 中心位置（Center）选择频率最高
- 符合"距离塔中心最近"的规则

平均高度：8.5 层
成功率：95%
```

### 场景 2：Point Cloud Heuristic

```
层数分布：
- 分布更分散
- 有时会选择底层（估计误差）
- 策略：基于点云质心估计

位置热力图：
- 分布较均匀
- 没有明显偏好

平均高度：7.2 层
成功率：72%
```

### 场景 3：RL Model

```
层数分布：
- 学习到类似 Oracle 的策略
- 主要选择中间层
- 但有更多探索

位置热力图：
- 有一定偏好
- 学习到了位置的重要性

平均高度：8.1 层
成功率：85%
```

---

## 高级用法

### 1. 对比不同场景

```bash
# Easy 场景
python evaluate_jenga.py --model rl --num_episodes 10 --seed 42 \
    --checkpoint runs/jenga_ppo --record_decisions \
    --output results/rl_easy.json

# Hard 场景
python evaluate_jenga.py --model rl --num_episodes 10 --seed 42 \
    --checkpoint runs/jenga_ppo --record_decisions \
    --ood_mode hard --output results/rl_hard.json

# 可视化对比
python visualize_decisions.py \
    --decisions results/rl_easy.json results/rl_hard.json \
    --names "RL (Easy)" "RL (Hard)" \
    --output visualizations/scenario_comparison/
```

### 2. 可视化多个 Episodes

```bash
# 可视化 Episode 0
python visualize_decisions.py \
    --decisions results/rl.json \
    --names "RL Model" \
    --output visualizations/ep0/ \
    --episode 0

# 可视化 Episode 1
python visualize_decisions.py \
    --decisions results/rl.json \
    --names "RL Model" \
    --output visualizations/ep1/ \
    --episode 1
```

### 3. 批量生成

```bash
#!/bin/bash
# generate_all_visualizations.sh

MODELS=("heuristic" "pointcloud_heuristic" "rl")
NAMES=("Oracle Heuristic" "Point Cloud Heuristic" "RL Model")

# 1. 运行评估
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    
    if [ "$model" = "rl" ]; then
        python evaluate_jenga.py --model $model --num_episodes 10 --seed 42 \
            --checkpoint runs/jenga_ppo --record_decisions \
            --output results/${model}.json
    else
        python evaluate_jenga.py --model $model --num_episodes 10 --seed 42 \
            --record_decisions --output results/${model}.json
    fi
done

# 2. 生成可视化
python visualize_decisions.py \
    --decisions results/heuristic.json results/pointcloud_heuristic.json results/rl.json \
    --names "Oracle Heuristic" "Point Cloud Heuristic" "RL Model" \
    --output visualizations/ \
    --episode 0

echo "所有可视化已生成！"
```

---

## 论文中使用

### 图表说明示例

```
图 3: 不同模型的积木选择策略对比

(a) 层数分布：Oracle Heuristic 主要选择中间层（L5-L12），
    避开底层和顶层。Point Cloud Heuristic 由于估计误差，
    分布更分散。我们的 RL 模型学习到了类似的策略。

(b) 位置热力图：Oracle Heuristic 倾向选择中心位置，
    符合"距离塔中心最近"的规则。我们的 RL 模型也学习到了
    这一偏好，但更加灵活。

(c) 策略对比：我们的 RL 模型的平均选择高度（8.1 层）
    接近 Oracle（8.5 层），显著高于 Point Cloud Heuristic（7.2 层），
    说明学习到了更优的策略。
```

### 单 Episode 示例

```
图 4: 典型 Episode 的决策序列

显示了三种方法在同一初始配置下的决策序列。
Oracle Heuristic（绿色）稳定地从中间层抽取，
成功完成 12 步。Point Cloud Heuristic（蓝色）
由于估计误差，在第 8 步选择了不稳定的积木导致坍塌。
我们的 RL 模型（红色）成功完成 11 步，展现了
接近 Oracle 的性能。
```

---

## 故障排除

### 问题 1：没有生成决策记录

**原因：** 忘记添加 `--record_decisions` 参数

**解决：**
```bash
# ❌ 错误
python evaluate_jenga.py --model rl --num_episodes 10 --output results/rl.json

# ✅ 正确
python evaluate_jenga.py --model rl --num_episodes 10 --output results/rl.json \
    --record_decisions
```

### 问题 2：JSON 文件中没有 decisions 字段

**检查：**
```python
import json
with open('results/rl.json', 'r') as f:
    data = json.load(f)
    print('decisions' in data)  # 应该是 True
```

### 问题 3：可视化脚本报错

**常见原因：**
- matplotlib 未安装：`pip install matplotlib seaborn`
- 文件路径错误：检查 `--decisions` 参数
- 模型数量和名称数量不匹配

---

## 总结

### 工作流程

1. ✅ 运行评估（添加 `--record_decisions`）
2. ✅ 生成可视化（`visualize_decisions.py`）
3. ✅ 分析策略差异
4. ✅ 用于论文图表

### 关键文件

- `evaluate_jenga.py` - 评估并记录决策
- `visualize_decisions.py` - 生成可视化
- `results/*.json` - 决策记录
- `visualizations/*.png` - 可视化图表

### 论文价值

- 直观展示不同模型的策略
- 证明 RL 模型学习到了合理的策略
- 解释为什么 RL 模型性能更好
- 增强论文的可读性和说服力
