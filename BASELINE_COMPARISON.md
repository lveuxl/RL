# 三种 Heuristic Baseline 的区别

## 快速对比

| 方法 | 输入 | 位姿估计方法 | 准确度 | 公平性 | 适用场景 |
|------|------|-------------|--------|--------|---------|
| **Oracle Heuristic** | 物理引擎 | 直接读取 | 100% | ❌ 不公平 | 理论上界 |
| **Point Cloud Heuristic** | 点云 | 质心 | 70-80% | ✅ 公平 | 弱 Baseline |
| **3D Det + Rules** | 点云 | 3D 检测 | 85-95% | ✅ 公平 | 强 Baseline |

## 详细说明

### 1. Oracle Heuristic（理论上界）

```python
# 直接从物理引擎获取精确位姿
pose = actor.get_pose()
center = pose.p  # [x, y, z] 毫米级精度
```

**特点：**
- ✅ 完美的位姿信息
- ✅ 无噪声、无遮挡
- ❌ 实际中无法获得
- ❌ 不能作为公平对比

**用途：** 代表理论性能上界

---

### 2. Point Cloud Heuristic（弱 Baseline）

```python
# 从点云计算质心
def estimate_center(pointcloud):
    return pointcloud.mean(axis=0)  # 简单平均
```

**特点：**
- ✅ 使用点云输入（公平）
- ✅ 实现简单
- ❌ 不准确（质心 ≠ 真实中心）
- ❌ 对遮挡敏感

**问题示例：**
```
真实积木：长方体 [0, 0, 0.5] 到 [0.15, 0.05, 0.53]
真实中心：[0.075, 0.025, 0.515]

可见点云：只有前半部分（后半部分被遮挡）
质心估计：[0.04, 0.025, 0.515]  ← 偏移 3.5cm！
```

**用途：** 最简单的公平 Baseline，可能太弱

---

### 3. 3D Det + Rules（强 Baseline）

```python
# 使用 3D 目标检测器
def estimate_center(pointcloud):
    # 1. 特征提取（PointNet）
    features = pointnet(pointcloud)
    
    # 2. 中心回归
    center = mlp(features)  # 学习从部分点云推断完整中心
    
    return center
```

**特点：**
- ✅ 使用点云输入（公平）
- ✅ 更准确的位姿估计
- ✅ 对遮挡更鲁棒
- ❌ 需要训练检测器
- ❌ 计算量更大

**优势：**
- 可以从部分点云推断完整形状
- 学习了积木的先验知识
- 更接近 Oracle 性能

**用途：** 强 Baseline，代表传统方法的最佳实现

---

## 为什么需要三种 Baseline？

### 实验设计逻辑

```
Oracle Heuristic (100%)
    ↓ 差距 = 视觉感知的代价
3D Det + Rules (85-95%)  ← 传统方法的最佳实现
    ↓ 差距 = 学习的优势
Your RL Model (88-92%)   ← 你的方法
    ↓ 差距 = 简单方法的局限
Point Cloud Heuristic (70-80%)  ← 最简单的实现
```

### 论文中的叙述

```
我们对比了四种方法：

1. Oracle Heuristic (完美感知): SR=95%, MEB=12.1
   - 代表理论性能上界
   - 不可实际部署

2. 3D Det + Rules (点云 + 检测): SR=87%, MEB=10.3
   - 传统方法的最佳实现
   - 需要训练专门的检测器

3. Our Method (点云 + 端到端学习): SR=90%, MEB=11.2
   - 超过传统方法 3%
   - 端到端学习，无需单独训练检测器
   - 达到 Oracle 性能的 95%

4. Point Cloud Heuristic (点云 + 质心): SR=72%, MEB=8.1
   - 最简单的实现
   - 验证了准确位姿估计的重要性
```

---

## 实现对比

### Point Cloud Heuristic（已实现）

```python
# pointcloud_heuristic_baseline.py
class PointCloudHeuristicAgent:
    def _estimate_center(self, pointcloud):
        # 简单质心
        return pointcloud[:, :3].mean(axis=0)
```

**运行：**
```bash
python pointcloud_heuristic_baseline.py --max_episodes 100 --seed 42
```

### 3D Det + Rules（刚实现）

```python
# 3d_det_heuristic_baseline.py
class ThreeDDetHeuristicAgent:
    def __init__(self, detector):
        self.detector = detector  # 训练好的 3D 检测器
    
    def _estimate_center(self, pointcloud):
        # 使用检测器
        center, conf = self.detector.detect(pointcloud)
        return center
```

**运行：**
```bash
# 方式 1: 训练检测器（推荐）
python 3d_det_heuristic_baseline.py \
    --max_episodes 100 --seed 42 \
    --train_detector --detector_samples 1000

# 方式 2: 不训练（回退到质心）
python 3d_det_heuristic_baseline.py \
    --max_episodes 100 --seed 42
```

---

## 我的建议

### 方案 A：简化版（快速验证）

只对比三种：

```bash
# 1. Oracle Heuristic (上界)
python evaluate_jenga.py --model heuristic --num_episodes 100

# 2. Point Cloud Heuristic (公平对比)
python evaluate_jenga.py --model pointcloud_heuristic --num_episodes 100

# 3. Your RL Model
python evaluate_jenga.py --model rl --num_episodes 100 --checkpoint runs/jenga_ppo
```

**优点：** 简单、快速
**缺点：** Point Cloud Heuristic 可能太弱，对比不够有说服力

### 方案 B：完整版（推荐）

对比四种：

```bash
# 1. Oracle Heuristic (上界)
python heuristic_jenga_baseline.py --max_episodes 100

# 2. 3D Det + Rules (强 Baseline)
python 3d_det_heuristic_baseline.py --max_episodes 100 --train_detector

# 3. Point Cloud Heuristic (弱 Baseline)
python pointcloud_heuristic_baseline.py --max_episodes 100

# 4. Your RL Model
python evaluate_jenga.py --model rl --num_episodes 100 --checkpoint runs/jenga_ppo
```

**优点：** 完整、有说服力
**缺点：** 需要训练 3D 检测器

---

## 预期结果

### 如果 Point Cloud Heuristic 太弱

```
Oracle Heuristic:        SR=95%, MEB=12.1
3D Det + Rules:          SR=87%, MEB=10.3  ← 需要这个
Your RL Model:           SR=90%, MEB=11.2  ← 超过传统方法
Point Cloud Heuristic:   SR=72%, MEB=8.1   ← 太弱，对比不明显
```

**结论：** 需要 3D Det + Rules 作为强 Baseline

### 如果 Point Cloud Heuristic 足够强

```
Oracle Heuristic:        SR=95%, MEB=12.1
Point Cloud Heuristic:   SR=85%, MEB=10.0  ← 足够强
Your RL Model:           SR=90%, MEB=11.2  ← 明显超过
```

**结论：** 不需要 3D Det + Rules

---

## 快速决策流程

### 步骤 1：先运行 Point Cloud Heuristic

```bash
python pointcloud_heuristic_baseline.py --max_episodes 10 --seed 42
```

### 步骤 2：查看结果

**如果 SR > 80%：**
- Point Cloud Heuristic 足够强
- 不需要 3D Det + Rules
- 直接用于对比

**如果 SR < 75%：**
- Point Cloud Heuristic 太弱
- 需要 3D Det + Rules
- 训练检测器并重新评估

---

## 总结

### 三种方法的定位

1. **Oracle Heuristic** = 理论上界（不公平）
2. **Point Cloud Heuristic** = 最简单的公平对比
3. **3D Det + Rules** = 最强的公平对比

### 你需要哪些？

**最少：**
- Oracle Heuristic（上界）
- Point Cloud Heuristic（公平对比）
- Your RL Model

**推荐：**
- Oracle Heuristic（上界）
- 3D Det + Rules（强 Baseline）
- Your RL Model
- (可选) Point Cloud Heuristic（弱 Baseline）

### 下一步

1. ✅ 先运行 Point Cloud Heuristic，看看性能如何
2. ⏳ 如果太弱（SR < 75%），再实现 3D Det + Rules
3. ⏳ 完成训练你的 RL 模型到 200K steps
4. ⏳ 运行完整评估并对比
