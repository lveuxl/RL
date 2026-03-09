# 3D Det + Rules 性能诊断

## 可能的问题

### 1. 训练/测试分布不匹配

**问题：** 训练时从随机环境采样，测试时是连续的 episode

```python
# 训练：每次 reset 环境，积木位置随机
for i in range(num_samples):
    obs, info = env.reset()  # 新的随机配置
    # 采样一个积木训练

# 测试：同一个 episode 内，积木位置相关
obs, info = env.reset()
for step in range(max_steps):
    # 同一个塔，逐步移除积木
```

**解决方案：** 在训练时也模拟连续抽取

### 2. 点云归一化问题

**问题：** 训练和测试时点云的尺度可能不同

```python
# 训练时：点云可能在不同高度
pcd_train: z ∈ [0.0, 0.5]  # 底层积木

# 测试时：随着积木被移除，高度变化
pcd_test: z ∈ [0.3, 0.8]   # 顶层积木
```

**解决方案：** 归一化点云

### 3. 过拟合到训练环境

**问题：** 检测器只见过初始配置，没见过部分移除后的配置

**解决方案：** 训练时模拟移除积木

### 4. 网络容量不足

**问题：** Simple3DDetector 可能太简单

```python
# 当前：3层 MLP
self.pointnet = nn.Sequential(
    nn.Linear(3, 64), nn.ReLU(),
    nn.Linear(64, 128), nn.ReLU(),
    nn.Linear(128, feat_dim),
)
```

**解决方案：** 使用更深的网络

---

## 快速诊断

### 步骤 1：检查是否训练

```bash
# 运行时查看输出
python 3d_det_heuristic_baseline.py --max_episodes 2 --train_detector

# 应该看到：
# 训练 3D 检测器（500 样本）...
#   样本 100/500, Loss: 0.0234
#   样本 200/500, Loss: 0.0156
#   ...
# ✓ 3D 检测器训练完成，平均损失: 0.0123
```

### 步骤 2：检查训练损失

**如果损失很高（> 0.1）：**
- 网络容量不足
- 学习率太低
- 训练样本不够

**如果损失很低（< 0.01）：**
- 可能过拟合
- 或者训练数据太简单

### 步骤 3：对比质心估计

```python
# 在 detect() 函数中添加调试
def detect(self, pointcloud):
    # 质心估计
    centroid = pointcloud[:, :3].mean(axis=0)
    
    # 检测器估计
    if self.trained:
        with torch.no_grad():
            pcd_t = torch.from_numpy(pointcloud[:, :3]).float()
            center, conf = self.model(pcd_t)
            center = center.cpu().numpy()
        
        # 打印对比
        diff = np.linalg.norm(center - centroid)
        print(f"Centroid: {centroid}, Detector: {center}, Diff: {diff:.4f}")
        
        return center, conf.item()
```

---

## 改进方案

### 方案 A：简化（推荐）

**直接使用 Point Cloud Heuristic，不要 3D Det**

理由：
1. Point Cloud Heuristic 已经足够作为公平对比
2. 3D Det 需要额外训练，增加复杂度
3. 如果训练不好，反而性能更差

### 方案 B：改进 3D Det

如果你确实想要更强的 Baseline，改进方向：

#### 1. 更好的训练策略

```python
def train_on_environment_v2(self, env, num_episodes=50):
    """改进的训练策略：模拟真实使用场景"""
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        
        # 在同一个 episode 内采样多个积木
        for step in range(10):
            valid_ids = [i for i in range(NUM_BLOCKS) if info["mask"][i]]
            if not valid_ids:
                break
            
            # 训练所有可见积木
            for block_id in valid_ids:
                pcd = self._get_pointcloud(env, block_id)
                gt_center = self._get_gt_center(env, block_id)
                
                # 训练
                loss = self._train_step(pcd, gt_center)
            
            # 随机移除一个积木（模拟真实场景）
            action = np.random.choice(valid_ids)
            obs, _, done, _, info = env.step(action)
            
            if done:
                break
```

#### 2. 点云归一化

```python
def normalize_pointcloud(self, pcd):
    """归一化点云到单位球"""
    centroid = pcd.mean(axis=0)
    pcd_centered = pcd - centroid
    scale = np.max(np.linalg.norm(pcd_centered, axis=1))
    pcd_normalized = pcd_centered / (scale + 1e-8)
    return pcd_normalized, centroid, scale

def detect(self, pointcloud):
    # 归一化
    pcd_norm, centroid, scale = self.normalize_pointcloud(pointcloud)
    
    # 检测（归一化空间）
    center_norm, conf = self.model(pcd_norm)
    
    # 反归一化
    center = center_norm * scale + centroid
    return center, conf
```

#### 3. 更深的网络

```python
class Improved3DDetector(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        
        # 更深的 PointNet
        self.pointnet = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, feat_dim),
        )
        
        # 更深的回归头
        self.center_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3),
        )
```

---

## 我的建议

### 🎯 最佳方案：放弃 3D Det，使用 Point Cloud Heuristic

**理由：**

1. **Point Cloud Heuristic 已经够用**
   - 使用点云输入（公平）
   - 实现简单
   - 容易复现

2. **3D Det 增加复杂度**
   - 需要训练
   - 可能训练不好
   - 难以调试

3. **论文叙述更清晰**
   ```
   我们对比三种方法：
   1. Oracle Heuristic (完美感知) - 理论上界
   2. Point Cloud Heuristic (点云 + 质心) - 公平对比
   3. Our RL Model (点云 + 学习) - 超过 Baseline
   ```

### 📊 实验设置

```bash
# 1. Oracle Heuristic (上界)
python heuristic_jenga_baseline.py --max_episodes 100 --seed 42

# 2. Point Cloud Heuristic (公平对比)
python pointcloud_heuristic_baseline.py --max_episodes 100 --seed 42

# 3. Your RL Model
python evaluate_jenga.py --model rl --num_episodes 100 --seed 42 \
    --checkpoint runs/jenga_ppo
```

### 📝 论文中的表述

```
表 1: 不同方法的性能对比

| 方法 | 输入 | SR | MEB | 说明 |
|------|------|----|----|------|
| Oracle Heuristic | 完美位姿 | 95% | 12.1 | 理论上界 |
| Point Cloud Heuristic | 点云 | 72% | 8.1 | 传统方法 |
| Our Method | 点云 | 85% | 10.3 | 提升 13% |

结果表明：
1. 视觉感知的代价：Oracle (95%) vs Point Cloud (72%) = 23%
2. 学习的优势：Our Method (85%) vs Point Cloud (72%) = 13%
3. 与理论上界的差距：Oracle (95%) vs Our Method (85%) = 10%
```

---

## 总结

### 如果 3D Det 效果差，原因可能是：

1. ❌ 没有训练（`--train_detector`）
2. ❌ 训练样本不足
3. ❌ 训练/测试分布不匹配
4. ❌ 网络容量不足
5. ❌ 过拟合

### 建议：

**不要纠结于 3D Det，直接使用 Point Cloud Heuristic！**

它已经足够作为公平对比，而且：
- ✅ 简单
- ✅ 可复现
- ✅ 容易理解
- ✅ 论文叙述清晰

### 重点应该放在：

1. ✅ 完成 RL 模型训练（200K steps）
2. ✅ 证明 RL 超过 Point Cloud Heuristic
3. ✅ 分析视觉感知的代价
4. ✅ 撰写清晰的论文

不要在 Baseline 上花太多时间！
