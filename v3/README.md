# 堆叠抓取强化学习训练系统

一个基于ManiSkill和Stable-Baselines3的智能抓取顺序学习系统。

## 任务描述

- **环境**: 9个物体分3层堆叠（3x3网格布局）
- **目标**: 学习自上而下的最优抓取顺序，避免抓取被遮挡的物体
- **动作空间**: 离散动作，每次选择一个物体进行抓取
- **奖励设计**: 
  1. 🎯 成功抓取奖励 (+10)
  2. 🏔️ 抓取顶层物体额外奖励 (+5)  
  3. ❌ 抓取被遮挡物体惩罚 (-15)
  4. ⏱️ 时间惩罚 (-0.1/步)
  5. 🏆 完成所有物体奖励 (+50)

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install torch stable-baselines3 mani-skill gymnasium tensorboard

# 激活conda环境 (如需要)
conda activate -p /path/to/your/env
```

### 2. 开始训练

```bash
# 方式1: 使用启动脚本 (推荐)
python run_training.py

# 方式2: 直接运行训练脚本
python train_optimized.py --mode train
```

### 3. 监控训练进度

```bash
# 在另一个终端中启动tensorboard
tensorboard --logdir ./tensorboard_logs
```

然后在浏览器中打开 `http://localhost:6006` 查看训练曲线。

### 4. 评估模型

```bash
# 评估最佳模型
python run_training.py --eval

# 或评估特定模型
python train_optimized.py --mode eval --model_path ./models/checkpoints/ppo_clutter_1000000_steps
```

## 训练配置

### 核心超参数
- **并行环境数**: 256个 (高效并行采样)
- **学习率**: 3e-4 → 0 (线性衰减)
- **批次大小**: 2048
- **PPO更新轮数**: 10
- **网络结构**: [256, 256] 全连接网络
- **总训练步数**: 2,000,000步

### 优化特性
- ⚡ **快速仿真**: 简化FSM，降低仿真频率
- 📚 **课程学习**: 从简单到复杂逐步增加难度  
- 🎯 **智能奖励**: 强化自上而下抓取策略
- 💾 **自动保存**: 定期保存检查点和最佳模型
- 📊 **详细记录**: Tensorboard记录训练和评估指标

## 文件结构

```
v3/
├── env_clutter_optimized.py    # 优化版环境实现
├── train_optimized.py          # 主训练脚本
├── run_training.py             # 便捷启动脚本
├── models/                     # 模型保存目录
│   ├── best_model/            # 最佳模型
│   ├── checkpoints/           # 训练检查点
│   └── final_model.zip        # 最终训练模型
├── tensorboard_logs/          # Tensorboard日志
└── logs/                      # 评估日志
```

## 预期结果

经过充分训练后，智能体应该学会：

1. ✅ **自上而下策略**: 优先抓取顶层物体
2. ✅ **避免遮挡**: 不选择被其他物体遮挡的目标
3. ✅ **高效顺序**: 在9步内抓取所有物体
4. ✅ **稳定收敛**: Loss和Reward曲线平滑收敛

### 性能指标
- **成功率**: >95%
- **平均奖励**: >40分/episode  
- **平均步数**: ~9步/episode
- **训练时间**: ~2-4小时 (GPU)

## 故障排除

### 常见问题

**Q: 训练过程中GPU内存不足？**
A: 减少`num_envs`从256改为128或64

**Q: 收敛速度太慢？**  
A: 检查奖励函数设计，确保引导信号足够强

**Q: 模型不学习自上而下策略？**
A: 增大层级奖励权重`REWARD_TOP_LAYER`或遮挡惩罚`PENALTY_OCCLUSION`

**Q: Tensorboard没有数据？**
A: 确保tensorboard指向正确目录: `tensorboard --logdir ./tensorboard_logs`

## 进阶配置

要调整训练参数，请修改`train_optimized.py`中的`config`字典：

```python
config = {
    "num_envs": 128,        # 减少内存占用
    "learning_rate": 1e-4,  # 更保守的学习率
    "total_timesteps": 1_000_000,  # 更短训练时间
    # ... 其他参数
}
```

## 技术细节

- **状态空间**: 59维向量 (物体特征45维 + 全局状态5维 + 动作掩码9维)
- **动作空间**: 离散空间，9个可选动作 (对应9个物体)
- **奖励函数**: 密集奖励，实时反馈抓取质量
- **终止条件**: 抓取完所有物体或达到最大步数

---
🎯 **目标**: 训练出能够自上而下高效抓取堆叠物体的智能策略！