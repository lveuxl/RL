# 复杂堆叠杂乱环境 (Complex Stacking Clutter Environment)

基于ManiSkill和Stable Baselines3的复杂堆叠杂乱环境，用于机器人操作的强化学习研究。该环境集成了暴露度计算、智能物体选择和复杂堆叠场景。

## 主要特性

- **智能物体选择**: 基于物体类别、暴露度和可抓取性的智能选择算法
- **暴露度计算**: 3D射线投射计算物体的暴露程度
- **复杂堆叠场景**: 支持多物体堆叠和复杂布局
- **课程学习**: 渐进式难度提升的训练策略
- **丰富的可视化**: 训练过程监控和结果分析
- **完整的评估系统**: 详细的性能分析和报告生成

## 文件结构

```
├── env_clutter.py          # 环境定义
├── config.py              # 配置参数
├── training.py            # 训练脚本
├── inference.py           # 推理脚本
├── utils.py               # 工具函数
├── README.md              # 使用说明
└── requirements.txt       # 依赖包
```

## 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖

- `mani_skill`: ManiSkill机器人操作环境
- `stable-baselines3`: PPO强化学习算法
- `torch`: PyTorch深度学习框架
- `gymnasium`: OpenAI Gym环境接口
- `numpy`: 数值计算
- `matplotlib`: 数据可视化
- `wandb`: 实验跟踪和可视化

## 快速开始

### 1. 训练模型

```bash
# 基础训练
python training.py

# 自定义参数训练
python training.py --total-timesteps 1000000 --num-envs 8 --learning-rate 3e-4

# 恢复训练
python training.py --resume-training --model-path ./models/ppo_model_checkpoint.zip

# 启用课程学习
python training.py --enable-curriculum --curriculum-stages 3
```

### 2. 模型推理

```bash
# 基础推理
python inference.py --model-path ./models/ppo_model_final.zip

# 详细推理（带可视化）
python inference.py --model-path ./models/ppo_model_final.zip --render --save-video --verbose

# 大规模评估
python inference.py --model-path ./models/ppo_model_final.zip --n-episodes 100 --deterministic
```

### 3. 配置测试

```bash
# 测试配置
python config.py

# 测试工具函数
python utils.py
```

## 配置说明

### 环境配置

在`config.py`中可以调整以下参数：

```python
ENV_CONFIG = {
    "scene_config": {
        "workspace_bounds": [[-0.3, -0.3, 0.0], [0.3, 0.3, 0.5]],
        "max_objects": 16,
        "min_objects": 8,
        "stacking_probability": 0.3,
    },
    "exposure_config": {
        "ray_directions": 26,  # 射线方向数量
        "ray_length": 1.0,     # 射线长度
        "min_clearance": 0.05, # 最小间隙
    },
    "reward_config": {
        "success_reward": 10.0,
        "failure_penalty": -1.0,
        "time_penalty": -0.01,
    }
}
```

### 训练配置

```python
TRAINING_CONFIG = {
    "total_timesteps": 1000000,
    "num_envs": 4,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "n_epochs": 10,
}
```

## 环境详解

### ComplexStackingClutterEnv

主要环境类，继承自ManiSkill的BaseEnv：

- **观察空间**: 包含机器人状态、物体位置、暴露度等信息
- **动作空间**: 离散动作空间，对应不同的抓取策略
- **奖励函数**: 基于成功率、效率和智能选择的复合奖励

### 关键方法

```python
# 计算物体暴露度
exposure = env.calculate_exposure(object_position, object_size)

# 智能选择目标物体
target_object = env.select_optimal_target()

# 评估抓取可行性
graspability = env.calculate_graspability(object_properties)
```

## 训练策略

### 课程学习

系统支持三阶段课程学习：

1. **简单阶段**: 少量物体，简单布局
2. **中等阶段**: 中等数量物体，部分堆叠
3. **困难阶段**: 最大物体数量，复杂堆叠

### 奖励设计

- **成功奖励**: 成功抓取物体获得正奖励
- **智能选择奖励**: 选择高暴露度物体获得额外奖励
- **效率奖励**: 减少无效动作的时间惩罚
- **堆叠奖励**: 成功处理堆叠场景的额外奖励

## 可视化和分析

### 训练监控

使用Weights & Biases进行实时监控：

```python
# 在training.py中自动启用
wandb.init(project="complex-stacking-clutter")
```

### 结果分析

```python
# 生成训练曲线
plot_training_curves("./logs/", "./plots/training_curves.png")

# 物体统计分析
plot_object_statistics(object_data, "./plots/object_stats.png")

# 暴露度热力图
plot_exposure_heatmap(scene_data, "./plots/exposure_heatmap.png")
```

## 性能优化

### GPU加速

```python
# 在config.py中启用GPU
ENV_CONFIG["gpu_memory_gb"] = 4.0
```

### 并行环境

```python
# 增加并行环境数量
python training.py --num-envs 16
```

### 内存优化

```python
# 调整批次大小
python training.py --batch-size 128
```

## 扩展和自定义

### 添加新物体类别

在`config.py`中添加新的物体属性：

```python
OBJECT_PROPERTIES["new_category"] = {
    "dimensions": [0.05, 0.05, 0.1],
    "mass": 0.2,
    "base_success_rate": 0.7,
    "shape_type": "box",
    "grasp_difficulty": "medium"
}
```

### 自定义奖励函数

在`env_clutter.py`中修改`calculate_reward`方法：

```python
def calculate_reward(self, action, success, info):
    # 自定义奖励逻辑
    reward = 0.0
    if success:
        reward += self.success_reward
    # 添加其他奖励组件
    return reward
```

### 新的评估指标

在`utils.py`中添加新的分析函数：

```python
def analyze_custom_metric(results):
    # 自定义分析逻辑
    pass
```

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减少`num_envs`或`batch_size`
2. **训练不收敛**: 调整学习率或增加训练时间
3. **环境加载失败**: 检查ManiSkill安装和资源文件

### 调试模式

```bash
# 启用详细输出
python training.py --verbose

# 单环境调试
python training.py --num-envs 1 --render
```

## 实验建议

### 超参数调优

建议的超参数搜索范围：

- 学习率: [1e-4, 3e-4, 1e-3]
- 批次大小: [64, 128, 256]
- 环境数量: [4, 8, 16]

### 消融研究

测试不同组件的影响：

```bash
# 禁用智能选择
python training.py --disable-intelligent-selection

# 不同奖励模式
python training.py --reward-mode sparse
```

## 引用

如果您在研究中使用了这个环境，请引用：

```bibtex
@misc{complex-stacking-clutter-env,
  title={Complex Stacking Clutter Environment for Robot Manipulation},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 联系方式

如有问题请联系：your.email@example.com
