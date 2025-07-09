# EnvClutter 环境

一个基于ManiSkill的复杂堆叠物品抓取环境，支持PPO算法训练和推理。

## 环境描述

EnvClutter是一个机器人抓取环境，任务是在复杂的堆叠场景中抓取指定的物品。环境特点：

- **复杂场景**：包含多种YCB物品的随机堆叠
- **多样化物品**：支持盒子、罐子、瓶子等不同类型的物品
- **智能奖励**：结合抓取成功、物品位移和时间效率的综合奖励函数
- **丰富观测**：提供物品类别、尺寸、中心坐标、暴露面积等详细信息

## 项目结构

```
├── env_clutter.py      # 环境实现
├── training.py         # PPO训练脚本
├── inference.py        # 推理脚本
├── config.py          # 配置文件
├── utils.py           # 工具函数
├── README.md          # 项目说明
└── requirements.txt   # 依赖包列表
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- `mani_skill`：机器人仿真环境
- `torch`：深度学习框架
- `gymnasium`：强化学习环境接口
- `numpy`：数值计算
- `opencv-python`：图像处理
- `matplotlib`：数据可视化

## 使用方法

### 1. 训练模型

使用PPO算法训练智能体：

```bash
# 基础训练
python training.py --epochs 1000 --steps_per_epoch 2048

# 自定义配置训练
python training.py --config custom --epochs 2000 --lr_actor 3e-4 --lr_critic 1e-3

# 多GPU训练
python training.py --device cuda --num_envs 16 --epochs 1000
```

训练参数说明：
- `--epochs`：训练轮数
- `--steps_per_epoch`：每轮步数
- `--num_envs`：并行环境数量
- `--lr_actor`：Actor学习率
- `--lr_critic`：Critic学习率
- `--config`：配置预设（default/fast/robust）
- `--log_dir`：日志保存目录
- `--model_dir`：模型保存目录

### 2. 推理演示

使用训练好的模型进行推理：

```bash
# 单次演示
python inference.py --model_path ./models/ppo_model.pth --mode demo --render

# 批量评估
python inference.py --model_path ./models/ppo_model.pth --mode eval --num_episodes 100

# 性能基准测试
python inference.py --model_path ./models/ppo_model.pth --mode benchmark --num_episodes 50

# 交互式演示
python inference.py --model_path ./models/ppo_model.pth --mode interactive
```

推理参数说明：
- `--model_path`：模型文件路径
- `--mode`：运行模式（demo/eval/benchmark/interactive）
- `--num_episodes`：评估episode数量
- `--record_video`：是否录制视频
- `--video_dir`：视频保存目录
- `--render`：是否实时渲染

### 3. 配置管理

配置文件支持多种预设和自定义：

```python
from config import get_config, Config

# 使用预设配置
config = get_config('default')  # 默认配置
config = get_config('fast')     # 快速训练配置
config = get_config('robust')   # 鲁棒训练配置

# 自定义配置
config = Config()
config.env.num_envs = 8
config.training.epochs = 2000
config.model.hidden_dim = 256

# 保存和加载配置
config.save('my_config.json')
config = Config.load('my_config.json')
```

### 4. 工具函数

项目提供了丰富的工具函数：

```python
from utils import (
    setup_seed,           # 设置随机种子
    VideoRecorder,        # 视频录制
    RewardTracker,        # 奖励追踪
    PerformanceProfiler,  # 性能分析
    evaluate_model,       # 模型评估
    visualize_training_progress  # 训练可视化
)

# 设置随机种子
setup_seed(42)

# 录制视频
recorder = VideoRecorder('./videos')
recorder.start_recording()
# ... 运行环境 ...
recorder.stop_recording('demo.mp4')

# 追踪奖励
tracker = RewardTracker()
tracker.add_episode(reward, success, length)
stats = tracker.get_stats()
```

## 环境详细说明

### 观测空间

环境提供丰富的观测信息：

- **机器人状态**：关节位置、速度、末端执行器状态
- **物品信息**：
  - 类别：盒子、罐子、瓶子等
  - 尺寸：长、宽、高
  - 位置：中心坐标(x, y, z)
  - 暴露面积：可见表面积比例
- **场景信息**：相机图像、深度信息、分割掩码

### 动作空间

连续动作空间，包括：
- 末端执行器位置控制(x, y, z)
- 末端执行器姿态控制(roll, pitch, yaw)
- 抓取器开合控制

### 奖励函数

多层次奖励设计：

1. **主要奖励**：成功抓取目标物品 (+10.0)
2. **位移惩罚**：其他物品位移 (-0.1 × 位移距离)
3. **时间惩罚**：步数惩罚 (-0.01 × 步数)
4. **接近奖励**：接近目标物品 (+0.1 × 距离减少)

### 成功条件

- 成功抓取目标物品
- 物品离开桌面一定高度
- 保持抓取状态一定时间

## 性能优化

### 训练优化

1. **并行环境**：使用多个并行环境加速数据收集
2. **批处理**：合理设置batch_size平衡内存和性能
3. **学习率调度**：使用学习率衰减提高训练稳定性
4. **梯度裁剪**：防止梯度爆炸

### 推理优化

1. **模型压缩**：使用较小的网络结构进行推理
2. **批量推理**：同时处理多个观测
3. **GPU加速**：使用GPU进行推理加速

## 实验建议

### 训练策略

1. **分阶段训练**：
   - 第一阶段：简单场景，少量物品
   - 第二阶段：复杂场景，多种物品
   - 第三阶段：困难场景，密集堆叠

2. **课程学习**：
   - 逐步增加物品数量
   - 逐步增加堆叠复杂度
   - 逐步减少奖励提示

3. **多任务学习**：
   - 同时训练多种物品类型
   - 混合不同难度的场景

### 评估指标

- **成功率**：完成任务的比例
- **平均奖励**：每个episode的平均奖励
- **平均步数**：完成任务的平均步数
- **位移惩罚**：其他物品的平均位移
- **收敛速度**：达到目标性能的训练轮数

## 故障排除

### 常见问题

1. **环境无法创建**：
   - 检查ManiSkill安装
   - 确认GPU驱动和CUDA版本
   - 验证依赖包版本

2. **训练不收敛**：
   - 调整学习率
   - 增加训练时间
   - 检查奖励函数设计

3. **推理性能差**：
   - 检查模型加载
   - 验证环境配置一致性
   - 确认随机种子设置

4. **内存不足**：
   - 减少并行环境数量
   - 降低batch_size
   - 使用梯度累积

### 调试技巧

1. **日志分析**：查看训练日志找出问题
2. **可视化**：使用渲染和视频录制观察行为
3. **性能分析**：使用PerformanceProfiler分析瓶颈
4. **逐步调试**：从简单场景开始逐步增加复杂度

## 扩展开发

### 添加新物品类型

1. 在`env_clutter.py`中添加新的物品定义
2. 更新`_get_object_size`方法
3. 修改观测空间维度

### 自定义奖励函数

1. 修改`compute_dense_reward`方法
2. 添加新的奖励组件
3. 调整奖励权重

### 集成新算法

1. 实现新的智能体类
2. 修改训练脚本
3. 更新配置文件

## 引用

如果您在研究中使用了这个项目，请引用：

```bibtex
@misc{envclutter2024,
  title={EnvClutter: A Complex Stacking Environment for Robotic Manipulation},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/envclutter}}
}
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱：your.email@example.com
- GitHub Issues：[项目Issues页面]
- 讨论区：[项目Discussions页面]
