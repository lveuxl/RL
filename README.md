# RL_RobotArm
A 6-DOF Robot arm simulation using ManiSkill (PickCube) environment with dense and sparse rewards

# ManiSkill 复杂堆叠抓取项目

本项目是从PyBullet环境迁移到ManiSkill环境的复杂堆叠场景机械臂抓取任务，使用PPO算法进行强化学习训练。

## 项目概述

### 任务描述
- **环境**: ManiSkill 3.0 仿真环境
- **任务**: 6自由度机械臂在复杂堆叠场景中进行智能物体选择和抓取
- **算法**: Proximal Policy Optimization (PPO)
- **特色**: 基于物体类别、尺寸、坐标、暴露度等多维特征的智能选择策略

### 主要功能
1. **多模态观测**: RGB-D图像 + 物体特征向量
2. **智能物体选择**: 基于暴露度、可抓取性、历史成功率的综合评分
3. **复杂场景**: 8-16个不同类别物体的随机堆叠
4. **并行训练**: 支持多环境并行训练加速
5. **完整评估**: 训练过程监控、模型评估、视频录制

## 项目结构

```
├── stack_picking_maniskill_env.py    # ManiSkill环境定义
├── ppo_maniskill_training.py         # PPO训练脚本
├── maniskill_config.py               # 配置文件
├── maniskill_utils.py                # 工具函数
├── maniskill_inference.py            # 推理评估脚本
├── requirements.txt                  # Python依赖
├── README.md                         # 项目说明
└── maniskill_outputs/                # 输出目录
    ├── videos/                       # 训练和评估视频
    ├── images/                       # 图像数据
    ├── episode_data/                 # Episode详细数据
    └── models/                       # 训练好的模型
```

## 安装说明

### 1. 环境要求
- Python 3.8+
- CUDA 11.0+ (推荐GPU训练)
- 8GB+ RAM
- 2GB+ GPU显存 (如使用GPU)

### 2. 安装依赖

```bash
# 克隆项目
git clone <your-repo-url>
cd maniskill-stacking-project

# 创建虚拟环境 (推荐)
python -m venv maniskill_env
source maniskill_env/bin/activate  # Linux/Mac
# 或 maniskill_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装ManiSkill (如果需要最新版本)
pip install mani-skill --upgrade
```

### 3. 验证安装

```python
import mani_skill
import torch
import stable_baselines3

print("ManiSkill版本:", mani_skill.__version__)
print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())
```

## 使用方法

### 1. 训练模型

```bash
# 基础训练 (默认参数)
python ppo_maniskill_training.py

# 自定义参数训练
python ppo_maniskill_training.py \
    --total_timesteps 1000000 \
    --num_envs 8 \
    --learning_rate 3e-4 \
    --obj_num 12 \
    --max_episode_steps 500

# 从检查点继续训练
python ppo_maniskill_training.py \
    --continue_training \
    --checkpoint_path maniskill_ppo_model/Checkpoint/model_checkpoint_100000.zip
```

### 2. 模型评估

```bash
# 基础评估
python maniskill_inference.py \
    --model_path maniskill_ppo_model/trained_model/maniskill_model_1.zip \
    --num_episodes 10

# 录制评估视频
python maniskill_inference.py \
    --model_path maniskill_ppo_model/trained_model/maniskill_model_1.zip \
    --num_episodes 5 \
    --record_video

# 批量评估多个模型
python -c "
from maniskill_inference import evaluate_multiple_models
evaluate_multiple_models('maniskill_ppo_model/trained_model', num_episodes=5)
"
```

### 3. 配置调整

编辑 `maniskill_config.py` 文件来调整训练参数:

```python
# 环境配置
ENV_CONFIG = {
    "obj_num": 12,              # 场景物体数量
    "max_objects": 16,          # 最大物体数量
    "circle_radius": 0.15,      # 物体分布半径
    # ... 其他配置
}

# PPO算法配置
PPO_CONFIG = {
    "learning_rate": 3e-4,      # 学习率
    "n_steps": 2048,            # 每次更新的步数
    "batch_size": 64,           # 批次大小
    # ... 其他配置
}
```

## 核心特性详解

### 1. 多模态观测空间

```python
observation_space = {
    "rgb": Box(0, 255, (224, 224, 3)),           # RGB图像
    "depth": Box(0, 10, (224, 224, 1)),          # 深度图像
    "features": Box(-np.inf, np.inf, (160,)),    # 物体特征向量
    "robot_state": Box(-np.inf, np.inf, (13,))   # 机械臂状态
}
```

### 2. 智能物体选择策略

```python
def select_best_object(object_infos, exposures, failure_counts):
    """
    综合评分策略:
    - 基础成功率 (根据物体类别)
    - 暴露度奖励 (位置高度 + 周围空间)
    - 失败惩罚 (历史失败次数)
    - 类别奖励 (物体形状适应性)
    """
    for obj_info in object_infos:
        graspability = calculate_object_graspability(
            obj_info['category'], 
            exposures[obj_info['id']], 
            failure_counts[obj_info['id']]
        )
        # 选择评分最高的物体
```

### 3. 奖励函数设计

```python
reward_components = {
    "object_selection": 2.0,     # 选择合适物体
    "grasp_success": 5.0,        # 成功抓取
    "grasp_failure": -1.0,       # 抓取失败
    "collision": -0.5,           # 碰撞惩罚
    "distance_bonus": 1.0,       # 接近奖励
    "stability_bonus": 0.5       # 稳定性奖励
}
```

## 训练监控

### 1. TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir maniskill_ppo_model/tensorboard

# 在浏览器中查看: http://localhost:6006
```

### 2. 关键指标

- **成功率**: 每个episode的抓取成功率
- **剩余物体数**: episode结束时未抓取的物体数量
- **总奖励**: episode累计奖励
- **episode长度**: 完成任务所需步数
- **物体选择质量**: 基于暴露度的选择评分

### 3. 日志文件

```
log/
├── hyperparameters/           # 训练配置记录
├── run_X/                     # 每次运行的详细日志
└── maniskill_evaluation_results_X.csv  # 评估结果
```

## 性能优化建议

### 1. 硬件配置
- **GPU训练**: 推荐使用CUDA兼容GPU，训练速度提升5-10倍
- **内存**: 16GB+推荐，支持更大批次和更多并行环境
- **存储**: SSD推荐，加快数据加载和模型保存

### 2. 训练参数调优

```python
# 快速原型验证
quick_config = {
    "total_timesteps": 100000,
    "num_envs": 4,
    "n_steps": 1024,
    "obj_num": 8
}

# 高质量训练
production_config = {
    "total_timesteps": 2000000,
    "num_envs": 16,
    "n_steps": 2048,
    "obj_num": 12
}
```

### 3. 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 单环境调试模式
debug_config = {
    "num_envs": 1,
    "render_mode": "human",
    "record_video": True
}
```

## 常见问题解决

### 1. 安装问题

**Q: ManiSkill安装失败**
```bash
# 尝试从源码安装
pip install git+https://github.com/haosulab/ManiSkill.git

# 或使用conda
conda install -c conda-forge mani-skill
```

**Q: CUDA版本不兼容**
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 训练问题

**Q: 训练速度慢**
- 增加并行环境数量 (`num_envs`)
- 使用GPU训练
- 减少观测图像分辨率
- 优化网络结构

**Q: 收敛困难**
- 调整学习率
- 增加训练步数
- 检查奖励函数设计
- 调整网络架构

### 3. 环境问题

**Q: 仿真不稳定**
- 检查物理参数设置
- 调整仿真频率
- 验证场景构建逻辑

## 扩展开发

### 1. 添加新物体类型

```python
# 在maniskill_config.py中添加
CATEGORY_MAP["NewObject"] = 5

ENV_CONFIG["object_properties"]["NewObject"] = {
    "dimensions": [0.05, 0.05, 0.08],
    "mass": 0.1,
    "color": [0.8, 0.2, 0.2],
    "base_success_rate": 0.7
}
```

### 2. 自定义奖励函数

```python
def custom_reward_function(self, action, info):
    """自定义奖励函数"""
    reward = 0.0
    
    # 添加你的奖励逻辑
    if info.get("custom_condition"):
        reward += 1.0
    
    return reward
```

### 3. 新的观测模式

```python
def get_custom_observation(self):
    """添加新的观测信息"""
    obs = self._get_obs()
    
    # 添加自定义观测
    obs["custom_feature"] = self._compute_custom_feature()
    
    return obs
```

## 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- [ManiSkill](https://github.com/haosulab/ManiSkill) - 机器人仿真环境
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - 强化学习算法库
- [SAPIEN](https://sapien.ucsd.edu/) - 物理仿真引擎

## 联系方式

如有问题或建议，请通过以下方式联系:
- 提交Issue: [GitHub Issues](your-repo-issues-url)
- 邮箱: your-email@example.com

---

**注意**: 本项目基于PyBullet项目迁移而来，保持了原有的核心算法逻辑，同时充分利用了ManiSkill的GPU加速和并行化优势。
