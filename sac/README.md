# 🤖 SAC机器人抓取项目

## 📖 项目概述

这是一个基于**SAC (Soft Actor-Critic)** 强化学习算法的机器人视觉抓取项目。项目使用ManiSkill仿真环境，实现了从视觉图像到机器人动作控制的端到端学习，支持完整的训练-数据收集-推理流程。

### 🎯 核心特性
- **🔄 SAC算法**：实现了完整的Soft Actor-Critic算法，支持连续动作空间
- **👁️ 多模态感知**：融合RGB视觉信息和机器人状态信息
- **🚀 高效训练**：支持GPU并行环境和经验回放
- **📊 专家数据**：智能收集高质量轨迹用于预训练
- **🎬 可视化**：支持训练过程可视化和结果视频录制

---

## 🏗️ 系统架构

### 核心组件架构图
```
┌─────────────────────────────────────────────────────────────────┐
│                        SAC强化学习系统                           │
├─────────────────────────────────────────────────────────────────┤
│  🎯 任务环境层                                                  │
│  ├── ManiSkill仿真环境 (PickCube-v1)                           │
│  ├── RGB视觉观察 (32x32 或 64x64)                              │
│  ├── 机器人状态信息 (关节位置、速度等)                          │
│  └── 连续动作控制 (End-Effector姿态)                           │
├─────────────────────────────────────────────────────────────────┤
│  🧠 算法核心层                                                  │
│  ├── Actor网络: 视觉编码器 + MLP → 动作分布                     │
│  ├── 双Critic网络: Q1, Q2 (减少过估计偏差)                     │
│  ├── 目标网络: 软更新机制 (τ=0.01)                              │
│  └── 自适应熵调节: 自动调整探索-利用平衡                        │
├─────────────────────────────────────────────────────────────────┤
│  💾 数据管理层                                                  │
│  ├── 经验回放缓冲区 (支持字典观察)                              │
│  ├── 专家数据收集 (过滤成功轨迹)                                │
│  └── 高效采样机制                                               │
├─────────────────────────────────────────────────────────────────┤
│  🔧 工具支持层                                                  │
│  ├── 模型检查点管理                                             │
│  ├── TensorBoard日志记录                                       │
│  ├── 视频录制功能                                               │
│  └── 评估和推理工具                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构详解

```
sac/
├── 🎯 核心算法文件
│   ├── sac_rgb_base.py      # SAC主算法实现 (完整版)
│   └── sac_replay.py        # SAC算法实现 (优化版)
├── 📊 数据收集工具
│   ├── fill_buffer.py       # 智能专家数据收集 (成功轨迹过滤)
│   └── fill_buffer_no_filter.py  # 简单数据收集 (无过滤)
├── 🔍 推理评估工具
│   └── infer_sac.py         # 模型推理和性能评估
├── 💾 模型和数据
│   ├── best_chkpts/         # 最佳模型检查点
│   │   ├── dense_ckpt_440k_43.pt   # 稠密奖励训练模型
│   │   └── sparse_ckpt_240k_39.pt  # 稀疏奖励训练模型
│   └── result_videos/       # 结果演示视频
│       ├── dense/           # 稠密奖励结果
│       └── sparse/          # 稀疏奖励结果
└── 📋 辅助文件
    ├── obs_img.jpg          # 观察样例图像
    └── README.md            # 项目说明文档
```

---

## 🧠 算法核心详解

### 1. SAC (Soft Actor-Critic) 原理

SAC是一种基于最大熵强化学习的现代算法，核心思想是在最大化累积奖励的同时保持策略的高熵性：

```python
# 核心目标函数
J(π) = Σ E_τ~π [r(s,a) + α·H(π(·|s))]
```

#### 🎭 Actor网络架构
```python
class Actor(nn.Module):
    # 🖼️ 视觉编码器
    encoder: PlainConv (RGB → 256维特征)
    # 🧠 特征融合
    mlp: [256+状态维度] → [512, 256]
    # 🎯 策略输出
    fc_mean: 256 → 动作维度
    fc_logstd: 256 → 动作维度
```

**特色功能**：
- **多模态融合**：RGB图像 + 机器人状态
- **重参数化技巧**：确保梯度传播
- **动作缩放**：映射到实际动作范围

#### 🎯 Critic网络架构
```python
class SoftQNetwork(nn.Module):
    encoder: 共享视觉编码器
    mlp: [视觉特征+状态+动作] → [512, 256, 1]
```

**双Q网络设计**：减少Q值过估计问题

### 2. 视觉编码器设计

#### 🖼️ PlainConv卷积网络
```python
CNN架构 (适配不同图像尺寸):
├── Conv2d(in_channels, 16, 3×3) + ReLU + MaxPool
├── Conv2d(16, 32, 3×3) + ReLU + MaxPool  
├── Conv2d(32, 64, 3×3) + ReLU + MaxPool
├── Conv2d(64, 64, 3×3) + ReLU + MaxPool
├── Conv2d(64, 64, 1×1) + ReLU
└── FC层 → 256维特征向量
```

**智能尺寸适配**：
- 128×128 → 4×4特征图 (1024维)
- 64×64 → 4×4特征图 (1024维) 
- 32×32 → 2×2特征图 (256维)

### 3. 经验回放机制

#### 📦 DictArray数据结构
```python
# 支持嵌套字典观察的高效存储
DictArray:
├── RGB图像: [buffer_size, num_envs, H, W, 3]
├── 状态信息: [buffer_size, num_envs, state_dim]  
├── 动作: [buffer_size, num_envs, action_dim]
├── 奖励: [buffer_size, num_envs]
└── 终止标志: [buffer_size, num_envs]
```

**内存优化特性**：
- 数据类型智能转换 (float32/uint8/int32)
- GPU/CPU设备灵活切换
- 批量高效采样

---

## ⚙️ 关键超参数配置

### 🎛️ sac_rgb_base.py 配置
```python
# 🌐 环境配置
env_id = "PickCube-v1"           # ManiSkill抓取任务
num_envs = 16                    # 并行环境数量
camera_width/height = 32         # RGB图像分辨率
control_mode = "pd_ee_delta_pos" # 末端执行器位置控制

# 🧠 训练配置  
total_timesteps = 400_000        # 总训练步数
buffer_size = 100_000           # 经验回放缓冲区大小
batch_size = 256                # 训练批次大小
learning_starts = 4000          # 开始训练前的探索步数

# 🎯 算法参数
gamma = 0.8                     # 折扣因子
tau = 0.01                      # 软更新参数  
policy_lr = 3e-4                # Actor学习率
q_lr = 3e-4                     # Critic学习率
alpha = 0.2                     # 熵权重 (可自适应)
utd = 0.5                       # 更新频率比例
```

### 🎛️ sac_replay.py 优化配置
```python
# 📈 性能优化版本
num_envs = 64                   # 更多并行环境
buffer_size = 300_000           # 更大经验缓冲区
batch_size = 512                # 更大训练批次
camera_width/height = 64        # 更高图像分辨率
```

---

## 🚀 使用指南

### 1. 🏋️ 模型训练

#### 基础训练
```bash
# 标准配置训练
cd sac/
python sac_rgb_base.py

# 高性能配置训练  
python sac_replay.py
```

#### 训练流程解析
1. **🔄 环境初始化**: 创建并行仿真环境
2. **🎲 随机探索阶段**: 收集初始经验 (前4000步)
3. **📚 学习阶段**: 
   - 环境交互收集数据
   - 从缓冲区采样训练
   - 定期评估性能
   - 保存最佳模型
4. **📊 评估循环**: 每10000步评估一次

### 2. 📊 专家数据收集

#### 智能过滤收集 (推荐)
```bash
python fill_buffer.py
```

**核心逻辑**：
```python
def get_winner_envs(success_buffer):
    """筛选成功环境"""
    # 分析每个环境最后10步的成功率
    # 只保留成功率>40%的环境轨迹
    
def collect_data():
    """收集专家数据"""
    # 1. 加载已训练模型
    # 2. 运行50步episode  
    # 3. 过滤失败环境
    # 4. 仅保存成功轨迹到buffer
```

#### 无过滤收集
```bash  
python fill_buffer_no_filter.py
```

### 3. 🔍 模型评估和推理

```bash
python infer_sac.py
```

**推理功能**：
- 📹 自动录制演示视频
- 📊 性能指标统计
- 🎯 成功率分析
- 💾 结果保存到 `./results/`

### 4. 📈 训练监控

#### TensorBoard可视化
```bash
tensorboard --logdir runs/
```

**监控指标**：
- 📊 损失曲线 (Actor/Critic Loss)  
- 🎯 成功率变化
- ⏱️ 训练效率指标
- 🎲 熵值变化 (探索程度)

---

## 🏆 性能基准

### 训练成果对比

| 模型类型 | 训练步数 | 成功率 | 特点 |
|---------|----------|--------|------|
| **稠密奖励模型** | 440K | 43% | 连续奖励信号，收敛快 |
| **稀疏奖励模型** | 240K | 39% | 仅成功时奖励，更符合实际 |

### 🎯 任务表现分析

**PickCube-v1 任务特点**：
- **目标**: 抓取红色立方体到目标位置
- **观察**: 32×32 RGB + 29维状态向量  
- **动作**: 7维连续动作 (位置+方向控制)
- **成功条件**: 距离<0.025m 且机器人静止

**性能指标**：
- **平均回合奖励**: 2.5~4.2
- **平均成功步数**: 35~45步
- **训练稳定性**: 收敛稳定，无明显抖动

---

## 🔧 技术创新点

### 1. 🎨 多模态感知融合
```python
# 创新的观察融合机制
def forward(self, obs):
    # 视觉特征提取
    visual_feature = encoder(obs["rgb"])
    # 状态特征融合  
    fused_feature = concat([visual_feature, obs["state"]])
    return mlp(fused_feature)
```

### 2. 🎯 智能成功轨迹筛选
```python
# 基于滑动窗口的成功率评估
def get_winner_envs(success_buffer, window_size=10):
    success_rate = success_buffer[-window_size:].mean()
    return env_ids[success_rate > 0.4]
```

### 3. 📚 渐进式经验回放  
- **冷启动**: 专家数据预填充
- **在线学习**: 实时经验收集
- **质量控制**: 动态奖励加权

### 4. 🔄 自适应熵调节
```python
# 自动调整探索-利用平衡
if autotune:
    target_entropy = -action_dim  # 启发式设定
    alpha_loss = -(log_alpha * (log_pi + target_entropy)).mean()
```

---

## 🐛 常见问题和解决方案

### 1. ⚠️ 内存溢出问题
```python
# 解决方案: 调整缓冲区大小和批次大小
buffer_size = 50_000  # 减小缓冲区
batch_size = 128      # 减小批次
```

### 2. 🐌 训练速度慢
```python  
# 解决方案: 优化并行环境数量
num_envs = min(64, GPU_memory // 200MB)  # 根据显存调整
```

### 3. 📉 训练不收敛
```python
# 解决方案: 调整学习率和熵权重
policy_lr = 1e-4     # 降低学习率
alpha = 0.1          # 调整熵权重
```

### 4. 🎯 模型检查点加载错误  
```python
# 解决方案: 非严格加载模式
actor.load_state_dict(checkpoint['actor'], strict=False)
```

---

## 📚 核心代码解析

### 🧠 SAC训练循环
```python
while global_step < total_timesteps:
    # 🔄 数据收集阶段
    for local_step in range(steps_per_env):
        if not learning_has_started:
            actions = envs.action_space.sample()  # 随机探索
        else:
            actions, _, _, _ = actor.get_action(obs)  # 策略采样
        
        next_obs, rewards, terms, truncs, infos = envs.step(actions)
        rb.add(obs, next_obs, actions, rewards, terms)
        obs = next_obs
    
    # 🎓 学习更新阶段  
    if global_step >= learning_starts:
        for _ in range(grad_steps_per_iteration):
            data = rb.sample(batch_size)
            
            # Critic更新
            with torch.no_grad():
                next_actions, next_log_pi, _, _ = actor.get_action(data.next_obs)
                target_q = rewards + gamma * (1-dones) * (
                    torch.min(qf1_target(next_obs, next_actions), 
                             qf2_target(next_obs, next_actions)) - alpha * next_log_pi
                )
            
            qf1_loss = F.mse_loss(qf1(obs, actions), target_q)  
            qf2_loss = F.mse_loss(qf2(obs, actions), target_q)
            
            # Actor更新
            if global_update % policy_frequency == 0:
                pi, log_pi, _, _ = actor.get_action(data.obs)
                actor_loss = (alpha * log_pi - torch.min(
                    qf1(obs, pi), qf2(obs, pi))).mean()
            
            # 软更新目标网络
            if global_update % target_network_frequency == 0:
                soft_update(qf1_target, qf1, tau)
                soft_update(qf2_target, qf2, tau)
```

### 🎯 智能数据收集逻辑
```python
def collect_expert_data():
    """收集高质量专家轨迹"""
    success_buffer = []
    
    for episode in range(num_episodes):
        # 运行完整episode (50步)
        for step in range(50):
            action = actor.get_eval_action(obs)
            obs, reward, done, info = env.step(action)
            
            # 记录每步成功状态
            success_buffer.append(info['success'])
        
        # 分析最后10步成功率
        recent_success_rate = success_buffer[-10:].mean()
        
        # 只保存高成功率环境的轨迹  
        if recent_success_rate > 0.4:
            add_trajectory_to_buffer()
        else:
            discard_trajectory()
```

---

## 🌟 未来改进方向

### 1. 🚀 算法优化
- **TD3集成**: 结合Twin Delayed DDPG的稳定性优势
- **优先经验回放**: 基于TD误差的重要性采样
- **多任务学习**: 扩展到多种抓取任务

### 2. 🎨 网络架构升级  
- **注意力机制**: 增强视觉特征提取能力
- **残差连接**: 提升深度网络训练稳定性
- **图神经网络**: 处理复杂空间关系

### 3. 📊 数据效率提升
- **课程学习**: 从简单到复杂的任务安排
- **数据增强**: RGB图像的几何和颜色变换
- **元学习**: 快速适应新环境和任务

### 4. 🔧 工程优化
- **分布式训练**: 多GPU并行加速
- **模型压缩**: 轻量化部署
- **实物迁移**: Sim2Real技术

---

## 📞 技术支持

### 🆘 常见错误排查
1. **CUDA内存不足**: 减少`num_envs`和`batch_size`
2. **模型不收敛**: 调整学习率和奖励设计  
3. **环境版本不匹配**: 确保ManiSkill版本兼容

### 📖 参考资料
- [SAC原论文](https://arxiv.org/abs/1812.05905)
- [ManiSkill环境文档](https://maniskill.readthedocs.io/)
- [PyTorch深度学习教程](https://pytorch.org/tutorials/)

---

**🎉 祝你训练成功，机器人抓取任务完美达成！**

*Last updated: 2025年1月*
