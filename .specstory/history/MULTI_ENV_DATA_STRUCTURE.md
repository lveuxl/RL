# 多环境并行系统数据结构设计文档

## 概述

本文档详细说明了EnvClutter环境中多环境并行时所有变量的存储逻辑和结构设计。该系统采用了**混合存储策略**，既保证了GPU并行计算的效率，又维持了环境间的逻辑隔离。

## 核心设计原则

### 1. 混合存储策略
- **全局扁平化存储** + **环境分组索引** + **张量化状态管理**
- GPU负责物理仿真，CPU负责逻辑控制
- 热数据上GPU，冷数据留CPU

### 2. 环境隔离保证
- 每个环境的状态完全独立，避免竞态条件
- 通过`scene_idxs`确保物理仿真的环境隔离
- 原子操作保证状态更新的一致性

# 轻松创建不同规模的环境
env = EnvClutterEnv(config_preset="large_scene")  # 18个物体
env = EnvClutterEnv(config_preset="small_scene")  # 6个物体

# 配置会自动计算相关参数
config.env.num_object_types = len(config.env.box_objects)
config.env.total_objects_per_env = config.env.num_objects_per_type * config.env.num_object_types

## 数据结构架构

### 第一层：全局物体池

```python
# 扁平化存储所有环境的所有物体
self.all_objects = []  
# 结构：[env0_obj0, env0_obj1, ..., env0_objN, env1_obj0, env1_obj1, ..., env1_objN, ...]
```

**存储逻辑：**
- 所有环境的物体按环境顺序依次追加到同一个列表中
- 每个物体通过`builder.set_scene_idxs([env_idx])`绑定到特定环境
- 最终通过`Actor.merge(self.all_objects, name="all_objects")`合并为GPU可并行处理的张量

**创建过程：**
```python
# 为每个环境创建物体
for env_idx in range(self.num_envs):
    env_objects = []
    # 创建每种类型的物体
    for obj_type in self.BOX_OBJECTS:
        for i in range(self.num_objects_per_type):
            # 创建物体并设置环境索引
            builder.set_scene_idxs([env_idx])
            obj = builder.build(name=f"env_{env_idx}_{obj_type}_{i}")
            env_objects.append(obj)
    
    # 扩展到全局列表
    self.all_objects.extend(env_objects)

# 合并为GPU张量
if self.all_objects:
    self.merged_objects = Actor.merge(self.all_objects, name="all_objects")
```

### 第二层：环境分组索引

```python
# 按环境分组的物体列表
self.selectable_objects = []  
# 结构：[[env0_objects], [env1_objects], [env2_objects], ...]
```

**存储逻辑：**
- 每个环境维护独立的物体列表
- `self.selectable_objects[env_idx]`直接访问特定环境的物体
- 用于环境内的相对索引计算和物体选择

**创建过程：**
```python
for env_idx in range(self.num_envs):
    env_selectable = []
    # 为当前环境创建物体列表
    for obj_type in self.BOX_OBJECTS:
        for i in range(self.num_objects_per_type):
            obj = builder.build(name=f"env_{env_idx}_{obj_type}_{i}")
            env_selectable.append(obj)
    
    # 添加到环境分组列表
    self.selectable_objects.append(env_selectable)
```

### 第三层：元数据分组存储

```python
# 按环境分组的物体元数据
self.object_info = []  
# 结构：[[env0_info], [env1_info], [env2_info], ...]
```

**存储逻辑：**
- 存储每个物体的静态信息（类型、尺寸、初始位置等）
- 按环境分组，便于查询和管理
- 用于物体特征计算和暴露面积分析

**数据结构：**
```python
obj_info = {
    'type': obj_type,                    # 物体类型
    'size': self._get_object_size(obj_type),  # 物体尺寸 [w, h, d]
    'initial_pose': initial_pose,        # 初始位姿
    'center': [x, y, z],                # 中心位置
    'exposed_area': 1.0,                # 暴露面积
}
```

## 状态管理系统

### 环境级状态变量（Python列表）

```python
# 每个环境独立的状态列表
self.remaining_indices = []    # [[env0_indices], [env1_indices], ...]
self.step_count = []          # [env0_steps, env1_steps, ...]  
self.grasped_objects = []     # [[env0_grasped], [env1_grasped], ...]
self.is_suction_active = []   # [env0_active, env1_active, ...]
self.current_suction_object = [] # [env0_obj, env1_obj, ...]
```

**特点：**
- 使用Python列表存储，减少GPU内存占用
- 每个环境的状态完全独立
- 适合存储变长数据和复杂对象引用

### 并行状态变量（PyTorch张量）

```python
# GPU并行处理的张量状态
self.env_stage = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)      # [num_envs]
self.env_target = torch.full((self.num_envs,), -1, dtype=torch.int16, device=self.device) # [num_envs]
self.env_busy = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)      # [num_envs]
self.stage_tick = torch.zeros(self.num_envs, dtype=torch.int16, device=self.device)   # [num_envs]
self.stage_positions = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device) # [num_envs, 3]
```

**特点：**
- 使用PyTorch张量，利用GPU并行计算
- 固定长度数据，适合向量化操作
- 支持批量更新和原子操作

## 索引映射系统

### 双重索引体系

#### 1. 全局索引系统
```python
# 用于物理仿真和GPU并行计算
global_idx = env_idx * self.total_objects_per_env + local_idx
obj = self.all_objects[global_idx]
```

#### 2. 相对索引系统
```python
# 用于环境内逻辑和动作选择
obj = self.selectable_objects[env_idx][local_idx]
# 相对索引范围：[0, total_objects_per_env-1]
```

### 索引转换逻辑

```python
# 动作选择时：使用相对索引
pick = int(action[i])  # 智能体选择的相对索引
target_idx = self.remaining_indices[i][pick]  # 获取环境内的相对索引
target_obj = self.selectable_objects[env_idx][target_idx]  # 通过相对索引获取物体

# 物理仿真时：自动使用全局索引
obj.pose.p  # SAPIEN自动根据scene_idxs处理多环境并行
```

## 配置化参数系统

### 可配置参数

```python
# 在config.py中定义
@dataclass
class EnvConfig:
    # 物体配置
    box_objects: List[str] = None           # 物体类型列表
    num_objects_per_type: int = 4           # 每种类型的物体数量
    num_object_types: int = 3               # 物体类型数量
    total_objects_per_env: int = 12         # 每个环境的总物体数量
    goal_thresh: float = 0.03               # 成功阈值
    
    # 离散动作相关配置
    max_episode_steps_discrete: int = 15    # 最大抓取尝试次数
```

### 动态计算

```python
def __post_init__(self):
    # 自动计算相关参数
    self.num_object_types = len(self.box_objects)
    self.total_objects_per_env = self.num_objects_per_type * self.num_object_types
```

### 预设配置

```python
PRESET_CONFIGS = {
    "large_scene": {
        "env": {
            "num_objects_per_type": 6,      # 每种类型6个物体
            "total_objects_per_env": 18,    # 总共18个物体
            "max_episode_steps_discrete": 20,
        }
    },
    "small_scene": {
        "env": {
            "num_objects_per_type": 2,      # 每种类型2个物体
            "total_objects_per_env": 6,     # 总共6个物体
            "max_episode_steps_discrete": 8,
        }
    }
}
```

## 内存布局优化

### 数据分层策略

#### 热数据（频繁访问）
- **存储位置**：GPU内存（PyTorch张量）
- **数据类型**：`env_stage`, `env_busy`, `stage_positions`
- **优化目标**：利用GPU并行计算加速

#### 冷数据（偶尔访问）
- **存储位置**：CPU内存（Python列表）
- **数据类型**：`remaining_indices`, `grasped_objects`
- **优化目标**：减少GPU内存占用

#### 元数据（静态信息）
- **存储位置**：CPU内存（嵌套字典/列表）
- **数据类型**：`object_info`, `selectable_objects`
- **优化目标**：便于查询和管理

## 并发安全设计

### 环境隔离机制

1. **物理隔离**：通过`scene_idxs`确保物体只属于特定环境
2. **状态隔离**：每个环境的状态变量完全独立
3. **索引隔离**：相对索引系统避免环境间干扰

### 原子操作保证

```python
# 批量状态更新，保证原子性
self.env_stage[env_indices] = new_stages
self.env_busy[env_indices] = new_busy_states
```

### 同步机制

```python
# 所有环境同时更新，保持同步
for env_idx in range(self.num_envs):
    if self.env_busy[env_idx]:
        cmd[env_idx] = self._pick_object_step(env_idx)

# 统一执行仿真步
super().step(cmd)
```

## 性能优化要点

### 1. 存储分离
- **物理数据**：使用张量存储，支持GPU并行
- **逻辑数据**：使用列表存储，灵活高效

### 2. 计算并行
- **GPU**：负责物理仿真和张量运算
- **CPU**：负责逻辑控制和索引管理

### 3. 索引双轨
- **全局索引**：用于物理仿真的批量操作
- **相对索引**：用于环境内的逻辑决策

### 4. 内存分层
- **热数据**：放在GPU上，快速访问
- **冷数据**：留在CPU上，节省GPU内存

## 使用示例

### 基本使用

```python
from config import get_config
from env_clutter import EnvClutterEnv

# 使用默认配置
env = EnvClutterEnv(
    num_envs=4,
    use_discrete_action=True,
    config_preset="default"
)

# 使用大场景配置
env = EnvClutterEnv(
    num_envs=8,
    use_discrete_action=True,
    config_preset="large_scene"
)
```

### 自定义配置

```python
from config import Config

# 创建自定义配置
config = Config()
config.env.num_objects_per_type = 5
config.env.total_objects_per_env = 15
config.env.max_episode_steps_discrete = 18

# 使用自定义配置
env = EnvClutterEnv(
    num_envs=4,
    use_discrete_action=True,
    custom_config=config
)
```

### 访问物体数据

```python
# 访问全局物体池
all_objects = env.all_objects  # 所有环境的所有物体

# 访问特定环境的物体
env_0_objects = env.selectable_objects[0]  # 环境0的物体

# 访问物体元数据
env_0_info = env.object_info[0]  # 环境0的物体信息

# 访问环境状态
grasped_count = len(env.grasped_objects[0])  # 环境0已抓取的物体数量
remaining_count = len(env.remaining_indices[0])  # 环境0剩余可抓取的物体数量
```

## 扩展性考虑

### 1. 支持不同物体数量
- 通过配置文件轻松调整物体数量
- 自动计算相关参数，保持一致性

### 2. 支持不同环境数量
- 状态变量自动适应环境数量
- 内存使用线性扩展

### 3. 支持新的物体类型
- 只需在配置中添加新的物体类型
- 系统自动处理物体创建和管理

### 4. 支持新的状态变量
- 遵循热/冷数据分层原则
- 选择合适的存储方式（张量vs列表）

## 总结

这种数据结构设计的核心优势：

1. **高性能**：GPU并行 + CPU逻辑控制的混合架构
2. **高可扩展性**：配置化参数 + 模块化设计
3. **高可维护性**：清晰的分层结构 + 完善的文档
4. **高可靠性**：环境隔离 + 原子操作保证

该设计既保证了多环境并行训练的高效性，又维持了代码的可读性和可维护性，是一个优雅且实用的工程解决方案。
