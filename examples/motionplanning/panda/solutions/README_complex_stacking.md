# 复杂堆叠环境Motion Planning实现指南

## 🎯 核心设计理念

基于**奥卡姆剃刀原理**，我们创建了一个最小化、高效的复杂堆叠Motion Planning解决方案：

- **分层抽象**：将复杂堆叠分解为原子操作序列
- **动态规划**：实时更新碰撞约束和空间状态  
- **智能重试**：多策略并行尝试，提升成功率
- **零冗余**：每一行代码都有明确作用，无多余抽象

## 🏗️ 项目架构解析

### 核心文件结构
```
solutions/
├── complex_stacking_solver.py     # 核心求解器
├── complex_stacking_example.py    # 使用示例  
└── README_complex_stacking.md     # 本文档
```

### 核心类层次
```python
PandaArmMotionPlanningSolver          # 基础Motion Planning
    ↓
ComplexStackingMotionPlanner          # 复杂堆叠封装
    ↓  
DynamicObstacleComplexPlanner         # 动态障碍物扩展
```

## 🚀 快速开始

### 1. 基础集成

```python
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.solutions.complex_stacking_solver import (
    ComplexStackingMotionPlanner, StackingTarget
)

# 初始化
base_planner = PandaArmMotionPlanningSolver(env, vis=True)
complex_planner = ComplexStackingMotionPlanner(base_planner)

# 定义堆叠序列
stacking_sequence = [
    StackingTarget(source_obj=cubeA, target_obj=cubeB, stack_height=0.02)
]

# 执行堆叠
success = complex_planner.solve_complex_stacking(stacking_sequence)
```

### 2. 你的堆叠环境集成

```python
def solve_your_complex_stacking(env, seed=None, debug=False, vis=False):
    """适配你的复杂堆叠环境"""
    env.reset(seed=seed)
    
    # 创建求解器
    base_planner = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis,
        base_pose=env.unwrapped.agent.robot.pose
    )
    complex_planner = ComplexStackingMotionPlanner(base_planner)
    
    # 获取你的环境中的物体
    objects = env.get_all_objects()  # 替换为你的物体获取方法
    
    # 根据你的堆叠需求定义序列
    stacking_sequence = create_your_stacking_plan(objects)
    
    # 执行堆叠
    success = complex_planner.solve_complex_stacking(stacking_sequence)
    
    base_planner.close()
    return success

def create_your_stacking_plan(objects):
    """根据你的需求创建堆叠计划"""
    return [
        StackingTarget(
            source_obj=objects[i+1], 
            target_obj=objects[i],
            stack_height=0.02 * (i+1),  # 逐渐增加高度
            approach_angles=np.linspace(0, 2*np.pi, 8)  # 8个尝试角度
        ) 
        for i in range(len(objects)-1)
    ]
```

## 🔧 高级功能

### 1. 动态碰撞检测

```python
# 添加静态障碍物
complex_planner.planner.add_box_collision(
    extents=np.array([0.1, 0.1, 0.2]), 
    pose=sapien.Pose([0.2, 0, 0.1])
)

# 添加点云障碍物
obstacle_points = your_point_cloud_function()
complex_planner.planner.add_collision_pts(obstacle_points)
```

### 2. 多策略抓取

```python
stacking_target = StackingTarget(
    source_obj=cube,
    target_obj=base,
    stack_height=0.05,
    approach_angles=np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])  # 5个候选角度
)
```

### 3. 容错机制

```python
# 自动回退到安全状态
try:
    success = complex_planner.solve_complex_stacking(sequence)
except Exception as e:
    print(f"堆叠失败: {e}")
    complex_planner._rollback_to_safe_state()
```

## 🎨 预设堆叠模式

### 金字塔模式
```python
targets = create_complex_stacking_plan(objects, stacking_pattern="pyramid")
```

### 塔式模式  
```python
targets = create_complex_stacking_plan(objects, stacking_pattern="tower")
```

### 桥式模式
```python
targets = create_complex_stacking_plan(objects, stacking_pattern="bridge")
```

## ⚡ 性能优化建议

### 1. 抓取角度优化
- **简单堆叠**：使用4个主要方向 `[0, π/2, π, 3π/2]`
- **复杂堆叠**：使用8-12个候选角度
- **精确堆叠**：使用密集角度采样

### 2. 碰撞检测优化
```python
# 点云密度控制
def _sample_object_surface(self, obb, n_points=128):  # 减少点云密度提升速度
    # ...

# 选择性碰撞更新
def _update_collision_environment(self):
    # 只为关键物体添加碰撞约束
    if len(self.execution_history) > 3:
        # 只保留最近3个物体的碰撞约束
        relevant_items = self.execution_history[-3:]
```

### 3. 路径规划策略
- **开放空间**：优先使用螺旋运动（速度快）
- **复杂环境**：使用RRTConnect（避障能力强）
- **混合策略**：先尝试螺旋运动，失败后切换到RRTConnect

## 🧪 测试与调试

### 可视化调试
```python
# 启用详细可视化
solver = ComplexStackingMotionPlanner(base_planner)
solver.planner.debug = True
solver.planner.vis = True
solver.planner.visualize_target_grasp_pose = True
```

### 性能基准测试
```python
from complex_stacking_example import benchmark_complex_stacking_algorithms
benchmark_complex_stacking_algorithms()
```

## 📝 在你的项目中的集成步骤

### Step 1: 环境适配
1. 确保你的环境继承自ManiSkill的BaseEnv
2. 确保机器人是Panda臂或兼容的7自由度机械臂
3. 物体需要提供pose属性和collision mesh

### Step 2: 物体识别
```python
def get_stackable_objects(env):
    """获取环境中可堆叠的物体"""
    return [obj for obj in env.actors if obj.name.startswith("cube")]
```

### Step 3: 堆叠序列设计
```python
def design_stacking_sequence(objects, complexity_level="medium"):
    """根据复杂度级别设计堆叠序列"""
    if complexity_level == "simple":
        return create_tower_pattern(objects[:2])
    elif complexity_level == "medium":
        return create_pyramid_pattern(objects[:3])
    else:
        return create_custom_pattern(objects)
```

### Step 4: 异常处理
```python
def robust_stacking_execution(complex_planner, sequence, max_retries=3):
    """带重试机制的稳健执行"""
    for attempt in range(max_retries):
        try:
            if complex_planner.solve_complex_stacking(sequence):
                return True
        except Exception as e:
            print(f"尝试 {attempt+1} 失败: {e}")
            complex_planner._rollback_to_safe_state()
    return False
```

## 💡 最佳实践

1. **渐进式复杂度**：从简单的双物体堆叠开始，逐步增加复杂度
2. **安全边界**：为物体间保留足够的安全距离（通常0.02m）
3. **角度选择**：优先尝试与物体主轴对齐的抓取角度
4. **失败恢复**：每次失败后都回退到已知的安全状态
5. **性能监控**：记录每个步骤的执行时间和成功率

## 🔗 扩展方向

- **机器学习集成**：使用RL训练最优抓取策略
- **多机械臂协作**：扩展到双臂协同堆叠
- **实时感知**：集成视觉感知进行自适应堆叠
- **物理验证**：添加堆叠稳定性物理验证

---

*🎯 这个方案基于深度理解现有ManiSkill Motion Planning架构，采用最小化设计原则，确保每一行代码都有明确作用，无冗余抽象。*

