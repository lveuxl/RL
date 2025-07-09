# EnvClutter环境开发对话记录

## 日期
2024年当前日期

## 问题描述
用户实现了一个自定义的ManiSkill环境 `EnvClutter-v1`，该环境基于官方的 `pick_clutter_ycb` 和 `pick_cube` 环境。环境包含3种YCB物体的复杂堆叠场景，目标是设计奖励函数来优先考虑成功抓取、最小化其他物体位移和缩短完成时间。

## 遇到的错误
初始错误：`"Tensors must have same number of dimensions: got 2 and 1"`

## 解决方案

### 1. 环境代码修复 (env_clutter.py)
- 修复了张量维度不匹配问题
- 改进了 `_get_obs_extra` 方法，确保所有观测数据维度一致
- 修复了 `_initialize_episode` 方法中的张量设备和维度问题
- 改进了 `_sample_target_objects` 方法的错误处理
- 修复了构造函数中的参数传递问题

### 2. 训练代码修复 (training.py)
- 修复了 `flatten_obs` 函数，正确处理 `torch.Tensor` 类型的观测
- 改进了 `PPOAgent` 类的 `get_action` 和 `get_value` 方法
- 修复了 `update` 方法中的张量转换问题
- 添加了数据类型处理，确保标量值的正确处理
- 添加了可视化支持和调试信息

### 3. 测试和可视化
- 创建了 `test_visualization.py` 脚本用于测试可视化界面
- 确认环境可以正常创建和运行
- 验证了可视化界面正常工作

## 关键修复点

### 张量维度问题
```python
# 修复前
goal_pos = torch.zeros((b, 3))
# 修复后  
goal_pos = torch.zeros((b, 3), device=self.device)
```

### 观测数据处理
```python
# 修复前
if isinstance(obs, torch.Tensor):
    return torch.from_numpy(obs).flatten()  # 错误：Tensor不能用from_numpy
# 修复后
if isinstance(obs, torch.Tensor):
    return obs.flatten()
```

### 数据类型转换
```python
# 修复前
return action.cpu().numpy(), log_prob.cpu().numpy()
# 修复后
return action.cpu().numpy(), log_prob.item()
```

## 托盘功能开发过程

### 4. 托盘集成问题及解决方案

#### 问题1：托盘URDF文件解析错误
**错误信息**：`ValueError: 托盘URDF文件中没有找到articulation`

**原因分析**：
- 托盘URDF文件(`traybox.urdf`)定义为单链接静态对象
- 没有关节定义，无法作为articulation加载
- 需要使用静态actor方式加载

**解决方案**：
```python
def _load_tray(self):
    # 使用actor_builders方式加载托盘
    actor_builders = parsed_result.get("actor_builders", [])
    if not actor_builders:
        raise ValueError("托盘URDF文件中没有找到actor_builders")
    
    # 使用build_static创建静态托盘
    tray = builder.build_static(name=f"tray_{env_idx}")
```

#### 问题2：物体生成位置超出托盘边界
**问题描述**：物体在托盘外部生成，不符合预期

**原因分析**：
- 托盘中心位置：`[0.1, 0.0, 0.02]`
- 托盘尺寸：0.6×0.6米，边界墙在±0.25米处
- 原始生成区域：`[0.28, 0.28]`过大
- 实际生成范围：x轴从-0.18到0.38，超出托盘边界(-0.15到0.35)

**解决方案**：
```python
def _generate_object_position_in_tray(self, stack_level=0):
    # 托盘边界计算（基于URDF文件中的边界墙位置）
    # 边界墙在托盘中心的±0.25米处，考虑边界墙厚度0.02米
    # 实际可用空间：从中心向两边各0.23米（留出安全边距）
    safe_spawn_area_x = 0.23
    safe_spawn_area_y = 0.23
    
    # 在托盘内随机生成xy位置
    x = tray_center_x + random.uniform(-safe_spawn_area_x, safe_spawn_area_x)
    y = tray_center_y + random.uniform(-safe_spawn_area_y, safe_spawn_area_y)
```

#### 问题3：观测数据维度不匹配
**错误信息**：`IndexError: too many indices for tensor of dimension 2`

**解决方案**：
```python
def test_tray_environment():
    # 修正观测数据访问方式
    if isinstance(obs, dict) and 'qpos' in obs:
        print(f"观测数据类型: {type(obs['qpos'])}")
        print(f"观测数据形状: {obs['qpos'].shape}")
    else:
        print(f"观测数据结构: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
```

### 5. YCB物体类型更新

#### 原始物体类型：
- `004_sugar_box` (糖盒)
- `006_mustard_bottle` (芥末瓶)
- `008_pudding_box` (布丁盒)

#### 更新后的物体类型（用户要求）：
- `003_cracker_box` (饼干盒)
- `009_gelatin_box` (明胶盒)
- `010_potted_meat_can` (罐装肉罐头)
- `006_mustard_bottle` (芥末瓶)
- `004_sugar_box` (糖盒)
- `013_apple_jello_box` (苹果果冻盒，替代布丁盒)

#### 物体尺寸标准化：
```python
def _get_object_size(self, obj_type):
    # 基于YCB数据集的实际物体尺寸（单位：米）
    sizes = {
        "003_cracker_box": [0.16, 0.21, 0.07],         # 饼干盒: 16cm x 21cm x 7cm
        "009_gelatin_box": [0.028, 0.085, 0.114],      # 明胶盒: 2.8cm x 8.5cm x 11.4cm
        "010_potted_meat_can": [0.101, 0.051, 0.051],  # 罐装肉罐头: 10.1cm x 5.1cm x 5.1cm
        "006_mustard_bottle": [0.095, 0.095, 0.177],   # 芥末瓶: 9.5cm x 9.5cm x 17.7cm
        "004_sugar_box": [0.09, 0.175, 0.044],         # 糖盒: 9cm x 17.5cm x 4.4cm
        "013_apple_jello_box": [0.078, 0.109, 0.032],  # 苹果果冻盒: 7.8cm x 10.9cm x 3.2cm
    }
    return sizes.get(obj_type, [0.05, 0.05, 0.05])
```

### 6. 托盘静态设置

#### 问题：托盘移动问题
**用户需求**：托盘应保持静态，不可移动

**解决方案**：
```python
def _load_tray(self):
    # 设置托盘的物理属性
    loader.fix_root_link = True  # 固定托盘不动
    
    # 使用build_static创建静态托盘，确保不会移动
    tray = builder.build_static(name=f"tray_{env_idx}")
```

## 最终完整的托盘功能特性

### 托盘规格：
- **尺寸**：0.6×0.6×0.15米
- **位置**：[0.1, 0.0, 0.02]（机器人前方桌面上）
- **物理属性**：静态摩擦0.8，动态摩擦0.6，恢复系数0.1
- **状态**：完全静态，不可移动

### 物体生成：
- **生成区域**：托盘中心±0.23米范围内
- **堆叠支持**：支持多层堆叠（每层3cm间隔）
- **边界安全**：确保物体不会生成在托盘边界外

### 测试验证：
- **环境创建**：成功
- **托盘加载**：成功
- **物体生成**：位置正确，在托盘内部
- **可视化**：正常工作

## 环境特性
- **状态维度**: 43
- **动作维度**: 7
- **物体类型**: 6种YCB物体
- **每种物体数量**: 5个
- **奖励模式**: dense/sparse
- **控制模式**: pd_ee_delta_pose

## 奖励函数设计
1. **接近奖励** (权重2.0) - 优先级最高
2. **抓取奖励** (权重3.0)
3. **放置奖励** (权重2.0)
4. **位移惩罚** (权重1.5) - 优先级第二
5. **时间惩罚** (权重0.01) - 优先级第三
6. **静止奖励** (权重1.0)
7. **成功奖励** (权重10.0)

## 使用方法

### 基本测试
```bash
python test_tray_env.py
```

### 训练（带可视化）
```bash
python training.py --epochs 10 --steps_per_epoch 50 --render
```

### 训练（无可视化）
```bash
python training.py --epochs 1000 --steps_per_epoch 2048
```

## 文件结构
- `env_clutter.py` - 自定义环境实现（包含托盘功能）
- `training.py` - PPO训练脚本
- `test_tray_env.py` - 托盘环境测试脚本
- `config.py` - 配置文件
- `inference.py` - 推理脚本
- `utils.py` - 工具函数
- `assets/tray/traybox.urdf` - 托盘URDF文件

## 成功验证
- ✅ 环境创建成功
- ✅ 托盘功能集成成功
- ✅ 物体生成位置正确（在托盘内部）
- ✅ 6种YCB物体类型支持
- ✅ 物体尺寸标准化
- ✅ 托盘静态设置正确
- ✅ 可视化界面正常工作
- ✅ 训练脚本运行正常
- ✅ 张量维度问题已解决
- ✅ 数据类型转换问题已解决

## 技术亮点

### 1. 托盘URDF文件处理
- 正确识别单链接静态对象
- 使用actor_builders方式加载
- 确保托盘完全静态

### 2. 精确的边界计算
- 基于URDF文件的实际尺寸
- 考虑边界墙厚度和安全边距
- 确保物体生成在托盘内部

### 3. 多物体类型支持
- 6种不同的YCB物体
- 基于真实数据集的尺寸
- 灵活的物体生成策略

### 4. 完整的测试覆盖
- 环境创建测试
- 托盘加载测试
- 物体位置生成测试
- 可视化功能测试

## 注意事项
1. 确保`assets/tray/traybox.urdf`文件存在
2. 托盘位置固定在`[0.1, 0.0, 0.02]`
3. 物体生成区域限制在托盘内部
4. 使用`--render`参数开启可视化
5. 确保设备一致性 (CPU/GPU)
6. 注意多环境和单环境的数据处理差异
7. 张量操作时要注意设备和维度匹配

## 未来改进方向
1. 支持动态调整托盘位置
2. 添加更多YCB物体类型
3. 实现更复杂的堆叠策略
4. 优化物体碰撞检测
5. 添加托盘倾斜功能 