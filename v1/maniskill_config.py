# ManiSkill环境配置文件 - 从PyBullet项目迁移

# 物体类别映射 - 参考PyBullet项目的category_map
CATEGORY_MAP = {
    "CrackerBox": 1,
    "GelatinBox": 2, 
    "PottedMeatCan": 3,
    "MustardBottle": 4
}

# 环境配置 - 参考PyBullet项目的ENV_CONFIG
ENV_CONFIG = {
    # 物体数量配置
    "obj_num": 16,
    "max_objects": 16,
    
    # 场景配置
    "circle_center": [1.0, -0.2],  # 物体分布的圆心位置
    "circle_radius": 0.15,         # 物体分布的圆形区域半径
    
    # 相机配置 - 参考PyBullet项目的相机参数
    "camera_config": {
        "width": 1408,               # 图像宽度
        "height": 1024,              # 图像高度
        "fov": 50,                   # 相机视角
        "near": 0.01,                # 近裁剪面
        "far": 20,                   # 远裁剪面
        "cam_distance": 0.58,        # 相机距离
        "cam_yaw": -90,              # 相机偏航角
        "cam_pitch": -89,            # 相机俯仰角
        "cam_roll": 0,               # 相机滚转角
        "cam_targetPos": [1.0, -0.2, 1.3],  # 相机目标位置
        "cam_up_axis_idx": 2,        # 相机上轴索引
    },
    
    # 物体类别信息
    "category_map": CATEGORY_MAP,
    
    # 物体物理属性配置
    "object_properties": {
        "CrackerBox": {
            "dimensions": [0.16, 0.21, 0.07],  # 长宽高
            "mass": 0.411,                     # 质量(kg)
            "color": [0.9, 0.7, 0.3, 1],      # RGBA颜色
            "base_success_rate": 0.8,          # 基础抓取成功率
        },
        "GelatinBox": {
            "dimensions": [0.028, 0.085, 0.073],
            "mass": 0.097,
            "color": [1, 0.2, 0.2, 1],        # 红色
            "base_success_rate": 0.9,
        },
        "PottedMeatCan": {
            "radius": 0.051,                   # 圆柱体半径
            "height": 0.097,                   # 圆柱体高度
            "mass": 0.370,
            "color": [0.7, 0.7, 0.7, 1],      # 灰色
            "base_success_rate": 0.7,
        },
        "MustardBottle": {
            "radius": 0.031,
            "height": 0.179,
            "mass": 0.603,
            "color": [1, 1, 0, 1],            # 黄色
            "base_success_rate": 0.6,
        }
    },
    
    # 奖励函数配置 - 参考PyBullet项目的奖励设置
    "reward_config": {
        "success_reward": 10.0,               # 成功抓取奖励
        "failure_penalty": -2.0,              # 失败惩罚
        "displacement_penalty_factor": -0.5,   # 位移惩罚系数
        "time_penalty": -0.1,                 # 时间惩罚
        "exposure_bonus_factor": 2.0,         # 暴露度奖励系数
        "collision_penalty": -1.0,            # 碰撞惩罚
    },
    
    # 训练配置
    "training_config": {
        "max_episode_steps": 16,              # 每个episode最大步数
        "num_envs": 16,                        # 并行环境数量
        "total_timesteps": 50000,             # 总训练步数
        "eval_episodes": 10,                  # 评估episode数量
    },
    
    # 暴露度计算配置
    "exposure_config": {
        "ray_directions": 8,                  # 射线方向数量
        "ray_length": 0.5,                    # 射线长度
        "height_weight": 0.5,                 # 高度权重
        "distance_weight": 0.5,               # 距离权重
        "min_exposure": 0.0,                  # 最小暴露度
        "max_exposure": 1.0,                  # 最大暴露度
    },
    
    # 机械臂配置
    "robot_config": {
        "robot_type": "panda",                # 机械臂类型：panda 或 fetch
        "control_mode": "pd_ee_delta_pose",   # 控制模式
        "robot_position": [-0.615, 0, 0],    # 机械臂初始位置
    },
    
    # 场景构建配置
    "scene_config": {
        "use_table": True,                    # 是否使用桌子
        "table_height": 0.8,                  # 桌子高度
        "workspace_bounds": {                 # 工作空间边界
            "x_min": 0.5, "x_max": 1.5,
            "y_min": -0.7, "y_max": 0.3,
            "z_min": 0.8, "z_max": 1.5
        }
    },
    
    # 数据收集配置 - 参考PyBullet项目的数据保存
    "data_config": {
        "save_images": True,                  # 是否保存图像
        "save_annotations": True,             # 是否保存标注
        "save_depths": True,                  # 是否保存深度图
        "output_dir": "maniskill_outputs",   # 输出目录
        "image_format": "png",                # 图像格式
    }
}

# PPO算法配置 - 参考PyBullet项目的PPO参数
PPO_CONFIG = {
    # 基础PPO参数
    "policy": "MlpPolicy",
    "n_steps": 256,                           # 每次更新收集的步数
    "batch_size": 128,                        # 批次大小
    "learning_rate": 5e-4,                    # 学习率
    "gamma": 0.995,                           # 折扣因子
    "n_epochs": 8,                            # 每次更新的训练轮数
    "clip_range": 0.15,                       # PPO剪切范围
    "ent_coef": 0.01,                         # 熵系数
    "vf_coef": 0.25,                          # 价值函数系数
    "max_grad_norm": 0.3,                     # 梯度剪切
    "gae_lambda": 0.95,                       # GAE lambda参数
    
    # 网络架构配置
    "policy_kwargs": {
        "net_arch": [128, 128, 64],           # 网络架构
        "activation_fn": "tanh",              # 激活函数
    },
    
    # 回调配置
    "checkpoint_freq": 2000,                  # 检查点保存频率
    "log_freq": 100,                          # 日志记录频率
    "eval_freq": 5000,                        # 评估频率
}

# 观测空间配置
OBSERVATION_CONFIG = {
    # 特征维度配置
    "features_per_object": 10,                # 每个物体的特征数量
    "max_objects": 16,                        # 最大物体数量
    "global_features": 3,                     # 全局特征数量
    "total_features": 16 * 10 + 3,           # 总特征数量：163
    
    # 物体特征构成
    "object_features": {
        "position": 4,                        # 位置特征：x, y, z, 距中心距离
        "category": 4,                        # 类别特征：one-hot编码
        "exposure": 1,                        # 暴露度特征
        "failure_count": 1,                   # 失败次数特征
    },
    
    # 全局特征构成
    "global_features_composition": {
        "step_progress": 1,                   # 步数进度
        "remaining_objects": 1,               # 剩余物体数
        "success_rate": 1,                    # 成功率
    }
}

# 动作空间配置
ACTION_CONFIG = {
    "action_type": "discrete",                # 动作类型：离散
    "num_actions": 16,                        # 动作数量：选择物体ID
    "action_description": "选择要抓取的物体ID (0-15)",
}

# 日志和可视化配置
LOGGING_CONFIG = {
    # 日志目录
    "log_dir": "log",
    "tensorboard_dir": "maniskill_ppo_model/tensorboard",
    "checkpoint_dir": "maniskill_ppo_model/Checkpoint",
    "model_dir": "maniskill_ppo_model/trained_model",
    
    # 日志级别和频率
    "log_level": "INFO",
    "print_freq": 100,                        # 控制台输出频率
    "save_freq": 1000,                        # 模型保存频率
    
    # 统计信息配置
    "metrics_to_track": [
        "episode_reward",
        "success_count", 
        "success_rate",
        "remaining_objects",
        "total_distance",
        "episode_length"
    ]
}

# 评估配置
EVALUATION_CONFIG = {
    "eval_episodes": 10,                      # 评估episode数量
    "eval_deterministic": True,               # 是否使用确定性策略评估
    "eval_render": False,                     # 是否渲染评估过程
    "eval_save_video": True,                  # 是否保存评估视频
    "eval_metrics": [
        "mean_reward",
        "success_rate", 
        "mean_episode_length",
        "mean_remaining_objects"
    ]
} 