"""
复杂堆叠杂乱环境配置文件
包含物体属性、场景配置、奖励参数、暴露度计算等配置
"""

# 物体类别映射
CATEGORY_MAP = {
    "002_master_chef_can": 1,
    "003_cracker_box": 2,
    "004_sugar_box": 3,
    "005_tomato_soup_can": 4,
    "006_mustard_bottle": 5,
    "007_tuna_fish_can": 6,
    "008_pudding_box": 7,
    "009_gelatin_box": 8,
    "010_potted_meat_can": 9,
    "011_banana": 10,
    "019_pitcher_base": 11,
    "021_bleach_cleanser": 12,
    "024_bowl": 13,
    "025_mug": 14,
    "035_power_drill": 15,
    "036_wood_block": 16,
    "037_scissors": 17,
    "040_large_marker": 18,
    "051_large_clamp": 19,
    "052_extra_large_clamp": 20,
    "061_foam_brick": 21,
}

# 物体属性配置
OBJECT_PROPERTIES = {
    # 易抓取物体
    "002_master_chef_can": {
        "dimensions": [0.102, 0.102, 0.142],
        "mass": 0.414,
        "base_success_rate": 0.8,
        "shape_type": "cylinder",
        "grasp_difficulty": "easy"
    },
    "005_tomato_soup_can": {
        "dimensions": [0.067, 0.067, 0.101],
        "mass": 0.349,
        "base_success_rate": 0.85,
        "shape_type": "cylinder",
        "grasp_difficulty": "easy"
    },
    "007_tuna_fish_can": {
        "dimensions": [0.084, 0.084, 0.032],
        "mass": 0.171,
        "base_success_rate": 0.75,
        "shape_type": "cylinder",
        "grasp_difficulty": "easy"
    },
    "010_potted_meat_can": {
        "dimensions": [0.102, 0.067, 0.089],
        "mass": 0.370,
        "base_success_rate": 0.78,
        "shape_type": "box",
        "grasp_difficulty": "easy"
    },
    
    # 中等难度物体
    "003_cracker_box": {
        "dimensions": [0.160, 0.213, 0.078],
        "mass": 0.453,
        "base_success_rate": 0.65,
        "shape_type": "box",
        "grasp_difficulty": "medium"
    },
    "004_sugar_box": {
        "dimensions": [0.090, 0.175, 0.044],
        "mass": 0.514,
        "base_success_rate": 0.68,
        "shape_type": "box",
        "grasp_difficulty": "medium"
    },
    "008_pudding_box": {
        "dimensions": [0.109, 0.067, 0.032],
        "mass": 0.187,
        "base_success_rate": 0.70,
        "shape_type": "box",
        "grasp_difficulty": "medium"
    },
    "009_gelatin_box": {
        "dimensions": [0.086, 0.028, 0.073],
        "mass": 0.097,
        "base_success_rate": 0.72,
        "shape_type": "box",
        "grasp_difficulty": "medium"
    },
    "006_mustard_bottle": {
        "dimensions": [0.095, 0.095, 0.196],
        "mass": 0.603,
        "base_success_rate": 0.60,
        "shape_type": "bottle",
        "grasp_difficulty": "medium"
    },
    "021_bleach_cleanser": {
        "dimensions": [0.067, 0.067, 0.258],
        "mass": 1.131,
        "base_success_rate": 0.55,
        "shape_type": "bottle",
        "grasp_difficulty": "medium"
    },
    
    # 困难物体
    "011_banana": {
        "dimensions": [0.028, 0.028, 0.162],
        "mass": 0.066,
        "base_success_rate": 0.45,
        "shape_type": "irregular",
        "grasp_difficulty": "hard"
    },
    "019_pitcher_base": {
        "dimensions": [0.147, 0.147, 0.180],
        "mass": 0.178,
        "base_success_rate": 0.50,
        "shape_type": "complex",
        "grasp_difficulty": "hard"
    },
    "024_bowl": {
        "dimensions": [0.162, 0.162, 0.055],
        "mass": 0.147,
        "base_success_rate": 0.40,
        "shape_type": "bowl",
        "grasp_difficulty": "hard"
    },
    "025_mug": {
        "dimensions": [0.118, 0.118, 0.095],
        "mass": 0.118,
        "base_success_rate": 0.48,
        "shape_type": "mug",
        "grasp_difficulty": "hard"
    },
    "035_power_drill": {
        "dimensions": [0.190, 0.075, 0.190],
        "mass": 0.895,
        "base_success_rate": 0.35,
        "shape_type": "tool",
        "grasp_difficulty": "hard"
    },
    "036_wood_block": {
        "dimensions": [0.085, 0.085, 0.201],
        "mass": 0.729,
        "base_success_rate": 0.75,
        "shape_type": "block",
        "grasp_difficulty": "easy"
    },
    "037_scissors": {
        "dimensions": [0.069, 0.165, 0.021],
        "mass": 0.082,
        "base_success_rate": 0.30,
        "shape_type": "tool",
        "grasp_difficulty": "hard"
    },
    "040_large_marker": {
        "dimensions": [0.021, 0.021, 0.137],
        "mass": 0.016,
        "base_success_rate": 0.55,
        "shape_type": "cylinder",
        "grasp_difficulty": "medium"
    },
    "051_large_clamp": {
        "dimensions": [0.125, 0.125, 0.061],
        "mass": 0.202,
        "base_success_rate": 0.42,
        "shape_type": "tool",
        "grasp_difficulty": "hard"
    },
    "052_extra_large_clamp": {
        "dimensions": [0.200, 0.200, 0.061],
        "mass": 0.125,
        "base_success_rate": 0.38,
        "shape_type": "tool",
        "grasp_difficulty": "hard"
    },
    "061_foam_brick": {
        "dimensions": [0.051, 0.051, 0.051],
        "mass": 0.028,
        "base_success_rate": 0.82,
        "shape_type": "block",
        "grasp_difficulty": "easy"
    },
}

# 环境总配置
ENV_CONFIG = {
    # 场景配置
    "scene_config": {
        "workspace_bounds": {
            "x_min": -0.2,
            "x_max": 0.2,
            "y_min": -0.4,
            "y_max": 0.0,
            "z_min": 0.82,
            "z_max": 1.2
        },
        "table_height": 0.8,
        "max_stack_height": 3,
        "stacking_probability": 0.6,
        "min_objects": 8,
        "max_objects": 16,
    },
    
    # 暴露度计算配置
    "exposure_config": {
        "ray_directions": [
            [0, 0, 1],    # 顶部
            [1, 0, 0],    # 右侧
            [-1, 0, 0],   # 左侧
            [0, 1, 0],    # 前方
            [0, -1, 0],   # 后方
            [0.707, 0.707, 0],    # 对角线
            [-0.707, 0.707, 0],   # 对角线
            [0.707, -0.707, 0],   # 对角线
            [-0.707, -0.707, 0],  # 对角线
        ],
        "ray_length": 0.3,
        "min_clearance": 0.05,
        "height_weight": 0.4,
        "distance_weight": 0.6,
    },
    
    # 智能选择参数
    "selection_params": {
        "category_weight": 0.3,
        "exposure_weight": 0.4,
        "graspability_weight": 0.3,
        "failure_penalty": 0.1,
        "success_bonus": 0.05,
        "distance_threshold": 0.15,
        "height_threshold": 0.1,
    },
    
    # 奖励配置
    "reward_config": {
        "success_reward": 10.0,
        "failure_penalty": -2.0,
        "time_penalty": -0.1,
        "exposure_bonus_factor": 2.0,
        "displacement_penalty_factor": -5.0,
        "invalid_action_penalty": -5.0,
        "efficiency_bonus": 1.0,
    },
    
    # 训练配置
    "training_config": {
        "total_timesteps": 1000000,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "n_steps": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.01,
    },
    
    # 网络配置
    "network_config": {
        "policy_layers": [256, 256, 128],
        "value_layers": [256, 256, 128],
        "activation": "tanh",
        "ortho_init": True,
        "log_std_init": 0.0,
    },
    
    # 评估配置
    "eval_config": {
        "eval_freq": 10000,
        "n_eval_episodes": 10,
        "eval_deterministic": True,
        "render_eval": False,
    },
    
    # 日志配置
    "logging_config": {
        "log_interval": 1000,
        "save_freq": 50000,
        "tensorboard_log": "./logs/",
        "verbose": 1,
    },
}

# PPO特定配置
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": ENV_CONFIG["training_config"]["learning_rate"],
    "n_steps": ENV_CONFIG["training_config"]["n_steps"],
    "batch_size": ENV_CONFIG["training_config"]["batch_size"],
    "n_epochs": ENV_CONFIG["training_config"]["n_epochs"],
    "gamma": ENV_CONFIG["training_config"]["gamma"],
    "gae_lambda": ENV_CONFIG["training_config"]["gae_lambda"],
    "clip_range": ENV_CONFIG["training_config"]["clip_range"],
    "ent_coef": ENV_CONFIG["training_config"]["ent_coef"],
    "vf_coef": ENV_CONFIG["training_config"]["vf_coef"],
    "max_grad_norm": ENV_CONFIG["training_config"]["max_grad_norm"],
    "target_kl": ENV_CONFIG["training_config"]["target_kl"],
    "tensorboard_log": ENV_CONFIG["logging_config"]["tensorboard_log"],
    "verbose": ENV_CONFIG["logging_config"]["verbose"],
}

# 推理配置
INFERENCE_CONFIG = {
    "model_path": "./models/best_model.zip",
    "deterministic": True,
    "render": True,
    "n_episodes": 10,
    "max_steps": 16,
    "save_video": True,
    "video_path": "./videos/",
    "fps": 30,
}

# 物体难度分类
DIFFICULTY_GROUPS = {
    "easy": [
        "002_master_chef_can", "005_tomato_soup_can", "007_tuna_fish_can",
        "010_potted_meat_can", "036_wood_block", "061_foam_brick"
    ],
    "medium": [
        "003_cracker_box", "004_sugar_box", "008_pudding_box", "009_gelatin_box",
        "006_mustard_bottle", "021_bleach_cleanser", "040_large_marker"
    ],
    "hard": [
        "011_banana", "019_pitcher_base", "024_bowl", "025_mug",
        "035_power_drill", "037_scissors", "051_large_clamp", "052_extra_large_clamp"
    ]
}

# 课程学习配置
CURRICULUM_CONFIG = {
    "enable": True,
    "stages": [
        {
            "name": "stage_1_easy",
            "timesteps": 200000,
            "difficulty_groups": ["easy"],
            "max_objects": 8,
            "success_threshold": 0.7,
        },
        {
            "name": "stage_2_mixed",
            "timesteps": 300000,
            "difficulty_groups": ["easy", "medium"],
            "max_objects": 12,
            "success_threshold": 0.6,
        },
        {
            "name": "stage_3_all",
            "timesteps": 500000,
            "difficulty_groups": ["easy", "medium", "hard"],
            "max_objects": 16,
            "success_threshold": 0.5,
        }
    ]
}

def get_objects_by_difficulty(difficulty: str):
    """根据难度获取物体列表"""
    return DIFFICULTY_GROUPS.get(difficulty, [])

def get_object_properties(category: str):
    """获取物体属性"""
    return OBJECT_PROPERTIES.get(category, {})

def get_env_config():
    """获取环境配置"""
    return ENV_CONFIG

def get_training_config():
    """获取训练配置"""
    return ENV_CONFIG["training_config"]

def get_ppo_config():
    """获取PPO配置"""
    return PPO_CONFIG

def get_inference_config():
    """获取推理配置"""
    return INFERENCE_CONFIG

def get_curriculum_config():
    """获取课程学习配置"""
    return CURRICULUM_CONFIG

if __name__ == "__main__":
    # 配置测试
    print("=== 环境配置测试 ===")
    print(f"物体类别数量: {len(CATEGORY_MAP)}")
    print(f"物体属性数量: {len(OBJECT_PROPERTIES)}")
    print(f"简单物体: {len(get_objects_by_difficulty('easy'))}")
    print(f"中等物体: {len(get_objects_by_difficulty('medium'))}")
    print(f"困难物体: {len(get_objects_by_difficulty('hard'))}")
    
    print("\n=== 训练配置 ===")
    print(f"总训练步数: {ENV_CONFIG['training_config']['total_timesteps']}")
    print(f"学习率: {ENV_CONFIG['training_config']['learning_rate']}")
    print(f"批大小: {ENV_CONFIG['training_config']['batch_size']}")
    
    print("\n=== 奖励配置 ===")
    print(f"成功奖励: {ENV_CONFIG['reward_config']['success_reward']}")
    print(f"失败惩罚: {ENV_CONFIG['reward_config']['failure_penalty']}")
    print(f"时间惩罚: {ENV_CONFIG['reward_config']['time_penalty']}") 