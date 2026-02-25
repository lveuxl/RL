"""
EnvClutter环境配置文件
包含环境参数、训练参数、模型参数等配置
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class EnvConfig:
    """环境配置"""
    # 环境基本参数
    env_name: str = "EnvClutter-v1"
    num_envs: int = 1
    max_episode_steps: int = 200
    
    # 观测和控制模式
    obs_mode: str = "state"  # "state", "rgbd", "rgb"
    control_mode: str = "pd_ee_delta_pose"  # "pd_ee_delta_pose", "pd_joint_delta_pos"
    reward_mode: str = "dense"  # "dense", "sparse"
    
    # 机器人配置
    robot_uids: str = "panda"  # "panda", "fetch"
    robot_init_qpos_noise: float = 0.02
    
    # 物体配置
    box_objects: List[str] = None
    num_objects_per_type: int = 3  # 每种类型的物体数量
    num_object_types: int = 3      # 物体类型数量
    total_objects_per_env: int = 9 # 每个环境的总物体数量 (num_objects_per_type * num_object_types)
    spawn_radius: float = 0.15
    goal_thresh: float = 0.03
    
    # 离散动作相关配置
    max_episode_steps_discrete: int = 9  # 离散动作模式下的最大抓取尝试次数
    
    # 渲染配置
    render_mode: str = None  # None, "rgb_array", "human"
    camera_width: int = 128
    camera_height: int = 128
    
    # 新增：每个环境的最大物体数量
    max_objects_per_env: int = 9
    
    def __post_init__(self):
        if self.box_objects is None:
            self.box_objects = [
                #"002_master_chef_can",  
                "004_sugar_box",            # 糖盒
                "006_mustard_bottle",       # 芥末瓶
                "008_pudding_box",          # 布丁盒
                #"009_gelatin_box",          # 明胶盒
                #"010_potted_meat_can",      # 罐装肉罐头
            ]
        
        # 自动计算相关参数
        self.num_object_types = len(self.box_objects)
        self.total_objects_per_env = self.num_objects_per_type * self.num_object_types

@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练基本参数
    epochs: int = 1000
    steps_per_epoch: int = 2048
    batch_size: int = 64
    
    # PPO参数
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 10
    
    # 网络参数
    hidden_dim: int = 256
    num_layers: int = 3
    activation: str = "relu"
    
    # 训练策略
    grad_clip: float = 0.5
    target_kl: float = 0.01
    
    # 日志和保存
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    
    # 目录配置
    log_dir: str = "./logs/env_clutter"
    model_dir: str = "./models/env_clutter"
    video_dir: str = "./videos/env_clutter"
    
    # 设备配置
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_workers: int = 4
    
    # 早停配置
    early_stopping: bool = True
    patience: int = 100
    min_improvement: float = 0.01

@dataclass
class ModelConfig:
    """模型配置"""
    # 网络架构
    actor_hidden_dims: List[int] = None
    critic_hidden_dims: List[int] = None
    shared_backbone: bool = False
    
    # 激活函数
    activation: str = "relu"
    output_activation: str = "tanh"
    
    # 初始化
    weight_init: str = "orthogonal"
    bias_init: str = "zeros"
    
    # 正则化
    dropout: float = 0.0
    batch_norm: bool = False
    layer_norm: bool = False
    
    # 输出层
    log_std_init: float = 0.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    
    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [256, 256, 256]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [256, 256, 256]

@dataclass
class RewardConfig:
    """奖励配置"""
    # 奖励权重
    reaching_weight: float = 2.0
    grasping_weight: float = 3.0
    placing_weight: float = 2.0
    displacement_weight: float = 1.5
    time_weight: float = 0.01
    static_weight: float = 1.0
    success_weight: float = 10.0
    
    # 奖励形状参数
    reaching_scale: float = 5.0
    placing_scale: float = 5.0
    static_scale: float = 5.0
    
    # 稀疏奖励
    sparse_success_reward: float = 1.0
    sparse_displacement_weight: float = 0.1
    
    # 新增：离散动作选择奖励参数
    grasp_success_reward: float = 1.0
    disp_coeff: float = 1.5
    time_coeff: float = 0.01

@dataclass
class EvaluationConfig:
    """评估配置"""
    # 评估参数
    eval_episodes: int = 100
    eval_deterministic: bool = True
    eval_render: bool = False
    
    # 成功条件
    success_threshold: float = 0.8
    
    # 评估指标
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "success_rate",
                "average_reward", 
                "episode_length",
                "displacement_penalty",
                "grasp_success_rate"
            ]

class Config:
    """主配置类"""
    def __init__(self):
        self.env = EnvConfig()
        self.training = TrainingConfig()
        self.model = ModelConfig()
        self.reward = RewardConfig()
        self.evaluation = EvaluationConfig()
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "env": self.env.__dict__,
            "training": self.training.__dict__,
            "model": self.model.__dict__,
            "reward": self.reward.__dict__,
            "evaluation": self.evaluation.__dict__,
        }
    
    def save(self, filepath: str):
        """保存配置到文件"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str):
        """从文件加载配置"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.update_from_dict(config_dict)
        return config

# 预定义配置
PRESET_CONFIGS = {
    "default": {
        "env": {
            "num_envs": 1,
            "reward_mode": "dense",
        },
        "training": {
            "epochs": 1000,
            "steps_per_epoch": 2048,
        }
    },
    
    "fast_train": {
        "env": {
            "num_envs": 4,
            "max_episode_steps": 100,
        },
        "training": {
            "epochs": 500,
            "steps_per_epoch": 1024,
            "log_interval": 5,
            "save_interval": 50,
        }
    },
    
    "high_quality": {
        "env": {
            "num_envs": 1,
            "max_episode_steps": 300,
        },
        "training": {
            "epochs": 2000,
            "steps_per_epoch": 4096,
            "lr_actor": 1e-4,
            "lr_critic": 1e-4,
        },
        "model": {
            "actor_hidden_dims": [512, 512, 256],
            "critic_hidden_dims": [512, 512, 256],
        }
    },
    
    "sparse_reward": {
        "env": {
            "reward_mode": "sparse",
        },
        "training": {
            "epochs": 1500,
            "entropy_coef": 0.02,
        }
    },
    
    "multi_env": {
        "env": {
            "num_envs": 8,
        },
        "training": {
            "epochs": 500,
            "steps_per_epoch": 512,
            "batch_size": 128,
        }
    },
    
    "large_scene": {
        "env": {
            "num_objects_per_type": 6,  # 每种类型6个物体
            "total_objects_per_env": 18, # 总共18个物体
            "max_episode_steps_discrete": 20,  # 更多抓取尝试次数
        },
        "training": {
            "epochs": 800,
            "steps_per_epoch": 2048,
        }
    },
    
    "small_scene": {
        "env": {
            "num_objects_per_type": 2,  # 每种类型2个物体
            "total_objects_per_env": 6,  # 总共6个物体
            "max_episode_steps_discrete": 8,   # 较少抓取尝试次数
        },
        "training": {
            "epochs": 600,
            "steps_per_epoch": 1024,
        }
    }
}

def get_config(preset: str = "default") -> Config:
    """获取预设配置"""
    config = Config()
    if preset in PRESET_CONFIGS:
        config.update_from_dict(PRESET_CONFIGS[preset])
    return config

def create_directories(config: Config):
    """创建必要的目录"""
    directories = [
        config.training.log_dir,
        config.training.model_dir,
        config.training.video_dir,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# 使用示例
if __name__ == "__main__":
    # 创建默认配置
    config = get_config("default")
    
    # 保存配置
    config.save("config_default.json")
    
    # 加载配置
    loaded_config = Config.load("config_default.json")
    
    # 创建目录
    create_directories(config)
    
    print("配置文件创建完成！")
    print(f"环境: {config.env.env_name}")
    print(f"训练轮数: {config.training.epochs}")
    #print(f"奖励模式: {config.env.reward_mode}") 