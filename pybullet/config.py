# config.py

# 环境相关参数
ENV_CONFIG = {
    "category_map": {
        "CrackerBox": 1,
        "GelatinBox": 2,
        "PottedMeatCan": 3,
        "MustardBottle": 4  
        #"SugarBox": 5,
        #"PuddingBox": 6,
    },
    "obj_num": 16,
    "circle_center": (0.5, -0.2),
    "radius_max": 0.55,
    "radius_min": 0.5,
    "theta_min": 0,  # 弧度
    "theta_max": 51.7,  # 弧度
    "r_min": 0.55,
    "r_max": 0.63
}

# 物理引擎参数
PHYSICS_CONFIG = {
    "gravity": [0, 0, -50],
    "time_step": 1.0/240.0
}

# 机械臂参数
ROBOT_CONFIG = {
    "base_position": [0.5, -0.200000, 0.600000],
    "base_orientation": [0.000000, 0.000000, 0.000000, 1.000000],
    "end_effector_idx": 11,
    "joint_positions": [-1.6137, 1.3258, 1.9346, -0.8884, -1.6172, 1.0867, -3.0494, 0, 0, 0, 0, 0]
}

# 桌子和墙壁参数
TABLE_CONFIG = {
    "length": 1,
    "width": 1.5,
    "wall_height": 1,
    "wall_thickness": 0.02
}