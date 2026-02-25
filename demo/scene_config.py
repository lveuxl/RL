"""
论文展示场景配置文件
可以轻松修改场景布局、物体类型和视觉参数
"""

import numpy as np

class SceneConfig:
    """场景配置类，用于管理物体布局和视觉参数"""
    
    # === YCB物体数据库 ===（15个真实YCB物体，全长方体版本）
    YCB_OBJECTS = {
        # 大型长方体 - 适合底层支撑
        "003_cracker_box": {
            "size": [0.16, 0.21, 0.07],
            "type": "box",
            "stability": "high",
            "description": "饼干盒 - 大长方体"
        },
        "004_sugar_box": {
            "size": [0.09, 0.175, 0.044],
            "type": "box", 
            "stability": "high",
            "description": "糖盒 - 稳定长方体"
        },
        "006_mustard_bottle": {  # 保持原名，当作长方体处理
            "size": [0.095, 0.095, 0.177],
            "type": "box",
            "stability": "medium",
            "description": "芥末瓶 - 高长方体"
        },
        "008_pudding_box": {
            "size": [0.078, 0.109, 0.032],
            "type": "box",
            "stability": "medium",
            "description": "布丁盒 - 中等长方体"
        },
        "009_gelatin_box": {
            "size": [0.028, 0.085, 0.114],
            "type": "box",
            "stability": "low",
            "description": "明胶盒 - 细长方体"
        },
        
        # 将罐头当作长方体处理
        "005_tomato_soup_can": {
            "size": [0.065, 0.065, 0.101],
            "type": "box",  # 当作长方体
            "stability": "medium",
            "description": "番茄汤罐头 - 小圆柱当长方体"
        },
        "007_tuna_fish_can": {
            "size": [0.085, 0.085, 0.032],
            "type": "box",  # 当作长方体
            "stability": "high",
            "description": "金枪鱼罐头 - 扁圆柱当长方体"
        },
        "010_potted_meat_can": {
            "size": [0.101, 0.051, 0.051],
            "type": "box",  # 当作长方体
            "stability": "medium",
            "description": "罐装肉罐头 - 圆柱当长方体"
        },
        
        # 将不规则物体也当作长方体处理
        "011_banana": {
            "size": [0.18, 0.055, 0.055],
            "type": "box",  # 当作长方体
            "stability": "low",
            "description": "香蕉 - 长条当长方体"
        },
        "013_apple": {
            "size": [0.09, 0.09, 0.105],
            "type": "box",  # 当作长方体
            "stability": "low",
            "description": "苹果 - 球形当长方体"
        },
        "014_lemon": {
            "size": [0.055, 0.055, 0.08],
            "type": "box",  # 当作长方体
            "stability": "low",
            "description": "柠檬 - 椭球当长方体"
        },
        "015_peach": {
            "size": [0.07, 0.07, 0.085],
            "type": "box",  # 当作长方体
            "stability": "low",
            "description": "桃子 - 球形当长方体"
        },
        
        # 使用一些其他的真实YCB物体
        "016_pear": {
            "size": [0.06, 0.06, 0.095],
            "type": "box",
            "stability": "low",
            "description": "梨 - 当作长方体"
        },
        "017_orange": {
            "size": [0.075, 0.075, 0.075],
            "type": "box", 
            "stability": "low",
            "description": "橙子 - 当作长方体"
        },
        "018_plum": {
            "size": [0.06, 0.06, 0.065],
            "type": "box",
            "stability": "low", 
            "description": "李子 - 当作长方体"
        },
    }
    
    # === 场景布局配置 ===
    
    # 基础配置：15个真实YCB物体密集堆叠
    BALANCED_STACK_CONFIG = {
        'target_object': '009_gelatin_box',     # 目标物体O_i - 中层位置（细长盒子）
        'support_objects': [                    # 底层支撑物体 - 4个长方体
            '003_cracker_box',      # 大饼干盒作为主要支撑
            '004_sugar_box',        # 糖盒支撑
            '008_pudding_box',      # 布丁盒支撑
            '006_mustard_bottle'    # 芥末瓶支撑
        ],
        'direct_risk': '005_tomato_soup_can',   # 直接风险物体 - 罐头压在目标上
        'indirect_risks': [                     # 间接风险物体 - 上层长方体 (6个)
            '007_tuna_fish_can',    # 金枪鱼罐头
            '010_potted_meat_can',  # 罐装肉罐头
            '011_banana',           # 香蕉
            '013_apple',            # 苹果
            '014_lemon',            # 柠檬
            '015_peach'             # 桃子
        ],
        'neutral_objects': [                    # 中性物体 - 3个分散放置（总计15个）
            '016_pear',             # 梨
            '017_orange',           # 橙子
            '018_plum'              # 李子
        ],
        'description': '密集15个真实YCB物体配置：无托盘桌面堆叠'
    }
    
    # 挑战配置：15个真实YCB物体不稳定堆叠
    CHALLENGING_STACK_CONFIG = {
        'target_object': '011_banana',          # 目标物体 - 长条不稳定物体（中层）
        'support_objects': [                    # 底层支撑物体 - 4个长方体
            '003_cracker_box',      # 大饼干盒支撑
            '004_sugar_box',        # 糖盒支撑
            '006_mustard_bottle',   # 芥末瓶支撑
            '008_pudding_box'       # 布丁盒支撑
        ],
        'direct_risk': '009_gelatin_box',       # 直接风险物体 - 细长盒子
        'indirect_risks': [                     # 间接风险物体 - 上层物体 (6个)
            '005_tomato_soup_can',  # 番茄汤罐头
            '007_tuna_fish_can',    # 金枪鱼罐头
            '010_potted_meat_can',  # 罐装肉罐头
            '013_apple',            # 苹果
            '014_lemon',            # 柠檬
            '015_peach'             # 桃子
        ],
        'neutral_objects': [                    # 中性物体 - 3个长方体（总计15个）
            '016_pear',             # 梨
            '017_orange',           # 橙子
            '018_plum'              # 李子
        ],
        'description': '挑战性15个真实YCB物体：不稳定物体的复杂堆叠'
    }
    
    # 混合配置：15个真实YCB物体场景模拟
    REALISTIC_STACK_CONFIG = {
        'target_object': '006_mustard_bottle',  # 目标物体 - 芥末瓶（中层）
        'support_objects': [                    # 底层支撑物体 - 4个长方体
            '003_cracker_box',      # 大饼干盒支撑
            '004_sugar_box',        # 糖盒支撑
            '008_pudding_box',      # 布丁盒支撑
            '009_gelatin_box'       # 明胶盒支撑
        ],
        'direct_risk': '011_banana',            # 直接风险物体 - 香蕉长条
        'indirect_risks': [                     # 间接风险物体 - 上层物体 (6个)
            '005_tomato_soup_can',  # 番茄汤罐头
            '007_tuna_fish_can',    # 金枪鱼罐头
            '010_potted_meat_can',  # 罐装肉罐头
            '013_apple',            # 苹果
            '014_lemon',            # 柠檬
            '015_peach'             # 桃子
        ],
        'neutral_objects': [                    # 中性物体 - 3个长方体（总计15个）
            '016_pear',             # 梨
            '017_orange',           # 橙子
            '018_plum'              # 李子
        ],
        'description': '真实15个YCB物体场景：芥末瓶目标的复合堆叠'
    }
    
    # === 相机配置 ===
    CAMERA_CONFIGS = {
        'paper_presentation': {
            # 主相机：论文主图 - 高分辨率
            'main_camera': {
                'eye': [0.5, 0.5, 0.6],
                'target': [-0.15, 0.0, 0.15],
                'fov': np.pi / 3,  # 60度
                'resolution': (2560, 1920),  # 4:3比例，超高清适合论文
                'description': '45度俯视角，展示整体堆叠结构（超高清）'
            },
            
            # 侧面相机：展示高度 - 高分辨率
            'side_camera': {
                'eye': [0.7, 0.0, 0.4],
                'target': [-0.2, 0.0, 0.2],
                'fov': np.pi / 3,
                'resolution': (2560, 1920),  # 超高清
                'description': '侧面视角，强调堆叠高度（超高清）'
            },
            
            # 顶部相机：展示布局 - 高分辨率
            'top_camera': {
                'eye': [-0.2, 0.0, 1.0],
                'target': [-0.2, 0.0, 0.05],
                'fov': np.pi / 4,  # 45度
                'resolution': (2048, 2048),  # 超高清正方形，适合顶视图
                'description': '鸟瞰图，展示空间布局（超高清）'
            }
        },
        
        'detailed_analysis': {
            # 近距离特写
            'close_up': {
                'eye': [0.2, 0.3, 0.3],
                'target': [-0.15, 0.0, 0.1],
                'fov': np.pi / 4,
                'resolution': (1920, 1080),  # 高清
                'description': '特写镜头，展示物体细节'
            },
            
            # 低角度视角
            'low_angle': {
                'eye': [0.1, 0.4, 0.15],
                'target': [-0.2, 0.0, 0.25],
                'fov': np.pi / 2.5,
                'resolution': (1280, 720),
                'description': '低角度仰视，增强视觉冲击力'
            }
        }
    }
    
    # === 物理参数配置 ===
    PHYSICS_CONFIG = {
        'stabilization_steps': 100,     # 场景稳定化所需的仿真步数
        'object_separation': 0.005,     # 物体间最小间距（米）
        'stack_stability_margin': 0.02, # 堆叠稳定性边距
        'rotation_variance': 15,        # 物体旋转随机性（度）
        'position_variance': 0.03,      # 位置随机性（米）
    }
    
    # === 托盘和桌面配置 ===
    ENVIRONMENT_CONFIG = {
        'tray_center': [-0.2, 0.0, 0.006],
        'tray_size': [0.6, 0.6, 0.15],
        'tray_spawn_area': [0.23, 0.23],
        'table_height': 0.0,
        'lighting': 'natural',  # 自然光照
        'background': 'neutral', # 中性背景
    }
    
    @classmethod
    def get_scene_config(cls, config_name: str = 'balanced'):
        """
        获取指定的场景配置
        
        Args:
            config_name: 配置名称 ('balanced', 'challenging', 'realistic')
            
        Returns:
            场景配置字典
        """
        config_map = {
            'balanced': cls.BALANCED_STACK_CONFIG,
            'challenging': cls.CHALLENGING_STACK_CONFIG, 
            'realistic': cls.REALISTIC_STACK_CONFIG
        }
        
        if config_name not in config_map:
            print(f"警告: 未找到配置'{config_name}'，使用默认配置'balanced'")
            config_name = 'balanced'
            
        return config_map[config_name]
    
    @classmethod
    def get_camera_config(cls, style: str = 'paper_presentation'):
        """
        获取相机配置
        
        Args:
            style: 相机风格 ('paper_presentation', 'detailed_analysis')
            
        Returns:
            相机配置字典
        """
        return cls.CAMERA_CONFIGS.get(style, cls.CAMERA_CONFIGS['paper_presentation'])
    
    @classmethod
    def print_config_summary(cls, config_name: str = 'balanced'):
        """打印配置摘要信息"""
        config = cls.get_scene_config(config_name)
        
        print(f"\n=== 场景配置摘要: {config_name.upper()} ===")
        print(f"📝 描述: {config['description']}")
        print(f"🏗️  底层支撑物体:")
        for i, obj in enumerate(config['support_objects']):
            print(f"   {i+1}. {obj} - {cls.YCB_OBJECTS[obj]['description']}")
        print(f"🎯 目标物体 (中层): {config['target_object']} - {cls.YCB_OBJECTS[config['target_object']]['description']}")
        print(f"⚠️  直接风险: {config['direct_risk']} - {cls.YCB_OBJECTS[config['direct_risk']]['description']}")
        print("🔺 间接风险 (上层):")
        for i, obj in enumerate(config['indirect_risks']):
            print(f"   {i+1}. {obj} - {cls.YCB_OBJECTS[obj]['description']}")
        print("🌟 中性物体 (不规则形状):")
        for i, obj in enumerate(config['neutral_objects']):
            print(f"   {i+1}. {obj} - {cls.YCB_OBJECTS[obj]['description']}")
        
        # 统计物体类型
        support_count = len(config['support_objects'])
        target_count = 1
        direct_risk_count = 1
        indirect_risk_count = len(config['indirect_risks'])
        neutral_count = len(config['neutral_objects'])
        total_count = support_count + target_count + direct_risk_count + indirect_risk_count + neutral_count
        
        print(f"📊 物体统计: 支撑{support_count} + 目标{target_count} + 直接风险{direct_risk_count} + 间接风险{indirect_risk_count} + 中性{neutral_count} = 总计{total_count}个")
        print(f"📐 形状分布: 长方体/圆柱{support_count + target_count + direct_risk_count + indirect_risk_count}个, 不规则{neutral_count}个")
        
    @classmethod
    def validate_config(cls, config_name: str = 'balanced') -> bool:
        """验证配置的有效性"""
        try:
            config = cls.get_scene_config(config_name)
            
            # 检查所有物体是否在数据库中
            all_objects = config['support_objects'] + [config['target_object'], config['direct_risk']] + \
                         config['indirect_risks'] + config['neutral_objects']
            
            for obj in all_objects:
                if obj not in cls.YCB_OBJECTS:
                    print(f"❌ 错误: 物体 {obj} 不在YCB数据库中")
                    return False
            
            # 检查物体数量（15个长方体）
            total_objects = len(all_objects)
            if total_objects != 15:
                print(f"❌ 错误: 物体总数应为15个，当前为{total_objects}个")
                return False
            
            # 检查是否有重复物体
            if len(set(all_objects)) != len(all_objects):
                print(f"❌ 错误: 配置中存在重复物体")
                return False
            
            # 检查所有物体都是长方体（15个）
            # 支撑物体(4) + 目标物体(1) + 直接风险(1) + 间接风险(6) + 中性物体(3) = 15个长方体
            support_count = len(config['support_objects'])
            indirect_count = len(config['indirect_risks'])  
            neutral_count = len(config['neutral_objects'])
            
            expected_support = 4
            expected_indirect = 6
            expected_neutral = 3
            
            if support_count != expected_support:
                print(f"❌ 错误: 支撑物体应为{expected_support}个，当前为{support_count}个")
                return False
            if indirect_count != expected_indirect:
                print(f"❌ 错误: 间接风险物体应为{expected_indirect}个，当前为{indirect_count}个")
                return False  
            if neutral_count != expected_neutral:
                print(f"❌ 错误: 中性物体应为{expected_neutral}个，当前为{neutral_count}个")
                return False
                
            # 检查所有物体都是长方体类型
            non_box_objects = []
            for obj in all_objects:
                if cls.YCB_OBJECTS[obj]['type'] != 'box':
                    non_box_objects.append(f"{obj}({cls.YCB_OBJECTS[obj]['type']})")
            
            if non_box_objects:
                print(f"❌ 错误: 发现非长方体物体: {non_box_objects}")
                return False
                
            print(f"✅ 配置 '{config_name}' 验证通过 - 15个长方体物体")
            return True
            
        except Exception as e:
            print(f"❌ 配置验证失败: {e}")
            return False


# 使用示例
if __name__ == "__main__":
    # 验证所有配置
    for config_name in ['balanced', 'challenging', 'realistic']:
        SceneConfig.validate_config(config_name)
        SceneConfig.print_config_summary(config_name)
        print("-" * 50)
