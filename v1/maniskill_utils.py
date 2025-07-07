import os
import csv
import json
import time
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from pathlib import Path

from maniskill_config import ENV_CONFIG, CATEGORY_MAP

def setup_directories():
    """设置必要的目录结构 - 参考PyBullet项目的目录创建"""
    directories = [
        "log",
        "log/hyperparameters",
        "maniskill_ppo_model",
        "maniskill_ppo_model/Checkpoint",
        "maniskill_ppo_model/tensorboard", 
        "maniskill_ppo_model/trained_model",
        "maniskill_outputs",
        "maniskill_outputs/images",
        "maniskill_outputs/annotations",
        "maniskill_outputs/depths",
        "maniskill_outputs/videos"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

def get_next_run_number(model_dir: str = "maniskill_ppo_model/trained_model") -> int:
    """获取下一个运行序号 - 参考PyBullet项目的序号管理"""
    if not os.path.exists(model_dir):
        return 1
    
    existing_models = [f for f in os.listdir(model_dir) 
                      if f.startswith("maniskill_model_") and f.endswith(".zip")]
    
    if not existing_models:
        return 1
    
    model_nums = []
    for f in existing_models:
        try:
            num = int(f.split("_")[-1].split(".")[0])
            model_nums.append(num)
        except:
            continue
    
    return max(model_nums) + 1 if model_nums else 1

def save_training_config(config_dict: Dict, run_number: int):
    """保存训练配置 - 参考PyBullet项目的配置保存"""
    config_dir = "log/hyperparameters"
    os.makedirs(config_dir, exist_ok=True)
    
    # 保存JSON格式
    json_file = os.path.join(config_dir, f"maniskill_config_{run_number}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    
    # 保存CSV格式便于查看
    csv_file = os.path.join(config_dir, f"maniskill_config_{run_number}.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['参数', '值'])
        
        def write_nested_dict(d, prefix=''):
            for key, value in d.items():
                full_key = f"{prefix}{key}" if prefix else key
                if isinstance(value, dict):
                    write_nested_dict(value, f"{full_key}_")
                else:
                    writer.writerow([full_key, value])
        
        write_nested_dict(config_dict)
    
    print(f"配置已保存到: {json_file} 和 {csv_file}")

def calculate_object_exposure_simple(obj_pos: np.ndarray, other_positions: List[np.ndarray]) -> float:
    """简化的物体暴露度计算 - 参考PyBullet项目的暴露度算法"""
    # 高度得分 (归一化到0-1)
    height_score = min(obj_pos[2] / 0.3, 1.0)
    
    # 距离得分 (与其他物体的最小距离)
    if not other_positions:
        distance_score = 1.0
    else:
        min_distance = float('inf')
        for other_pos in other_positions:
            # 只计算x-y平面的距离
            distance = np.linalg.norm(obj_pos[:2] - other_pos[:2])
            min_distance = min(min_distance, distance)
        
        # 归一化距离得分
        distance_score = min(min_distance / 0.1, 1.0)
    
    # 综合暴露度
    exposure = (height_score + distance_score) / 2
    return max(0.0, min(1.0, exposure))

def calculate_object_graspability(obj_category: str, exposure: float, failure_count: int) -> float:
    """计算物体可抓取性 - 参考PyBullet项目的抓取评估"""
    # 获取基础成功率
    base_rates = ENV_CONFIG["object_properties"]
    base_success_rate = base_rates.get(obj_category, {}).get("base_success_rate", 0.5)
    
    # 暴露度奖励
    exposure_bonus = exposure * 0.3
    
    # 失败惩罚
    failure_penalty = failure_count * 0.1
    
    # 计算总体可抓取性
    graspability = base_success_rate + exposure_bonus - failure_penalty
    return max(0.1, min(0.95, graspability))

def select_best_object(object_infos: List[Dict], exposures: Dict[int, float], 
                      failure_counts: Dict[int, int]) -> int:
    """选择最佳抓取物体 - 参考PyBullet项目的物体选择策略"""
    if not object_infos:
        return 0
    
    best_obj_id = 0
    best_score = -1
    
    for obj_info in object_infos:
        obj_id = obj_info['id']
        category = obj_info['category']
        exposure = exposures.get(obj_id, 0.0)
        failure_count = failure_counts.get(obj_id, 0)
        
        # 计算综合评分
        graspability = calculate_object_graspability(category, exposure, failure_count)
        
        # 额外的评分因子
        category_bonus = {
            "GelatinBox": 0.1,      # 小物体更容易抓取
            "CrackerBox": 0.05,
            "PottedMeatCan": 0.0,
            "MustardBottle": -0.05   # 细长物体较难抓取
        }.get(category, 0.0)
        
        total_score = graspability + category_bonus
        
        if total_score > best_score:
            best_score = total_score
            best_obj_id = obj_id
    
    return best_obj_id

def normalize_observation_features(features: np.ndarray) -> np.ndarray:
    """归一化观测特征 - 参考PyBullet项目的特征处理"""
    # 位置特征归一化 (假设工作空间范围)
    workspace_bounds = ENV_CONFIG["scene_config"]["workspace_bounds"]
    
    normalized_features = features.copy()
    
    # 对每个物体的位置特征进行归一化
    for i in range(16):  # 16个物体槽位
        start_idx = i * 10
        if start_idx + 3 < len(features):
            # 归一化x, y, z坐标
            normalized_features[start_idx] = (features[start_idx] - workspace_bounds["x_min"]) / \
                                           (workspace_bounds["x_max"] - workspace_bounds["x_min"])
            normalized_features[start_idx + 1] = (features[start_idx + 1] - workspace_bounds["y_min"]) / \
                                               (workspace_bounds["y_max"] - workspace_bounds["y_min"])
            normalized_features[start_idx + 2] = (features[start_idx + 2] - workspace_bounds["z_min"]) / \
                                               (workspace_bounds["z_max"] - workspace_bounds["z_min"])
            # 距离特征已经在环境中归一化
    
    return normalized_features

def save_episode_data(episode_data: Dict, run_number: int, episode_number: int):
    """保存单个episode的数据 - 参考PyBullet项目的数据保存"""
    data_dir = "maniskill_outputs/episode_data"
    os.makedirs(data_dir, exist_ok=True)
    
    filename = os.path.join(data_dir, f"episode_{run_number}_{episode_number}.json")
    
    # 确保数据可序列化
    serializable_data = {}
    for key, value in episode_data.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            serializable_data[key] = value.tolist()
        else:
            serializable_data[key] = value
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)

def load_training_results(run_number: int) -> Dict:
    """加载训练结果 - 参考PyBullet项目的结果加载"""
    results_file = f"log/maniskill_evaluation_results_{run_number}.csv"
    
    if not os.path.exists(results_file):
        print(f"结果文件不存在: {results_file}")
        return {}
    
    results = {
        "objnumlist": [],
        "sucktimelist": [],
        "successobjlist": [],
        "failobjlist": [],
        "remainobjlist": [],
        "successratelist": [],
        "totaldistancelist": [],
        "averagedistancelist": [],
        "timelist": []
    }
    
    with open(results_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results["objnumlist"].append(int(row['物体个数']))
            results["sucktimelist"].append(int(row['总抓取次数']))
            results["successobjlist"].append(int(row['成功抓取物体个数']))
            results["failobjlist"].append(int(row['失败物体个数']))
            results["remainobjlist"].append(int(row['残留物体个数']))
            results["successratelist"].append(float(row['抓取成功率']))
            results["totaldistancelist"].append(float(row['物体位移总距离']))
            results["averagedistancelist"].append(float(row['物体位移平均距离']))
            results["timelist"].append(float(row['抓取时间']))
    
    return results

def calculate_training_statistics(results: Dict) -> Dict:
    """计算训练统计信息 - 参考PyBullet项目的统计计算"""
    if not results or not results.get("successobjlist"):
        return {}
    
    stats = {
        "total_episodes": len(results["successobjlist"]),
        "mean_success_count": np.mean(results["successobjlist"]),
        "std_success_count": np.std(results["successobjlist"]),
        "mean_success_rate": np.mean(results["successratelist"]),
        "std_success_rate": np.std(results["successratelist"]),
        "mean_remaining_objects": np.mean(results["remainobjlist"]),
        "std_remaining_objects": np.std(results["remainobjlist"]),
        "mean_total_distance": np.mean(results["totaldistancelist"]),
        "std_total_distance": np.std(results["totaldistancelist"]),
        "mean_episode_time": np.mean(results["timelist"]),
        "std_episode_time": np.std(results["timelist"]),
        "max_success_count": max(results["successobjlist"]),
        "min_success_count": min(results["successobjlist"]),
        "success_rate_above_50": sum(1 for rate in results["successratelist"] if rate > 0.5),
        "perfect_episodes": sum(1 for remaining in results["remainobjlist"] if remaining == 0)
    }
    
    return stats

def print_training_summary(stats: Dict, run_number: int):
    """打印训练总结 - 参考PyBullet项目的结果展示"""
    print(f"\n{'='*60}")
    print(f"ManiSkill训练总结 - 运行 #{run_number}")
    print(f"{'='*60}")
    print(f"总episode数: {stats.get('total_episodes', 0)}")
    print(f"平均成功抓取数: {stats.get('mean_success_count', 0):.2f} ± {stats.get('std_success_count', 0):.2f}")
    print(f"平均成功率: {stats.get('mean_success_rate', 0):.2f} ± {stats.get('std_success_rate', 0):.2f}")
    print(f"平均剩余物体数: {stats.get('mean_remaining_objects', 0):.2f} ± {stats.get('std_remaining_objects', 0):.2f}")
    print(f"平均位移距离: {stats.get('mean_total_distance', 0):.2f} ± {stats.get('std_total_distance', 0):.2f}")
    print(f"平均episode时间: {stats.get('mean_episode_time', 0):.2f} ± {stats.get('std_episode_time', 0):.2f}秒")
    print(f"最高成功抓取数: {stats.get('max_success_count', 0)}")
    print(f"最低成功抓取数: {stats.get('min_success_count', 0)}")
    print(f"成功率>50%的episodes: {stats.get('success_rate_above_50', 0)}")
    print(f"完美episodes (剩余物体=0): {stats.get('perfect_episodes', 0)}")
    print(f"{'='*60}\n")

def create_camera_data_structure():
    """创建相机数据结构 - 参考PyBullet项目的数据格式"""
    return {
        "categories": [
            {"id": 1, "name": "CrackerBox"},
            {"id": 2, "name": "GelatinBox"},
            {"id": 3, "name": "PottedMeatCan"},
            {"id": 4, "name": "MustardBottle"}
        ],
        "images": [],
        "annotations": []
    }

def generate_random_scene_config(seed: int = None) -> Dict:
    """生成随机场景配置 - 参考PyBullet项目的随机化"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 随机物体数量 (8-16)
    num_objects = random.randint(8, 16)
    
    # 随机物体类型分布
    object_types = list(CATEGORY_MAP.keys())
    scene_objects = []
    
    for i in range(num_objects):
        obj_type = random.choice(object_types)
        scene_objects.append({
            "id": i,
            "type": obj_type,
            "category_id": CATEGORY_MAP[obj_type]
        })
    
    # 随机场景参数
    circle_center = [
        1.0 + random.uniform(-0.1, 0.1),  # x轴小幅随机
        -0.2 + random.uniform(-0.1, 0.1)  # y轴小幅随机
    ]
    
    circle_radius = random.uniform(0.1, 0.2)
    
    return {
        "num_objects": num_objects,
        "objects": scene_objects,
        "circle_center": circle_center,
        "circle_radius": circle_radius,
        "seed": seed
    }

def validate_environment_config(config: Dict) -> bool:
    """验证环境配置的有效性"""
    required_keys = [
        "obj_num", "max_objects", "circle_center", "circle_radius",
        "category_map", "object_properties", "reward_config"
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"缺少必需的配置键: {key}")
            return False
    
    # 验证物体数量
    if config["obj_num"] > config["max_objects"]:
        print(f"物体数量 ({config['obj_num']}) 超过最大限制 ({config['max_objects']})")
        return False
    
    # 验证类别映射
    if not config["category_map"]:
        print("类别映射为空")
        return False
    
    # 验证物体属性
    for category in config["category_map"]:
        if category not in config["object_properties"]:
            print(f"缺少物体属性配置: {category}")
            return False
    
    print("环境配置验证通过")
    return True

def setup_logging(run_number: int) -> str:
    """设置日志记录 - 参考PyBullet项目的日志配置"""
    log_dir = f"log/run_{run_number}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件
    log_file = os.path.join(log_dir, f"training_{run_number}.log")
    
    # 记录开始时间
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"ManiSkill训练开始时间: {start_time}\n")
        f.write(f"运行序号: {run_number}\n")
        f.write("="*50 + "\n")
    
    return log_file

def log_training_progress(log_file: str, message: str):
    """记录训练进度"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

def cleanup_old_files(keep_recent: int = 5):
    """清理旧文件，保留最近的几个版本 - 参考PyBullet项目的清理逻辑"""
    directories_to_clean = [
        "log",
        "maniskill_ppo_model/trained_model",
        "maniskill_ppo_model/Checkpoint"
    ]
    
    for directory in directories_to_clean:
        if not os.path.exists(directory):
            continue
        
        # 获取所有文件，按修改时间排序
        files = []
        for f in os.listdir(directory):
            file_path = os.path.join(directory, f)
            if os.path.isfile(file_path):
                files.append((file_path, os.path.getmtime(file_path)))
        
        # 按时间排序，保留最新的文件
        files.sort(key=lambda x: x[1], reverse=True)
        
        # 删除多余的文件
        for file_path, _ in files[keep_recent:]:
            try:
                os.remove(file_path)
                print(f"删除旧文件: {file_path}")
            except Exception as e:
                print(f"删除文件失败 {file_path}: {e}")

# 兼容性函数，保持与PyBullet项目的接口一致
def safe_mean(values):
    """安全计算平均值"""
    return float(sum(values)) / max(len(values), 1) if values else 0.0 