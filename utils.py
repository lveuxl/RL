"""
工具函数文件
包含目录创建、数据保存加载、可视化、评估等功能
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import cv2
import torch
from pathlib import Path


def create_directories():
    """创建必要的目录结构"""
    directories = [
        "./logs/",
        "./logs/eval/",
        "./models/",
        "./models/checkpoints/",
        "./videos/",
        "./plots/",
        "./data/",
        "./results/",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("目录结构创建完成")


def save_training_info(info: Dict, filepath: str):
    """保存训练信息到JSON文件"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False, default=str)
        print(f"训练信息已保存到: {filepath}")
    except Exception as e:
        print(f"保存训练信息时出错: {e}")


def load_training_info(filepath: str) -> Dict:
    """从JSON文件加载训练信息"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            info = json.load(f)
        print(f"训练信息已从 {filepath} 加载")
        return info
    except Exception as e:
        print(f"加载训练信息时出错: {e}")
        return {}


def save_pickle(data: Any, filepath: str):
    """保存数据到pickle文件"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"数据已保存到: {filepath}")
    except Exception as e:
        print(f"保存pickle文件时出错: {e}")


def load_pickle(filepath: str) -> Any:
    """从pickle文件加载数据"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"数据已从 {filepath} 加载")
        return data
    except Exception as e:
        print(f"加载pickle文件时出错: {e}")
        return None


def plot_training_curves(log_dir: str, save_path: str = "./plots/training_curves.png"):
    """绘制训练曲线"""
    try:
        # 读取训练日志
        progress_path = os.path.join(log_dir, "progress.csv")
        if not os.path.exists(progress_path):
            print(f"未找到训练日志文件: {progress_path}")
            return
        
        df = pd.read_csv(progress_path)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("训练过程曲线", fontsize=16)
        
        # 奖励曲线
        if 'rollout/ep_rew_mean' in df.columns:
            axes[0, 0].plot(df['time/total_timesteps'], df['rollout/ep_rew_mean'])
            axes[0, 0].set_title("平均奖励")
            axes[0, 0].set_xlabel("训练步数")
            axes[0, 0].set_ylabel("平均奖励")
            axes[0, 0].grid(True)
        
        # episode长度曲线
        if 'rollout/ep_len_mean' in df.columns:
            axes[0, 1].plot(df['time/total_timesteps'], df['rollout/ep_len_mean'])
            axes[0, 1].set_title("平均Episode长度")
            axes[0, 1].set_xlabel("训练步数")
            axes[0, 1].set_ylabel("平均长度")
            axes[0, 1].grid(True)
        
        # 损失曲线
        if 'train/loss' in df.columns:
            axes[1, 0].plot(df['time/total_timesteps'], df['train/loss'])
            axes[1, 0].set_title("训练损失")
            axes[1, 0].set_xlabel("训练步数")
            axes[1, 0].set_ylabel("损失")
            axes[1, 0].grid(True)
        
        # 学习率曲线
        if 'train/learning_rate' in df.columns:
            axes[1, 1].plot(df['time/total_timesteps'], df['train/learning_rate'])
            axes[1, 1].set_title("学习率")
            axes[1, 1].set_xlabel("训练步数")
            axes[1, 1].set_ylabel("学习率")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存到: {save_path}")
        
    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")


def plot_object_statistics(object_data: List[Dict], save_path: str = "./plots/object_stats.png"):
    """绘制物体统计信息"""
    try:
        # 提取数据
        categories = [obj['category'] for obj in object_data]
        exposures = [obj.get('exposure', 0) for obj in object_data]
        graspabilities = [obj.get('graspability', 0) for obj in object_data]
        success_rates = [obj.get('success_rate', 0) for obj in object_data]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("物体统计信息", fontsize=16)
        
        # 暴露度分布
        axes[0, 0].hist(exposures, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title("暴露度分布")
        axes[0, 0].set_xlabel("暴露度")
        axes[0, 0].set_ylabel("频次")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 可抓取性分布
        axes[0, 1].hist(graspabilities, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title("可抓取性分布")
        axes[0, 1].set_xlabel("可抓取性")
        axes[0, 1].set_ylabel("频次")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 成功率分布
        axes[1, 0].hist(success_rates, bins=20, alpha=0.7, color='salmon', edgecolor='black')
        axes[1, 0].set_title("成功率分布")
        axes[1, 0].set_xlabel("成功率")
        axes[1, 0].set_ylabel("频次")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 类别分布
        category_counts = pd.Series(categories).value_counts()
        axes[1, 1].bar(range(len(category_counts)), category_counts.values, color='orange', alpha=0.7)
        axes[1, 1].set_title("物体类别分布")
        axes[1, 1].set_xlabel("类别")
        axes[1, 1].set_ylabel("数量")
        axes[1, 1].set_xticks(range(len(category_counts)))
        axes[1, 1].set_xticklabels(category_counts.index, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"物体统计图已保存到: {save_path}")
        
    except Exception as e:
        print(f"绘制物体统计图时出错: {e}")


def plot_exposure_heatmap(scene_data: Dict, save_path: str = "./plots/exposure_heatmap.png"):
    """绘制暴露度热力图"""
    try:
        # 提取位置和暴露度数据
        positions = []
        exposures = []
        
        for obj in scene_data.get('objects', []):
            pos = obj.get('position', [0, 0, 0])
            exposure = obj.get('exposure', 0)
            positions.append([pos[0], pos[1]])
            exposures.append(exposure)
        
        if not positions:
            print("没有找到位置数据")
            return
        
        positions = np.array(positions)
        exposures = np.array(exposures)
        
        # 创建网格
        x_min, x_max = positions[:, 0].min() - 0.1, positions[:, 0].max() + 0.1
        y_min, y_max = positions[:, 1].min() - 0.1, positions[:, 1].max() + 0.1
        
        # 创建热力图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(positions[:, 0], positions[:, 1], 
                            c=exposures, cmap='viridis', 
                            s=100, alpha=0.8, edgecolors='black')
        
        plt.colorbar(scatter, label='暴露度')
        plt.title("场景暴露度热力图")
        plt.xlabel("X坐标")
        plt.ylabel("Y坐标")
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (pos, exp) in enumerate(zip(positions, exposures)):
            plt.annotate(f'{exp:.2f}', (pos[0], pos[1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"暴露度热力图已保存到: {save_path}")
        
    except Exception as e:
        print(f"绘制暴露度热力图时出错: {e}")


def calculate_exposure_metrics(object_data: List[Dict]) -> Dict:
    """计算暴露度相关指标"""
    try:
        exposures = [obj.get('exposure', 0) for obj in object_data]
        
        metrics = {
            'mean_exposure': np.mean(exposures),
            'std_exposure': np.std(exposures),
            'min_exposure': np.min(exposures),
            'max_exposure': np.max(exposures),
            'median_exposure': np.median(exposures),
            'q25_exposure': np.percentile(exposures, 25),
            'q75_exposure': np.percentile(exposures, 75),
        }
        
        return metrics
        
    except Exception as e:
        print(f"计算暴露度指标时出错: {e}")
        return {}


def calculate_graspability_metrics(object_data: List[Dict]) -> Dict:
    """计算可抓取性相关指标"""
    try:
        graspabilities = [obj.get('graspability', 0) for obj in object_data]
        
        metrics = {
            'mean_graspability': np.mean(graspabilities),
            'std_graspability': np.std(graspabilities),
            'min_graspability': np.min(graspabilities),
            'max_graspability': np.max(graspabilities),
            'median_graspability': np.median(graspabilities),
            'q25_graspability': np.percentile(graspabilities, 25),
            'q75_graspability': np.percentile(graspabilities, 75),
        }
        
        return metrics
        
    except Exception as e:
        print(f"计算可抓取性指标时出错: {e}")
        return {}


def save_video_frames(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """保存视频帧为MP4文件"""
    try:
        if not frames:
            print("没有帧数据")
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # 确保帧是正确的格式
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        print(f"视频已保存到: {output_path}")
        
    except Exception as e:
        print(f"保存视频时出错: {e}")


def create_evaluation_report(results: Dict, save_path: str = "./results/evaluation_report.json"):
    """创建评估报告"""
    try:
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_episodes": results.get("total_episodes", 0),
                "success_rate": results.get("success_rate", 0.0),
                "average_reward": results.get("average_reward", 0.0),
                "average_steps": results.get("average_steps", 0.0),
            },
            "detailed_results": results,
            "metrics": {
                "exposure_metrics": calculate_exposure_metrics(results.get("object_data", [])),
                "graspability_metrics": calculate_graspability_metrics(results.get("object_data", [])),
            }
        }
        
        save_training_info(report, save_path)
        print(f"评估报告已保存到: {save_path}")
        
        return report
        
    except Exception as e:
        print(f"创建评估报告时出错: {e}")
        return {}


def print_evaluation_summary(report: Dict):
    """打印评估摘要"""
    try:
        print("\n" + "="*50)
        print("           评估结果摘要")
        print("="*50)
        
        summary = report.get("summary", {})
        print(f"总episode数: {summary.get('total_episodes', 0)}")
        print(f"成功率: {summary.get('success_rate', 0.0):.2%}")
        print(f"平均奖励: {summary.get('average_reward', 0.0):.2f}")
        print(f"平均步数: {summary.get('average_steps', 0.0):.1f}")
        
        # 暴露度指标
        exposure_metrics = report.get("metrics", {}).get("exposure_metrics", {})
        if exposure_metrics:
            print(f"\n暴露度指标:")
            print(f"  平均值: {exposure_metrics.get('mean_exposure', 0.0):.3f}")
            print(f"  标准差: {exposure_metrics.get('std_exposure', 0.0):.3f}")
            print(f"  最小值: {exposure_metrics.get('min_exposure', 0.0):.3f}")
            print(f"  最大值: {exposure_metrics.get('max_exposure', 0.0):.3f}")
        
        # 可抓取性指标
        graspability_metrics = report.get("metrics", {}).get("graspability_metrics", {})
        if graspability_metrics:
            print(f"\n可抓取性指标:")
            print(f"  平均值: {graspability_metrics.get('mean_graspability', 0.0):.3f}")
            print(f"  标准差: {graspability_metrics.get('std_graspability', 0.0):.3f}")
            print(f"  最小值: {graspability_metrics.get('min_graspability', 0.0):.3f}")
            print(f"  最大值: {graspability_metrics.get('max_graspability', 0.0):.3f}")
        
        print("="*50)
        
    except Exception as e:
        print(f"打印评估摘要时出错: {e}")


def log_experiment_config(config: Dict, save_path: str = "./logs/experiment_config.json"):
    """记录实验配置"""
    try:
        config_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            "config": config
        }
        
        save_training_info(config_with_timestamp, save_path)
        print(f"实验配置已保存到: {save_path}")
        
    except Exception as e:
        print(f"记录实验配置时出错: {e}")


def compare_models(model_results: Dict[str, Dict], save_path: str = "./plots/model_comparison.png"):
    """比较不同模型的性能"""
    try:
        model_names = list(model_results.keys())
        success_rates = [results.get("success_rate", 0) for results in model_results.values()]
        avg_rewards = [results.get("average_reward", 0) for results in model_results.values()]
        avg_steps = [results.get("average_steps", 0) for results in model_results.values()]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("模型性能比较", fontsize=16)
        
        # 成功率比较
        axes[0].bar(model_names, success_rates, color='skyblue', alpha=0.7)
        axes[0].set_title("成功率比较")
        axes[0].set_ylabel("成功率")
        axes[0].set_ylim(0, 1)
        for i, v in enumerate(success_rates):
            axes[0].text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom')
        
        # 平均奖励比较
        axes[1].bar(model_names, avg_rewards, color='lightgreen', alpha=0.7)
        axes[1].set_title("平均奖励比较")
        axes[1].set_ylabel("平均奖励")
        for i, v in enumerate(avg_rewards):
            axes[1].text(i, v + max(avg_rewards) * 0.01, f'{v:.2f}', ha='center', va='bottom')
        
        # 平均步数比较
        axes[2].bar(model_names, avg_steps, color='salmon', alpha=0.7)
        axes[2].set_title("平均步数比较")
        axes[2].set_ylabel("平均步数")
        for i, v in enumerate(avg_steps):
            axes[2].text(i, v + max(avg_steps) * 0.01, f'{v:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"模型比较图已保存到: {save_path}")
        
    except Exception as e:
        print(f"比较模型时出错: {e}")


def set_matplotlib_chinese():
    """设置matplotlib支持中文"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


# 初始化matplotlib中文支持
set_matplotlib_chinese()


if __name__ == "__main__":
    # 工具函数测试
    print("=== 工具函数测试 ===")
    
    # 创建目录
    create_directories()
    
    # 测试数据保存和加载
    test_data = {
        "test_key": "test_value",
        "numbers": [1, 2, 3, 4, 5],
        "timestamp": datetime.now().isoformat()
    }
    
    save_training_info(test_data, "./logs/test_info.json")
    loaded_data = load_training_info("./logs/test_info.json")
    print(f"加载的数据: {loaded_data}")
    
    print("工具函数测试完成") 