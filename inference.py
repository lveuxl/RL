"""
复杂堆叠杂乱环境推理脚本
用于加载训练好的模型并进行测试评估
包含可视化、视频录制、性能分析等功能
"""

import os
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import get_inference_config, get_env_config
from env_clutter import ComplexStackingClutterEnv
from utils import (
    create_directories, save_video_frames, create_evaluation_report,
    print_evaluation_summary, plot_object_statistics, plot_exposure_heatmap
)


class ComplexStackingInference:
    """复杂堆叠杂乱环境推理器"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.inference_config = get_inference_config()
        self.env_config = get_env_config()
        
        # 创建必要的目录
        create_directories()
        
        # 加载模型
        self.model = self._load_model()
        
        # 创建环境
        self.env = self._create_env()
        
        # 初始化结果存储
        self.results = {
            "episodes": [],
            "total_episodes": 0,
            "success_count": 0,
            "total_reward": 0.0,
            "total_steps": 0,
            "object_data": [],
            "scene_data": [],
        }
    
    def _load_model(self):
        """加载训练好的模型"""
        model_path = self.args.model_path or self.inference_config["model_path"]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"加载模型: {model_path}")
        model = PPO.load(model_path)
        
        return model
    
    def _create_env(self):
        """创建推理环境"""
        env = ComplexStackingClutterEnv(
            max_objects=self.args.max_objects,
            reward_mode=self.args.reward_mode,
            enable_intelligent_selection=self.args.enable_intelligent_selection,
            render_mode="rgb_array" if self.args.render else None,
        )
        
        return env
    
    def run_episode(self, episode_idx: int, render: bool = False, save_video: bool = False) -> Dict:
        """运行单个episode"""
        obs, info = self.env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        frames = []
        
        episode_data = {
            "episode_idx": episode_idx,
            "steps": [],
            "total_reward": 0.0,
            "total_steps": 0,
            "success": False,
            "objects_attempted": [],
            "objects_succeeded": [],
            "final_scene": None,
        }
        
        print(f"\n=== Episode {episode_idx + 1} ===")
        
        while not done:
            # 获取动作
            action, _states = self.model.predict(
                obs, 
                deterministic=self.args.deterministic
            )
            
            # 执行动作
            obs, reward, done, truncated, info = self.env.step(action)
            
            # 记录数据
            episode_reward += reward
            episode_steps += 1
            
            # 记录步骤信息
            step_info = {
                "step": episode_steps,
                "action": int(action),
                "reward": float(reward),
                "success": info.get("success", False),
                "remaining_objects": info.get("remaining_objects", 0),
                "target_category": info.get("target_category", "unknown"),
                "target_exposure": info.get("target_exposure", 0.0),
                "target_graspability": info.get("target_graspability", 0.0),
            }
            episode_data["steps"].append(step_info)
            
            # 记录尝试的物体
            if info.get("target_category"):
                episode_data["objects_attempted"].append(info["target_category"])
                if info.get("success"):
                    episode_data["objects_succeeded"].append(info["target_category"])
            
            # 渲染和录制
            if render or save_video:
                frame = self.env.render()
                if frame is not None:
                    frames.append(frame)
            
            # 打印步骤信息
            if self.args.verbose:
                print(f"  步骤 {episode_steps}: 动作={action}, 奖励={reward:.2f}, "
                      f"成功={'是' if info.get('success') else '否'}, "
                      f"剩余物体={info.get('remaining_objects', 0)}")
            
            # 检查是否结束
            if done or truncated:
                break
        
        # 更新episode数据
        episode_data["total_reward"] = episode_reward
        episode_data["total_steps"] = episode_steps
        episode_data["success"] = info.get("success", False) or len(episode_data["objects_succeeded"]) > 0
        
        # 获取最终场景信息
        if hasattr(self.env, 'current_objects'):
            episode_data["final_scene"] = {
                "remaining_objects": len(self.env.current_objects),
                "objects": [
                    {
                        "category": obj["category"],
                        "position": obj["position"],
                        "exposure": obj.get("exposure", 0.0),
                        "graspability": obj.get("graspability", 0.0),
                    }
                    for obj in self.env.current_objects
                ]
            }
        
        # 保存视频
        if save_video and frames:
            video_path = f"./videos/episode_{episode_idx + 1}.mp4"
            save_video_frames(frames, video_path, fps=self.args.fps)
        
        # 打印episode总结
        print(f"Episode {episode_idx + 1} 完成:")
        print(f"  总奖励: {episode_reward:.2f}")
        print(f"  总步数: {episode_steps}")
        print(f"  成功: {'是' if episode_data['success'] else '否'}")
        print(f"  成功抓取物体数: {len(episode_data['objects_succeeded'])}")
        
        return episode_data
    
    def run_evaluation(self) -> Dict:
        """运行完整评估"""
        print("=== 开始评估 ===")
        print(f"总episode数: {self.args.n_episodes}")
        print(f"确定性推理: {'是' if self.args.deterministic else '否'}")
        print(f"渲染: {'是' if self.args.render else '否'}")
        print(f"保存视频: {'是' if self.args.save_video else '否'}")
        
        start_time = time.time()
        
        # 运行多个episodes
        for episode_idx in range(self.args.n_episodes):
            episode_data = self.run_episode(
                episode_idx,
                render=self.args.render,
                save_video=self.args.save_video
            )
            
            # 更新总体结果
            self.results["episodes"].append(episode_data)
            self.results["total_episodes"] += 1
            self.results["total_reward"] += episode_data["total_reward"]
            self.results["total_steps"] += episode_data["total_steps"]
            
            if episode_data["success"]:
                self.results["success_count"] += 1
            
            # 收集物体数据
            if episode_data["final_scene"]:
                self.results["object_data"].extend(episode_data["final_scene"]["objects"])
                self.results["scene_data"].append(episode_data["final_scene"])
        
        evaluation_time = time.time() - start_time
        
        # 计算总体统计
        self.results.update({
            "evaluation_time": evaluation_time,
            "success_rate": self.results["success_count"] / self.results["total_episodes"],
            "average_reward": self.results["total_reward"] / self.results["total_episodes"],
            "average_steps": self.results["total_steps"] / self.results["total_episodes"],
        })
        
        print(f"\n评估完成，总用时: {evaluation_time:.2f} 秒")
        
        return self.results
    
    def analyze_results(self) -> Dict:
        """分析评估结果"""
        print("\n=== 分析评估结果 ===")
        
        # 创建评估报告
        report = create_evaluation_report(self.results)
        
        # 打印摘要
        print_evaluation_summary(report)
        
        # 生成可视化图表
        if self.results["object_data"]:
            # 物体统计图
            plot_object_statistics(
                self.results["object_data"],
                save_path="./plots/inference_object_stats.png"
            )
            
            # 暴露度热力图
            if self.results["scene_data"]:
                plot_exposure_heatmap(
                    {"objects": self.results["object_data"]},
                    save_path="./plots/inference_exposure_heatmap.png"
                )
        
        # 分析每个episode的表现
        self._analyze_episode_performance()
        
        # 分析物体类别表现
        self._analyze_object_category_performance()
        
        return report
    
    def _analyze_episode_performance(self):
        """分析每个episode的表现"""
        print("\n=== Episode表现分析 ===")
        
        rewards = [ep["total_reward"] for ep in self.results["episodes"]]
        steps = [ep["total_steps"] for ep in self.results["episodes"]]
        successes = [ep["success"] for ep in self.results["episodes"]]
        
        print(f"奖励统计:")
        print(f"  平均值: {np.mean(rewards):.2f}")
        print(f"  标准差: {np.std(rewards):.2f}")
        print(f"  最小值: {np.min(rewards):.2f}")
        print(f"  最大值: {np.max(rewards):.2f}")
        
        print(f"\n步数统计:")
        print(f"  平均值: {np.mean(steps):.1f}")
        print(f"  标准差: {np.std(steps):.1f}")
        print(f"  最小值: {np.min(steps)}")
        print(f"  最大值: {np.max(steps)}")
        
        print(f"\n成功率: {np.mean(successes):.2%}")
    
    def _analyze_object_category_performance(self):
        """分析物体类别表现"""
        print("\n=== 物体类别表现分析 ===")
        
        # 统计每个类别的尝试和成功次数
        category_stats = {}
        
        for episode in self.results["episodes"]:
            for obj_category in episode["objects_attempted"]:
                if obj_category not in category_stats:
                    category_stats[obj_category] = {"attempted": 0, "succeeded": 0}
                category_stats[obj_category]["attempted"] += 1
            
            for obj_category in episode["objects_succeeded"]:
                if obj_category in category_stats:
                    category_stats[obj_category]["succeeded"] += 1
        
        # 计算成功率
        print("各类别成功率:")
        for category, stats in category_stats.items():
            success_rate = stats["succeeded"] / stats["attempted"] if stats["attempted"] > 0 else 0
            print(f"  {category}: {success_rate:.2%} ({stats['succeeded']}/{stats['attempted']})")
    
    def save_results(self, save_path: str = "./results/inference_results.json"):
        """保存推理结果"""
        from utils import save_training_info
        save_training_info(self.results, save_path)
        print(f"推理结果已保存到: {save_path}")
    
    def close(self):
        """关闭环境"""
        self.env.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="复杂堆叠杂乱环境推理")
    
    # 基本参数
    parser.add_argument("--model-path", type=str, default=None,
                        help="模型文件路径")
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="评估episode数量")
    parser.add_argument("--max-steps", type=int, default=16,
                        help="每个episode最大步数")
    parser.add_argument("--deterministic", action="store_true",
                        help="使用确定性推理")
    
    # 环境参数
    parser.add_argument("--max-objects", type=int, default=16,
                        help="最大物体数量")
    parser.add_argument("--reward-mode", type=str, default="dense",
                        choices=["sparse", "dense"],
                        help="奖励模式")
    parser.add_argument("--enable-intelligent-selection", action="store_true",
                        help="启用智能物体选择")
    
    # 可视化参数
    parser.add_argument("--render", action="store_true",
                        help="渲染环境")
    parser.add_argument("--save-video", action="store_true",
                        help="保存视频")
    parser.add_argument("--fps", type=int, default=30,
                        help="视频帧率")
    
    # 其他参数
    parser.add_argument("--verbose", action="store_true",
                        help="详细输出")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # 创建推理器
    inference = ComplexStackingInference(args)
    
    try:
        # 运行评估
        results = inference.run_evaluation()
        
        # 分析结果
        report = inference.analyze_results()
        
        # 保存结果
        inference.save_results()
        
        print("\n推理完成！")
        
    except KeyboardInterrupt:
        print("\n推理被用户中断")
    except Exception as e:
        print(f"\n推理过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭环境
        inference.close()


if __name__ == "__main__":
    main() 