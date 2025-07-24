"""
EnvClutter环境推理脚本
用于加载训练好的模型并进行推理演示
"""

import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import time

# 导入自定义模块
from env_clutter import EnvClutterEnv
from training import PPOAgent, flatten_obs
from config import Config, get_config
from utils import (
    setup_seed, 
    VideoRecorder, 
    evaluate_model, 
    save_evaluation_results,
    print_evaluation_summary,
    PerformanceProfiler
)

import mani_skill.envs
import warnings
warnings.filterwarnings("ignore")

class InferenceEngine:
    """推理引擎"""
    def __init__(self, model_path: str, config: Config):
        self.model_path = model_path
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建环境
        self.env = self._create_environment()
        
        # 获取状态和动作维度
        obs, _ = self.env.reset()
        flattened_obs = flatten_obs(obs)
        self.state_dim = flattened_obs.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # 创建并加载智能体
        self.agent = PPOAgent(self.state_dim, self.action_dim)
        self.agent.load(model_path)
        
        print(f"模型已加载: {model_path}")
        print(f"状态维度: {self.state_dim}, 动作维度: {self.action_dim}")
        print(f"设备: {self.device}")
    
    def _create_environment(self):
        """创建环境"""
        return gym.make(
            self.config.env.env_name,
            num_envs=1,  # 推理时只使用单个环境
            obs_mode=self.config.env.obs_mode,
            control_mode=self.config.env.control_mode,
            reward_mode=self.config.env.reward_mode,
            render_mode="rgb_array",
        )
    
    def run_single_episode(self, render: bool = True, record_video: bool = False, 
                          video_path: str = None) -> dict:
        """运行单个episode"""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_success = False
        
        # 视频录制
        video_recorder = None
        if record_video and video_path:
            video_dir = Path(video_path).parent
            video_recorder = VideoRecorder(video_dir)
            video_recorder.start_recording()
        
        # 性能分析
        profiler = PerformanceProfiler()
        
        while True:
            profiler.start_timer("inference")
            
            # 获取动作
            flattened_obs = flatten_obs(obs)
            state = flattened_obs.numpy()
            action, _ = self.agent.get_action(state)
            
            profiler.end_timer("inference")
            profiler.start_timer("env_step")
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            profiler.end_timer("env_step")
            
            episode_reward += reward
            episode_length += 1
            
            if info.get('success', False):
                episode_success = True
            
            # 渲染
            if render or record_video:
                frame = self.env.render()
                if record_video and video_recorder:
                    video_recorder.add_frame(frame)
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # 保存视频
        if record_video and video_recorder:
            filename = Path(video_path).name if video_path else "inference.mp4"
            video_recorder.stop_recording(filename)
        
        # 返回结果
        results = {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'episode_success': episode_success,
            'performance_stats': profiler.get_stats()
        }
        
        return results
    
    def run_evaluation(self, num_episodes: int = 100, record_videos: bool = False,
                      video_dir: str = None) -> dict:
        """运行评估"""
        print(f"开始评估，共{num_episodes}个episode...")
        
        video_recorder = None
        if record_videos and video_dir:
            video_recorder = VideoRecorder(video_dir)
        
        results = evaluate_model(
            self.env, 
            self.agent, 
            num_episodes=num_episodes,
            render=False,
            video_recorder=video_recorder
        )
        
        return results
    
    def interactive_demo(self):
        """交互式演示"""
        print("=== 交互式演示 ===")
        print("按 'q' 退出，按 'r' 重置环境，按 's' 开始/暂停")
        
        obs, _ = self.env.reset()
        paused = False
        
        while True:
            if not paused:
                # 获取动作
                flattened_obs = flatten_obs(obs)
                state = flattened_obs.numpy()
                action, _ = self.agent.get_action(state)
                
                # 执行动作
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # 显示信息
                print(f"\r奖励: {reward:.4f}, 成功: {info.get('success', False)}", end="")
                
                if terminated or truncated:
                    print(f"\nEpisode结束，重置环境...")
                    obs, _ = self.env.reset()
            
            # 渲染
            self.env.render()
            time.sleep(0.1)  # 控制帧率
            
            # 检查键盘输入（简化版本）
            # 实际应用中可能需要更复杂的输入处理
    
    def benchmark_performance(self, num_episodes: int = 10):
        """性能基准测试"""
        print(f"开始性能基准测试，共{num_episodes}个episode...")
        
        total_profiler = PerformanceProfiler()
        episode_results = []
        
        for episode in range(num_episodes):
            total_profiler.start_timer("total_episode")
            
            result = self.run_single_episode(render=False, record_video=False)
            episode_results.append(result)
            
            total_profiler.end_timer("total_episode")
            
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"奖励={result['episode_reward']:.2f}, "
                  f"长度={result['episode_length']}, "
                  f"成功={result['episode_success']}")
        
        # 汇总统计
        total_rewards = [r['episode_reward'] for r in episode_results]
        total_lengths = [r['episode_length'] for r in episode_results]
        total_successes = [r['episode_success'] for r in episode_results]
        
        print("\n=== 性能基准测试结果 ===")
        print(f"平均奖励: {np.mean(total_rewards):.4f} ± {np.std(total_rewards):.4f}")
        print(f"平均长度: {np.mean(total_lengths):.2f} ± {np.std(total_lengths):.2f}")
        print(f"成功率: {np.mean(total_successes):.4f}")
        
        # 性能统计
        total_profiler.print_stats()
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_length': np.mean(total_lengths),
            'success_rate': np.mean(total_successes),
            'performance_stats': total_profiler.get_stats()
        }
    
    def close(self):
        """关闭环境"""
        self.env.close()

def main():
    parser = argparse.ArgumentParser(description='EnvClutter环境推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--config', type=str, default='default', help='配置名称或配置文件路径')
    parser.add_argument('--mode', type=str, default='demo', 
                       choices=['demo', 'eval', 'benchmark', 'interactive'],
                       help='运行模式')
    parser.add_argument('--num_episodes', type=int, default=100, help='评估episode数量')
    parser.add_argument('--record_video', action='store_true', help='是否录制视频')
    parser.add_argument('--video_dir', type=str, default='./videos/inference', help='视频保存目录')
    parser.add_argument('--output_dir', type=str, default='./results', help='结果保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--render', action='store_true', help='是否渲染')
    
    args = parser.parse_args()
    
    # 设置随机种子
    setup_seed(args.seed)
    
    # 加载配置
    if args.config.endswith('.json'):
        config = Config.load(args.config)
    else:
        config = get_config(args.config)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    if args.record_video:
        os.makedirs(args.video_dir, exist_ok=True)
    
    # 创建推理引擎
    inference_engine = InferenceEngine(args.model_path, config)
    
    try:
        if args.mode == 'demo':
            # 演示模式
            print("=== 演示模式 ===")
            result = inference_engine.run_single_episode(
                render=args.render,
                record_video=args.record_video,
                video_path=os.path.join(args.video_dir, "demo.mp4") if args.record_video else None
            )
            
            print(f"\n演示结果:")
            print(f"奖励: {result['episode_reward']:.4f}")
            print(f"长度: {result['episode_length']}")
            print(f"成功: {result['episode_success']}")
            
        elif args.mode == 'eval':
            # 评估模式
            print("=== 评估模式 ===")
            results = inference_engine.run_evaluation(
                num_episodes=args.num_episodes,
                record_videos=args.record_video,
                video_dir=args.video_dir if args.record_video else None
            )
            
            # 打印结果
            print_evaluation_summary(results)
            
            # 保存结果
            result_path = os.path.join(args.output_dir, "evaluation_results.json")
            save_evaluation_results(results, result_path)
            
        elif args.mode == 'benchmark':
            # 基准测试模式
            print("=== 基准测试模式 ===")
            results = inference_engine.benchmark_performance(args.num_episodes)
            
            # 保存结果
            result_path = os.path.join(args.output_dir, "benchmark_results.json")
            with open(result_path, 'w') as f:
                import json
                json.dump(results, f, indent=2)
            print(f"基准测试结果已保存到: {result_path}")
            
        elif args.mode == 'interactive':
            # 交互模式
            print("=== 交互模式 ===")
            inference_engine.interactive_demo()
            
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"运行时错误: {e}")
    finally:
        inference_engine.close()

if __name__ == "__main__":
    main() 