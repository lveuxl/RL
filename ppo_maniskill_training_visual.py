import os
import random
import time
import argparse
import csv
import json
from collections import deque
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

# 导入ManiSkill相关模块
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode

# 导入自定义环境
from stack_picking_maniskill_env import StackPickingManiSkillEnv

class VisualizationConfig:
    """可视化配置类"""
    def __init__(self):
        # 基础可视化设置
        self.enable_realtime_render = True  # 启用实时渲染
        self.render_mode = "human"  # human为窗口显示，rgb_array为图像数组
        self.render_freq = 1  # 每N步渲染一次
        self.save_video = True  # 保存视频
        self.video_fps = 30  # 视频帧率
        
        # 图表可视化设置
        self.enable_live_plots = True  # 启用实时图表
        self.plot_update_freq = 100  # 图表更新频率
        self.plot_window_size = 1000  # 图表显示的数据窗口大小
        
        # 录制设置
        self.record_episodes = True  # 录制episode
        self.record_freq = 10  # 每N个episode录制一次

class PPOTrainingConfig:
    """PPO训练配置类"""
    def __init__(self):
        # 基础训练参数 - 为了可视化，减少并行环境数量
        self.total_timesteps = 50000
        self.num_envs = 1  # 单环境便于可视化
        self.max_episode_steps = 200
        
        # PPO超参数
        self.n_steps = 256
        self.batch_size = 64  # 减小batch size适应单环境
        self.learning_rate = 5e-4
        self.gamma = 0.995
        self.n_epochs = 8
        self.clip_range = 0.15
        self.ent_coef = 0.01
        self.vf_coef = 0.25
        self.max_grad_norm = 0.3
        self.gae_lambda = 0.95
        
        # 网络架构配置
        self.policy_kwargs = {
            "net_arch": [128, 128, 64],
            "activation_fn": torch.nn.Tanh,
        }

class LivePlotter:
    """实时图表绘制类"""
    def __init__(self, config):
        self.config = config
        self.rewards = deque(maxlen=config.plot_window_size)
        self.episode_lengths = deque(maxlen=config.plot_window_size)
        self.success_rates = deque(maxlen=config.plot_window_size)
        self.steps = deque(maxlen=config.plot_window_size)
        
        # 创建图表
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('ManiSkill训练实时监控')
        
        # 设置子图标题
        self.axes[0, 0].set_title('奖励变化')
        self.axes[0, 1].set_title('Episode长度')
        self.axes[1, 0].set_title('成功率')
        self.axes[1, 1].set_title('累计步数')
        
        # 初始化线条
        self.reward_line, = self.axes[0, 0].plot([], [], 'b-')
        self.length_line, = self.axes[0, 1].plot([], [], 'g-')
        self.success_line, = self.axes[1, 0].plot([], [], 'r-')
        self.step_line, = self.axes[1, 1].plot([], [], 'm-')
        
        plt.tight_layout()
        plt.ion()  # 开启交互模式
        plt.show()
    
    def update(self, reward, episode_length, success_rate, total_steps):
        """更新图表数据"""
        self.rewards.append(reward)
        self.episode_lengths.append(episode_length)
        self.success_rates.append(success_rate)
        self.steps.append(total_steps)
        
        # 更新图表
        if len(self.rewards) > 1:
            x_data = list(range(len(self.rewards)))
            
            # 更新奖励图
            self.reward_line.set_data(x_data, list(self.rewards))
            self.axes[0, 0].relim()
            self.axes[0, 0].autoscale_view()
            
            # 更新episode长度图
            self.length_line.set_data(x_data, list(self.episode_lengths))
            self.axes[0, 1].relim()
            self.axes[0, 1].autoscale_view()
            
            # 更新成功率图
            self.success_line.set_data(x_data, list(self.success_rates))
            self.axes[1, 0].relim()
            self.axes[1, 0].autoscale_view()
            
            # 更新步数图
            self.step_line.set_data(x_data, list(self.steps))
            self.axes[1, 1].relim()
            self.axes[1, 1].autoscale_view()
            
            plt.draw()
            plt.pause(0.01)

class VisualizationCallback(BaseCallback):
    """可视化回调类"""
    
    def __init__(self, vis_config, plotter=None, verbose=0):
        super(VisualizationCallback, self).__init__(verbose)
        self.vis_config = vis_config
        self.plotter = plotter
        self.step_count = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        
        # 创建视频保存目录
        if vis_config.save_video:
            self.video_dir = "maniskill_training_videos"
            os.makedirs(self.video_dir, exist_ok=True)
    
    def _on_step(self):
        self.step_count += 1
        
        # 实时渲染
        if (self.vis_config.enable_realtime_render and 
            self.step_count % self.vis_config.render_freq == 0):
            try:
                # 渲染环境
                if hasattr(self.training_env, 'render'):
                    self.training_env.render()
            except Exception as e:
                if self.verbose > 0:
                    print(f"渲染错误: {e}")
        
        # 处理episode结束
        dones = self.locals.get('dones')
        if dones is not None and np.any(dones):
            self.episode_count += 1
            
            # 收集episode统计信息
            if self.locals.get('infos'):
                for info in self.locals['infos']:
                    if info and info.get('is_episode_done', False):
                        episode_reward = info.get('episode_reward', 0)
                        episode_length = info.get('episode_length', 0)
                        success_count = info.get('success_count', 0)
                        success_rate = success_count / max(episode_length, 1)
                        
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
                        self.success_rates.append(success_rate)
                        
                        # 更新实时图表
                        if (self.plotter and 
                            self.episode_count % self.vis_config.plot_update_freq == 0):
                            self.plotter.update(
                                episode_reward, episode_length, 
                                success_rate, self.step_count
                            )
                        
                        # 打印episode信息
                        if self.verbose > 0:
                            print(f"Episode {self.episode_count}: "
                                  f"奖励={episode_reward:.2f}, "
                                  f"步数={episode_length}, "
                                  f"成功率={success_rate:.2f}")
                        
                        break
        
        return True

def make_visual_env(env_id, rank, vis_config, seed=0):
    """创建支持可视化的环境"""
    def _init():
        # 创建ManiSkill环境 - 启用可视化
        env = gym.make(
            "StackPickingManiSkill-v1",
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            render_mode=vis_config.render_mode,  # 使用可视化渲染模式
            max_objects=3,
            robot_uids="panda",
        )
        
        # 如果需要录制视频
        if vis_config.save_video and rank == 0:  # 只在主环境录制
            video_dir = f"maniskill_training_videos/env_{rank}"
            os.makedirs(video_dir, exist_ok=True)
            
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_video=True,
                save_trajectory=False,
                max_steps_per_video=vis_config.max_episode_steps,
                video_fps=vis_config.video_fps,
            )
        
        # 设置种子
        if hasattr(env, 'seed'):
            env.seed(seed + rank)
        if hasattr(env.action_space, 'seed'):
            env.action_space.seed(seed + rank)
        if hasattr(env.observation_space, 'seed'):
            env.observation_space.seed(seed + rank)
        
        return env
    
    set_random_seed(seed)
    return _init

def train_with_visualization(train_config, vis_config):
    """带可视化的训练函数"""
    print("开始可视化训练...")
    
    # 创建目录结构
    main_dir = "maniskill_visual_training"
    os.makedirs(main_dir, exist_ok=True)
    
    tensorboard_dir = os.path.join(main_dir, "tensorboard")
    trained_model_dir = os.path.join(main_dir, "trained_model")
    
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(trained_model_dir, exist_ok=True)
    
    # 创建实时图表绘制器
    plotter = None
    if vis_config.enable_live_plots:
        try:
            plotter = LivePlotter(vis_config)
            print("实时图表已启用")
        except Exception as e:
            print(f"无法启用实时图表: {e}")
    
    try:
        # 创建可视化环境
        env_fns = [make_visual_env(f"StackPickingManiSkill-v1", i, vis_config, seed=42) 
                  for i in range(train_config.num_envs)]
        vec_env = DummyVecEnv(env_fns)
        
        print(f"创建了 {train_config.num_envs} 个可视化环境")
        
        # 创建PPO模型
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=tensorboard_dir,
            n_steps=train_config.n_steps,
            batch_size=train_config.batch_size,
            learning_rate=train_config.learning_rate,
            gamma=train_config.gamma,
            n_epochs=train_config.n_epochs,
            clip_range=train_config.clip_range,
            ent_coef=train_config.ent_coef,
            vf_coef=train_config.vf_coef,
            max_grad_norm=train_config.max_grad_norm,
            gae_lambda=train_config.gae_lambda,
            policy_kwargs=train_config.policy_kwargs,
        )
        
        # 创建可视化回调
        vis_callback = VisualizationCallback(vis_config, plotter, verbose=1)
        
        # 进度回调
        class ProgressCallback(BaseCallback):
            def __init__(self, verbose=0):
                super(ProgressCallback, self).__init__(verbose)
                self.step_count = 0
            
            def _on_step(self):
                self.step_count += 1
                if self.step_count % 500 == 0:
                    print(f"已完成 {self.step_count} 步训练")
                return True
        
        progress_callback = ProgressCallback()
        
        # 合并回调
        from stable_baselines3.common.callbacks import CallbackList
        callbacks = CallbackList([vis_callback, progress_callback])
        
        print(f"\n{'='*50}")
        print(f"开始可视化训练，总步数: {train_config.total_timesteps}")
        print(f"实时渲染: {'启用' if vis_config.enable_realtime_render else '禁用'}")
        print(f"实时图表: {'启用' if vis_config.enable_live_plots else '禁用'}")
        print(f"视频录制: {'启用' if vis_config.save_video else '禁用'}")
        print(f"{'='*50}\n")
        
        # 开始训练
        model.learn(
            total_timesteps=train_config.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        print(f"\n{'='*50}")
        print(f"可视化训练完成!")
        print(f"{'='*50}\n")
        
        # 保存模型
        model_path = os.path.join(trained_model_dir, "visual_model")
        model.save(model_path)
        print(f"模型已保存到: {model_path}")
        
        # 关闭环境
        vec_env.close()
        
        # 保持图表显示
        if plotter:
            print("按Enter键关闭图表...")
            input()
            plt.close()
        
        return model
        
    except Exception as e:
        print(f"可视化训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_visual_environment():
    """演示可视化环境"""
    print("启动环境可视化演示...")
    
    # 创建环境
    env = gym.make(
        "StackPickingManiSkill-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="human",  # 窗口显示
        max_objects=3,
        robot_uids="panda",
    )
    
    print("环境已创建，开始演示...")
    print("提示：关闭渲染窗口可结束演示")
    
    try:
        for episode in range(5):
            print(f"\n开始Episode {episode + 1}")
            obs, _ = env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 100:
                # 随机动作演示
                action = env.action_space.sample()
                obs, reward, done, _, info = env.step(action)
                
                # 渲染环境
                env.render()
                
                step_count += 1
                time.sleep(0.05)  # 控制播放速度
                
                if step_count % 20 == 0:
                    print(f"  步数: {step_count}, 奖励: {reward:.3f}")
            
            print(f"Episode {episode + 1} 完成，步数: {step_count}")
    
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
    finally:
        env.close()
        print("环境已关闭")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ManiSkill可视化训练脚本')
    parser.add_argument('--mode', type=str, 
                       choices=['train', 'demo'], 
                       default='train', 
                       help='选择训练或演示模式')
    parser.add_argument('--enable_render', action='store_true', 
                       help='启用实时渲染')
    parser.add_argument('--enable_plots', action='store_true', 
                       help='启用实时图表')
    parser.add_argument('--save_video', action='store_true', 
                       help='保存训练视频')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(42)
    
    if args.mode == 'demo':
        # 演示模式
        demo_visual_environment()
    
    elif args.mode == 'train':
        # 训练模式
        train_config = PPOTrainingConfig()
        vis_config = VisualizationConfig()
        
        # 根据命令行参数调整配置
        if args.enable_render:
            vis_config.enable_realtime_render = True
        if args.enable_plots:
            vis_config.enable_live_plots = True
        if args.save_video:
            vis_config.save_video = True
        
        model = train_with_visualization(train_config, vis_config)
        
        if model:
            print("训练成功完成！")
        else:
            print("训练失败！")

if __name__ == "__main__":
    main() 