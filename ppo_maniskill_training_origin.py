import os
import random
import time
import argparse
import csv
import json
from collections import deque

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
from stable_baselines3.common.logger import configure
import matplotlib.pyplot as plt
import cv2

# 导入ManiSkill相关模块
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode

# 导入自定义环境
from stack_picking_maniskill_env import StackPickingManiSkillEnv

class MultiEnvVisualizationCallback(BaseCallback):
    """多环境可视化回调 - 在SSH环境中保存可视化图像"""
    
    def __init__(self, render_freq=10, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.step_count = 0
        self.save_dir = "visualization_images"
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
    def _on_step(self):
        self.step_count += 1
        
        # 定期渲染所有环境
        if self.step_count % self.render_freq == 0:
            try:
                # 获取所有环境的渲染图像
                images = []
                for i, env in enumerate(self.training_env.envs):
                    try:
                        # 渲染环境
                        img = env.render()
                        if img is not None:
                            # 转换为RGB格式
                            if len(img.shape) == 3 and img.shape[2] == 3:
                                images.append(img)
                            else:
                                # 如果是灰度图，转换为RGB
                                if len(img.shape) == 2:
                                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                                    images.append(img_rgb)
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"环境 {i} 渲染失败: {e}")
                        continue
                
                # 如果有图像，组合并保存
                if images:
                    self._save_multi_env_image(images)
                    if self.verbose > 0:
                        print(f"步骤 {self.step_count}: 已保存多环境可视化图像")
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"多环境可视化失败: {e}")
        
        return True
    
    def _save_multi_env_image(self, images):
        """保存多环境组合图像到文件"""
        try:
            num_envs = len(images)
            if num_envs == 0:
                return
            
            # 计算网格布局
            cols = int(np.ceil(np.sqrt(num_envs)))
            rows = int(np.ceil(num_envs / cols))
            
            # 获取单个图像的尺寸
            h, w = images[0].shape[:2]
            
            # 创建组合图像
            combined_img = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
            
            # 填充图像
            for i, img in enumerate(images):
                row = i // cols
                col = i % cols
                
                # 确保图像尺寸一致
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h))
                
                # 放置图像
                combined_img[row*h:(row+1)*h, col*w:(col+1)*w] = img
                
                # 添加环境编号标签
                cv2.putText(combined_img, f'Env {i}', 
                           (col*w + 10, row*h + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 保存组合图像
            filename = f"{self.save_dir}/multi_env_step_{self.step_count:06d}.png"
            cv2.imwrite(filename, combined_img)
            
            # 保持最新的10张图像，删除旧的
            import glob
            all_images = sorted(glob.glob(f"{self.save_dir}/multi_env_step_*.png"))
            if len(all_images) > 10:
                for old_img in all_images[:-10]:
                    try:
                        os.remove(old_img)
                    except:
                        pass
            
        except Exception as e:
            if self.verbose > 0:
                print(f"保存组合图像失败: {e}")

class PPOTrainingConfig:
    """PPO训练配置类 - 参考PyBullet项目的配置结构"""
    
    def __init__(self):
        # 基础训练参数
        self.total_timesteps = 1000000  # 总训练步数
        self.num_envs = 4  # 并行环境数量
        self.n_steps = 512  # 每个环境的步数
        self.batch_size = 128  # 批次大小
        self.learning_rate = 3e-4  # 学习率
        self.gamma = 0.99  # 折扣因子
        self.n_epochs = 10  # 每次更新的训练轮数
        self.clip_range = 0.2  # PPO裁剪范围
        self.ent_coef = 0.01  # 熵系数
        self.vf_coef = 0.5  # 价值函数系数
        self.max_grad_norm = 0.5  # 梯度裁剪
        self.gae_lambda = 0.95  # GAE lambda
        
        # 可视化设置
        self.enable_render = True  # 启用可视化
        self.render_freq = 50  # 可视化频率
        
        # 环境相关参数
        self.max_episode_steps = 200
        self.env_reward_config = {
            "success_reward": 10.0,
            "reach_reward": 1.0,
            "grasp_reward": 2.0,
            "place_reward": 5.0,
        }
        
        # 策略网络参数
        self.policy_kwargs = {
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            "activation_fn": torch.nn.ReLU,
        }

class RewardMetricsCallback(BaseCallback):
    """奖励指标回调 - 参考PyBullet项目的回调"""
    
    def __init__(self, verbose=0, flush_freq=100, record_freq=50):
        super(RewardMetricsCallback, self).__init__(verbose)
        self.flush_freq = flush_freq
        self.record_freq = record_freq
        self.step_count = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.success_counts = []
        self.success_rates = []
        
    def _on_step(self):
        self.step_count += 1
        
        # 记录每步的基本信息
        if len(self.locals.get('rewards', [])) > 0:
            current_reward = self.locals['rewards'][0]
            self.logger.record('train/step_reward', current_reward)
        
        # 处理环境信息
        if self.locals.get('infos'):
            for i, info in enumerate(self.locals['infos']):
                if info:
                    # 记录成功相关数据
                    if 'success_count' in info:
                        self.logger.record('success/cumulative_successes', info['success_count'])
                    
                    if 'success_rate' in info:
                        self.logger.record('success/current_success_rate', info['success_rate'])
                    
                    if 'remaining_objects' in info:
                        self.logger.record('environment/objects_remaining', info['remaining_objects'])
                    
                    if 'episode_reward' in info:
                        self.logger.record('episode/total_reward', info['episode_reward'])
                    
                    if 'total_distance' in info:
                        self.logger.record('environment/total_distance', info['total_distance'])
                    
                    # 只处理第一个环境的数据
                    break
        
        # 检查回合结束
        dones = self.locals.get('dones')
        if dones is not None and np.any(dones):
            self.episode_count += 1
            self.logger.record('training/episodes_completed', self.episode_count)
            
            # 记录回合统计
            if self.locals.get('infos'):
                for info in self.locals['infos']:
                    if info and info.get('is_episode_done', False):
                        self.episode_rewards.append(info.get('episode_reward', 0))
                        self.success_counts.append(info.get('success_count', 0))
                        self.success_rates.append(info.get('success_rate', 0))
                        break
        
        # 定期刷新日志
        if self.step_count % self.flush_freq == 0:
            # 计算统计信息
            if self.episode_rewards:
                recent_rewards = self.episode_rewards[-10:]
                recent_successes = self.success_counts[-10:]
                recent_rates = self.success_rates[-10:]
                
                self.logger.record('episode/mean_reward_10', np.mean(recent_rewards))
                self.logger.record('episode/mean_success_count_10', np.mean(recent_successes))
                self.logger.record('episode/mean_success_rate_10', np.mean(recent_rates))
            
            self.logger.dump(self.step_count)
        
        return True

def make_env(env_id, rank, seed=0, enable_render=False):
    """创建环境的工厂函数 - 修复渲染模式一致性问题"""
    def _init():
        # 所有环境都使用相同的渲染模式，避免stable-baselines3的错误
        # 训练时统一使用rgb_array模式，不使用human模式避免冲突
        render_mode = "rgb_array"
        print("rank: ", rank)
        # 创建ManiSkill环境
        env = gym.make(
            "StackPickingManiSkill-v1",
            obs_mode="state",  # 使用state模式，避免RGBD问题
            control_mode="pd_joint_delta_pos",
            render_mode=render_mode,  # 统一使用rgb_array模式
            max_objects=3,  # 减少物体数量避免复杂性
            robot_uids="panda",  # 明确指定机器人类型
        )
        
        # 不使用任何包装器，直接使用原始环境
        
        # 设置种子 - 使用新的gymnasium标准方式
        try:
            # 新的gymnasium方式：通过reset传递种子
            env.reset(seed=seed + rank)
        except Exception:
            # 如果环境不支持新方式，尝试旧方式但避免警告
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'seed'):
                env.unwrapped.seed(seed + rank)
            elif hasattr(env, 'seed'):
                # 这里会产生弃用警告，但作为后备方案
                env.seed(seed + rank)
        
        # 为动作空间和观测空间设置种子（这些通常不会产生弃用警告）
        if hasattr(env.action_space, 'seed'):
            env.action_space.seed(seed + rank)
        if hasattr(env.observation_space, 'seed'):
            env.observation_space.seed(seed + rank)
        
        return env
    
    set_random_seed(seed)
    return _init

def save_hyperparameters(params, run_number):
    """保存超参数 - 参考PyBullet项目的保存逻辑"""
    # 确保目录存在
    hyperparams_dir = os.path.join('log', 'hyperparameters')
    os.makedirs(hyperparams_dir, exist_ok=True)
    
    # 创建文件名
    filename = os.path.join(hyperparams_dir, f'maniskill_hyperparameters_{run_number}.json')
    
    # 保存为JSON文件
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"超参数已保存到: {filename}")
    
    # 同时保存为CSV便于查看
    csv_filename = os.path.join(hyperparams_dir, f'maniskill_hyperparameters_{run_number}.csv')
    with open(csv_filename, 'w') as f:
        f.write("参数,值\n")
        # 处理普通参数
        for key, value in params.items():
            if isinstance(value, dict):
                continue
            f.write(f"{key},{value}\n")
        
        # 处理嵌套的奖励参数
        for key, value in params.get("env_reward_config", {}).items():
            f.write(f"reward_{key},{value}\n")
    
    print(f"超参数CSV版本已保存到: {csv_filename}")

def save_results_to_csv(results, mode, cycle_num, run_number):
    """保存结果到CSV文件 - 参考PyBullet项目的结果保存"""
    # 确保有结果可以保存
    if not any(results.values()):
        print("没有结果可保存，创建默认结果")
        results = {
            "objnumlist": [16],
            "sucktimelist": [0],
            "successobjlist": [0],
            "failobjlist": [0],
            "remainobjlist": [16],
            "successratelist": [0],
            "totaldistancelist": [0],
            "averagedistancelist": [0],
            "timelist": [time.time()]
        }
    
    # 调整循环次数
    actual_cycles = min(cycle_num, len(results["objnumlist"]))
    
    # 确保log目录存在
    os.makedirs('log', exist_ok=True)
    
    # 保存到log文件夹
    output_file = f'log/maniskill_evaluation_results_{run_number}.csv'
    
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # 写入表头
        header = ['环境编号', '物体个数', '总抓取次数', '成功抓取物体个数', '失败物体个数', 
                 '残留物体个数', '抓取成功率', '物体位移总距离', '物体位移平均距离', '抓取时间']
        writer.writerow(header)
        
        # 写入数据
        for i in range(actual_cycles):
            row = [i+1, results["objnumlist"][i], results["sucktimelist"][i], 
                  results["successobjlist"][i], results["failobjlist"][i],
                  results["remainobjlist"][i], results["successratelist"][i], 
                  results["totaldistancelist"][i], results["averagedistancelist"][i], 
                  results["timelist"][i]]
            writer.writerow(row)
    
    print(f"结果已保存到 {output_file} 文件")

def train_ppo_model(config):
    """训练PPO模型 - 参考PyBullet项目的训练逻辑"""
    print("开始PPO模型训练...")
    
    # 创建目录结构
    main_dir = "maniskill_ppo_model"
    os.makedirs(main_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(main_dir, "Checkpoint")
    tensorboard_dir = os.path.join(main_dir, "tensorboard")
    trained_model_dir = os.path.join(main_dir, "trained_model")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(trained_model_dir, exist_ok=True)
    os.makedirs("log", exist_ok=True)
    os.makedirs(os.path.join("log", "hyperparameters"), exist_ok=True)
    
    # 确定训练序号
    existing_models = [f for f in os.listdir(trained_model_dir) 
                      if f.startswith("maniskill_model_") and f.endswith(".zip")]
    
    if existing_models:
        model_nums = [int(f.split("_")[-1].split(".")[0]) for f in existing_models]
        current_run = max(model_nums) + 1
    else:
        current_run = 1
    
    # 创建本次训练的目录
    run_name = f"run_{current_run}"
    current_checkpoint_dir = os.path.join(checkpoint_dir, f"maniskill_ppo_model_{current_run}")
    os.makedirs(current_checkpoint_dir, exist_ok=True)
    
    tb_log_dir = os.path.join(tensorboard_dir, run_name)
    os.makedirs(tb_log_dir, exist_ok=True)
    
    model_path = os.path.join(trained_model_dir, f"maniskill_model_{current_run}")
    
    print(f"训练序号: {current_run}")
    print(f"TensorBoard日志目录: {tb_log_dir}")
    print(f"模型保存路径: {model_path}")
    
    # 保存超参数
    hyperparams = {
        "total_timesteps": config.total_timesteps,
        "num_envs": config.num_envs,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "gamma": config.gamma,
        "n_epochs": config.n_epochs,
        "clip_range": config.clip_range,
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
        "max_grad_norm": config.max_grad_norm,
        "gae_lambda": config.gae_lambda,
        "env_reward_config": config.env_reward_config,
        "max_episode_steps": config.max_episode_steps,
    }
    save_hyperparameters(hyperparams, current_run)
    
    try:
        # 创建向量化环境 - 添加可视化支持
        env_fns = [make_env(f"StackPickingManiSkill-v1", i, seed=42, enable_render=config.enable_render) 
                   for i in range(config.num_envs)]
        vec_env = DummyVecEnv(env_fns)
        
        print(f"创建了 {config.num_envs} 个并行环境")
        if config.enable_render:
            print("已启用实时可视化 - 第一个环境将显示渲染窗口")
        
        # 创建PPO模型
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=tb_log_dir,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            n_epochs=config.n_epochs,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            gae_lambda=config.gae_lambda,
            policy_kwargs=config.policy_kwargs,
        )
        
        # 设置检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=2000,
            save_path=current_checkpoint_dir,
            name_prefix="maniskill_ppo_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        
        # 创建回调列表
        callbacks = [checkpoint_callback]
        
        # 如果启用可视化，添加多环境可视化回调
        if config.enable_render:
            multi_env_viz_callback = MultiEnvVisualizationCallback(
                render_freq=config.render_freq, 
                verbose=1
            )
            callbacks.append(multi_env_viz_callback)
            print(f"已启用多环境可视化 - 将在一个窗口中显示所有 {config.num_envs} 个环境")
        
        # 创建指标回调
        metrics_callback = RewardMetricsCallback(verbose=1, flush_freq=100)
        callbacks.append(metrics_callback)
        
        # 进度回调
        class ProgressCallback(BaseCallback):
            def __init__(self, verbose=0):
                super(ProgressCallback, self).__init__(verbose)
                self.step_count = 0
            
            def _on_step(self):
                self.step_count += 1
                if self.step_count % 500 == 0:
                    print(f"已完成 {self.step_count} 步训练 (目标: {config.total_timesteps})")
                return True
        
        progress_callback = ProgressCallback()
        callbacks.append(progress_callback)
        
        # 模型保存回调
        class SaveModelCallback(BaseCallback):
            def __init__(self, save_path, save_freq=1000):
                super(SaveModelCallback, self).__init__()
                self.save_path = save_path
                self.step_count = 0
                self.save_freq = save_freq
            
            def _on_step(self):
                self.step_count += 1
                
                if self.step_count % self.save_freq == 0:
                    save_path = f"{self.save_path}_step_{self.step_count}"
                    self.model.save(save_path)
                    print(f"中间模型已保存到: {save_path}")
                
                return True
        
        save_callback = SaveModelCallback(model_path, save_freq=1000)
        callbacks.append(save_callback)
        
        # 合并回调
        from stable_baselines3.common.callbacks import CallbackList
        callbacks = CallbackList(callbacks)
        
        print(f"\n{'='*50}")
        print(f"开始训练，总步数: {config.total_timesteps}，并行环境: {config.num_envs}")
        print(f"TensorBoard日志保存到: {tb_log_dir}")
        print(f"{'='*50}\n")
        
        # 开始训练
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        print(f"\n{'='*50}")
        print(f"训练完成! 最终步数: {model.num_timesteps}")
        print(f"{'='*50}\n")
        
        # 保存最终模型
        model.save(model_path)
        print(f"最终模型已保存到: {model_path}")
        
        # 关闭环境
        vec_env.close()
        
        return model, current_run
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, current_run

def evaluate_model(model_path, num_episodes=10):
    """评估训练好的模型"""
    print(f"开始评估模型: {model_path}")
    
    # 创建评估环境 - 与训练环境保持一致
    eval_env = gym.make(
        "StackPickingManiSkill-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        max_objects=3,
        robot_uids="panda",
    )
    
    # 不使用任何包装器，与训练时保持一致
    
    # 加载模型
    model = PPO.load(model_path)
    print("模型加载成功")
    
    # 评估结果存储
    episode_rewards = []
    episode_lengths = []
    success_counts = []
    
    for episode in range(num_episodes):
        print(f"评估第 {episode + 1}/{num_episodes} 个episode")
        
        obs, _ = eval_env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(action)
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        success_counts.append(info.get('success_count', 0))
        
        print(f"Episode {episode + 1}: 奖励={total_reward:.3f}, 步数={steps}, 成功数={info.get('success_count', 0)}")
    
    # 计算统计信息
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_success = np.mean(success_counts)
    
    results = {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_episode_length': avg_length,
        'avg_success_count': avg_success,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_counts': success_counts
    }
    
    print(f"\n评估结果:")
    print(f"平均奖励: {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"平均步数: {avg_length:.1f}")
    print(f"平均成功数: {avg_success:.1f}")
    
    eval_env.close()
    return results

def main():
    """主函数 - 参考PyBullet项目的主函数结构"""
    parser = argparse.ArgumentParser(description='ManiSkill PPO训练')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'eval', 'visualize'], 
                       help='运行模式: train(训练), eval(评估), visualize(可视化训练)')
    parser.add_argument('--model_path', type=str, 
                       help='模型路径(用于评估模式)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='评估回合数')
    parser.add_argument('--no_render', action='store_true',
                       help='禁用可视化渲染')
    
    # 添加训练配置参数
    parser.add_argument('--num_envs', type=int, default=256,
                       help='并行环境数量')
    parser.add_argument('--total_timesteps', type=int, default=5000000,
                       help='总训练步数')
    parser.add_argument('--enable_render', action='store_true',
                       help='启用可视化渲染')
    parser.add_argument('--render_freq', type=int, default=100,
                       help='渲染频率（每N步渲染一次）')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--n_steps', type=int, default=2048,
                       help='每个环境的步数')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        config = PPOTrainingConfig()
        # 根据命令行参数更新配置
        config.num_envs = args.num_envs
        config.total_timesteps = args.total_timesteps
        config.learning_rate = args.learning_rate
        config.batch_size = args.batch_size
        config.n_steps = args.n_steps
        config.render_freq = args.render_freq
        
        # 设置可视化
        if args.enable_render:
            config.enable_render = True
        elif args.no_render:
            config.enable_render = False
        
        train_ppo_model(config)
        
    elif args.mode == 'visualize':
        # 可视化训练模式 - 强制启用渲染
        config = PPOTrainingConfig()
        config.enable_render = True
        config.render_freq = 50  # 更频繁的渲染
        config.num_envs = 1  # 使用单个环境以获得更好的可视化效果
        print("启动可视化训练模式...")
        print("注意: 使用单个环境进行可视化训练，训练速度会较慢")
        train_ppo_model(config)
        
    elif args.mode == 'eval':
        if not args.model_path:
            print("评估模式需要指定模型路径，请使用 --model_path 参数")
            return
        evaluate_model(args.model_path, args.episodes)
    
    else:
        print("未知模式，请使用 train, eval 或 visualize")

if __name__ == "__main__":
    main() 