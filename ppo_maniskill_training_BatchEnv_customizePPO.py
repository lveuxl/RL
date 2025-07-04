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
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# 导入自定义环境
from stack_picking_maniskill_env import StackPickingManiSkillEnv

import gymnasium.spaces

class TensorToNumpyWrapper(gym.Wrapper):
    """将ManiSkillVectorEnv的tensor输出转换为numpy，兼容stable-baselines3"""
    
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = env.num_envs
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space
    
    def reset(self, **kwargs):
        obs_info = self.env.reset(**kwargs)
        if isinstance(obs_info, tuple):
            obs, info = obs_info
            return self._convert_obs_to_numpy(obs), info
        else:
            return self._convert_obs_to_numpy(obs_info)
    
    def step(self, actions):
        # 确保动作是tensor格式
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.env.device)
        
        result = self.env.step(actions)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            # 调试信息
            print(f"DEBUG: reward type={type(reward)}, shape={getattr(reward, 'shape', 'no shape')}")
            converted_reward = self._convert_to_numpy(reward)
            print(f"DEBUG: converted reward type={type(converted_reward)}, value={converted_reward}")
            return (
                self._convert_obs_to_numpy(obs),
                converted_reward,
                self._convert_to_numpy(terminated),
                self._convert_to_numpy(truncated),
                info
            )
        else:
            obs, reward, done, info = result
            # 调试信息
            print(f"DEBUG: reward type={type(reward)}, shape={getattr(reward, 'shape', 'no shape')}")
            converted_reward = self._convert_to_numpy(reward)
            print(f"DEBUG: converted reward type={type(converted_reward)}, value={converted_reward}")
            return (
                self._convert_obs_to_numpy(obs),
                converted_reward,
                self._convert_to_numpy(done),
                info
            )
    
    def _convert_obs_to_numpy(self, obs):
        """转换观测数据为numpy格式"""
        if isinstance(obs, dict):
            numpy_obs = {}
            for key, value in obs.items():
                numpy_obs[key] = self._convert_to_numpy(value)
            return numpy_obs
        else:
            return self._convert_to_numpy(obs)
    
    def _convert_to_numpy(self, tensor_data):
        """转换tensor为numpy"""
        if hasattr(tensor_data, 'cpu'):
            numpy_data = tensor_data.cpu().numpy()
            # 如果是标量tensor，返回Python标量
            if numpy_data.shape == () or (numpy_data.ndim == 1 and len(numpy_data) == 1):
                return float(numpy_data) if numpy_data.dtype.kind == 'f' else int(numpy_data)
            return numpy_data
        elif isinstance(tensor_data, (list, tuple)):
            arr = np.array(tensor_data)
            # 如果是标量数组，返回Python标量
            if arr.shape == () or (arr.ndim == 1 and len(arr) == 1):
                return float(arr) if arr.dtype.kind == 'f' else int(arr)
            return arr
        else:
            return tensor_data
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
    
    def close(self):
        return self.env.close()
    
    def __getattr__(self, name):
        """代理其他属性到原始环境"""
        return getattr(self.env, name)

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
                # 对于ManiSkillVectorEnv，使用render方法获取所有环境的图像
                if hasattr(self.training_env, 'render'):
                    # ManiSkill向量环境支持批量渲染
                    images = self.training_env.render()
                    if images is not None:
                        if isinstance(images, list):
                            # 如果返回的是图像列表
                            self._save_multi_env_image(images)
                        elif isinstance(images, np.ndarray):
                            # 如果返回的是单个图像或批量图像
                            if len(images.shape) == 4:  # 批量图像 (batch, h, w, c)
                                image_list = [images[i] for i in range(images.shape[0])]
                                self._save_multi_env_image(image_list)
                            else:  # 单个图像
                                self._save_multi_env_image([images])
                        
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
                
                # 确保图像是RGB格式
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img_rgb = img
                elif len(img.shape) == 2:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    continue
                
                # 放置图像
                combined_img[row*h:(row+1)*h, col*w:(col+1)*w] = img_rgb
                
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

def create_maniskill_envs(config):
    """使用ManiSkill的批量环境创建功能 - 这是关键的性能优化！"""
    print(f"使用ManiSkill批量创建 {config.num_envs} 个环境...")
    
    # 环境配置参数
    env_kwargs = {
        "obs_mode": "state",  # 使用state模式，避免RGBD问题
        "control_mode": "pd_joint_delta_pos",
        "render_mode": "rgb_array" if config.enable_render else "none",
        "max_objects": 3,  # 减少物体数量避免复杂性
        "robot_uids": "panda",  # 明确指定机器人类型
        "sim_backend": "gpu",  # 使用GPU后端支持批量环境
    }
    
    # 使用ManiSkill的批量环境创建 - 这比for循环快得多！
    envs = gym.make(
        "StackPickingManiSkill-v1",
        num_envs=config.num_envs,  # 关键参数：批量创建多个环境
        **env_kwargs
    )
    
    # 应用必要的包装器来处理tensor到numpy转换
    if config.enable_render:
        print("已启用渲染模式")
    
    # 使用ManiSkillVectorEnv包装器，参考SAC代码的正确用法
    envs = ManiSkillVectorEnv(
        envs, 
        num_envs=config.num_envs, 
        ignore_terminations=True, 
        record_metrics=True
    )
    
    print(f"成功创建 {config.num_envs} 个并行环境（批量创建）")
    print("注意：直接使用ManiSkillVectorEnv，跳过stable-baselines3的包装器")
    return envs

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
    """训练PPO模型 - 使用批量环境创建优化性能"""
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
        "environment_creation_method": "batch_creation",  # 标记使用批量创建
    }
    save_hyperparameters(hyperparams, current_run)
    
    try:
        # 使用批量环境创建 - 这是关键的性能优化！
        print("使用批量环境创建方式，大幅提升创建速度...")
        vec_env = create_maniskill_envs(config)
        
        print(f"成功创建 {config.num_envs} 个并行环境（批量创建）")
        if config.enable_render:
            print("已启用批量渲染支持")
        
        # 实现自定义PPO训练循环，兼容ManiSkillVectorEnv
        print("开始自定义PPO训练，保持批量环境创建的性能优化...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建PPO智能体
        from torch.distributions import Normal
        import torch.nn.functional as F
        
        class PPOAgent(nn.Module):
            def __init__(self, obs_space, action_space, hidden_size=256):
                super().__init__()
                obs_dim = obs_space.shape[0] if hasattr(obs_space, 'shape') else obs_space.n
                action_dim = action_space.shape[0]
                
                # 共享特征提取器
                self.feature_net = nn.Sequential(
                    nn.Linear(obs_dim, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                )
                
                # 策略网络
                self.policy_mean = nn.Linear(hidden_size, action_dim)
                self.policy_logstd = nn.Parameter(torch.zeros(action_dim))
                
                # 价值网络
                self.value_net = nn.Linear(hidden_size, 1)
                
            def forward(self, obs):
                features = self.feature_net(obs)
                return features
                
            def get_action_and_value(self, obs, action=None):
                features = self.forward(obs)
                
                # 策略输出
                action_mean = self.policy_mean(features)
                action_std = torch.exp(self.policy_logstd)
                dist = Normal(action_mean, action_std)
                
                if action is None:
                    action = dist.sample()
                
                log_prob = dist.log_prob(action).sum(-1)
                entropy = dist.entropy().sum(-1)
                
                # 价值输出
                value = self.value_net(features).squeeze(-1)
                
                return action, log_prob, entropy, value
        
        # 初始化智能体
        agent = PPOAgent(vec_env.single_observation_space, vec_env.single_action_space).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate)
        
        # PPO超参数
        num_steps = config.n_steps
        num_envs = config.num_envs
        batch_size = config.batch_size
        num_epochs = config.n_epochs
        clip_range = config.clip_range
        
        # 存储rollout数据
        obs_buffer = torch.zeros((num_steps, num_envs) + vec_env.single_observation_space.shape).to(device)
        actions_buffer = torch.zeros((num_steps, num_envs) + vec_env.single_action_space.shape).to(device)
        logprobs_buffer = torch.zeros((num_steps, num_envs)).to(device)
        rewards_buffer = torch.zeros((num_steps, num_envs)).to(device)
        dones_buffer = torch.zeros((num_steps, num_envs)).to(device)
        values_buffer = torch.zeros((num_steps, num_envs)).to(device)
        
        print(f"开始PPO训练：{config.total_timesteps}步，{num_envs}个环境（批量创建）")
        
        # 重置环境
        obs, info = vec_env.reset()
        global_step = 0
        
        num_iterations = max(1, config.total_timesteps // (num_steps * num_envs))
        print(f"计算得出训练迭代数: {num_iterations} (总步数:{config.total_timesteps}, 每次rollout:{num_steps * num_envs})")
        
        for iteration in range(num_iterations):
            print(f"\n=== 迭代 {iteration + 1}/{num_iterations} ===")
            
            # Rollout阶段
            for step in range(num_steps):
                global_step += num_envs
                
                with torch.no_grad():
                    action, logprob, entropy, value = agent.get_action_and_value(obs)
                
                # 执行动作
                next_obs, reward, done, truncated, info = vec_env.step(action)
                
                # 存储数据
                obs_buffer[step] = obs
                actions_buffer[step] = action
                logprobs_buffer[step] = logprob
                rewards_buffer[step] = reward
                dones_buffer[step] = done.float()
                values_buffer[step] = value
                
                obs = next_obs
                
                if step % 50 == 0:
                    avg_reward = reward.mean().item()
                    print(f"  步骤 {step}/{num_steps}, 平均奖励: {avg_reward:.3f}")
            
            # 计算优势和回报
            with torch.no_grad():
                next_value = agent.get_action_and_value(obs)[3]
                advantages = torch.zeros_like(rewards_buffer)
                lastgaelam = 0
                
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - dones_buffer[t]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_buffer[t + 1]
                        nextvalues = values_buffer[t + 1]
                    
                    delta = rewards_buffer[t] + config.gamma * nextvalues * nextnonterminal - values_buffer[t]
                    advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                
                returns = advantages + values_buffer
            
            # 准备训练数据
            b_obs = obs_buffer.reshape((-1,) + vec_env.single_observation_space.shape)
            b_actions = actions_buffer.reshape((-1,) + vec_env.single_action_space.shape)
            b_logprobs = logprobs_buffer.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buffer.reshape(-1)
            
            # 归一化优势
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
            # PPO更新
            for epoch in range(num_epochs):
                # 随机打乱数据
                b_inds = torch.randperm(b_obs.shape[0])
                
                for start in range(0, b_obs.shape[0], batch_size):
                    end = start + batch_size
                    mb_inds = b_inds[start:end]
                    
                    # 前向传播
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    
                    # 计算损失
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    
                    # 策略损失
                    pg_loss1 = -b_advantages[mb_inds] * ratio
                    pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # 价值损失
                    v_loss = F.mse_loss(newvalue, b_returns[mb_inds])
                    
                    # 熵损失
                    entropy_loss = entropy.mean()
                    
                    # 总损失
                    loss = pg_loss + config.vf_coef * v_loss - config.ent_coef * entropy_loss
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    optimizer.step()
            
            # 记录统计信息
            if iteration % 10 == 0:
                avg_return = returns.mean().item()
                avg_value = values_buffer.mean().item()
                print(f"迭代 {iteration}: 平均回报={avg_return:.3f}, 平均价值={avg_value:.3f}")
        
        print("PPO训练完成！")
        
        # 保存模型
        torch.save({
            'agent_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.__dict__,
        }, model_path + '_custom.pt')
        
        print(f"模型已保存到: {model_path}_custom.pt")
        
        # 关闭环境
        vec_env.close()
        
        return agent, current_run
        
        # 创建回调列表
        callbacks = []
        
        # 设置检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=2000,
            save_path=current_checkpoint_dir,
            name_prefix="maniskill_ppo_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)
        
        # 如果启用可视化，添加多环境可视化回调
        if config.enable_render:
            multi_env_viz_callback = MultiEnvVisualizationCallback(
                render_freq=config.render_freq, 
                verbose=1
            )
            callbacks.append(multi_env_viz_callback)
            print(f"已启用多环境可视化 - 将批量渲染所有 {config.num_envs} 个环境")
        
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
        print(f"环境创建方式: 批量创建（性能优化）")
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
    
    # 创建评估环境 - 使用批量创建但只创建1个环境
    env_kwargs = {
        "obs_mode": "state",
        "control_mode": "pd_joint_delta_pos",
        "render_mode": "rgb_array",
        "max_objects": 3,
        "robot_uids": "panda",
        "sim_backend": "gpu",
    }
    
    eval_env = gym.make(
        "StackPickingManiSkill-v1",
        num_envs=1,  # 评估时只需要1个环境
        **env_kwargs
    )
    
    # 使用ManiSkillVectorEnv包装器
    eval_env = ManiSkillVectorEnv(eval_env, num_envs=1, ignore_terminations=True, record_metrics=True)
    
    # 加载模型
    model = PPO.load(model_path)
    print("模型加载成功")
    
    # 评估结果存储
    episode_rewards = []
    episode_lengths = []
    success_counts = []
    
    for episode in range(num_episodes):
        print(f"评估第 {episode + 1}/{num_episodes} 个episode")
        
        # ManiSkillVectorEnv的reset返回(obs, info)元组
        obs_info = eval_env.reset()
        if isinstance(obs_info, tuple):
            obs, _ = obs_info
        else:
            obs = obs_info
        
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done and steps < 200:
            # 转换观测为numpy数组（如果需要）
            if hasattr(obs, 'cpu'):  # 如果是tensor
                obs_np = obs.cpu().numpy()
            else:
                obs_np = obs
            
            action, _ = model.predict(obs_np, deterministic=True)
            
            # ManiSkillVectorEnv的step返回(obs, reward, terminated, truncated, info)
            step_result = eval_env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            # 处理奖励（可能是tensor或数组）
            if hasattr(reward, 'item'):
                total_reward += reward.item()
            elif isinstance(reward, (list, np.ndarray)) and len(reward) > 0:
                total_reward += float(reward[0])
            else:
                total_reward += float(reward)
            
            steps += 1
            
            # 检查是否完成
            if isinstance(done, (list, np.ndarray)):
                done = done[0] if len(done) > 0 else False
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # 从info中获取成功计数
        success_count = 0
        if info:
            if isinstance(info, list) and len(info) > 0:
                info_dict = info[0]
                if isinstance(info_dict, dict):
                    success_count = info_dict.get('success_count', 0)
            elif isinstance(info, dict):
                success_count = info.get('success_count', 0)
        
        success_counts.append(success_count)
        print(f"Episode {episode + 1}: 奖励={total_reward:.3f}, 步数={steps}, 成功数={success_count}")
    
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
        
        print(f"使用批量环境创建方式，将创建 {config.num_envs} 个并行环境")
        train_ppo_model(config)
        
    elif args.mode == 'visualize':
        # 可视化训练模式 - 强制启用渲染
        config = PPOTrainingConfig()
        config.enable_render = True
        config.render_freq = 50  # 更频繁的渲染
        config.num_envs = 4  # 使用少量环境以获得更好的可视化效果
        print("启动可视化训练模式...")
        print(f"注意: 使用 {config.num_envs} 个环境进行可视化训练")
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