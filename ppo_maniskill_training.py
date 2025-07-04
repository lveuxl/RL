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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
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

# 导入优化的可视化回调
from optimized_visualization_callback import create_optimized_callback

# 导入超轻量级可视化回调 - 新增！
from ultra_lightweight_visualization import create_ultra_lightweight_callback, create_minimal_callback

class PPOTrainingConfig:
    """PPO训练配置类 - 参考PyBullet项目的配置结构"""
    
    def __init__(self):
        # 基础训练参数
        self.total_timesteps = 1000000  # 总训练步数
        self.num_envs = 1  # 并行环境数量
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
        
        # 可视化设置 - 优化后的默认值
        self.enable_render = True  # 启用可视化
        self.render_freq = 200  # 降低可视化频率，减少卡顿
        
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

class ManiSkillVecEnvWrapper(VecEnv):
    """
    将ManiSkillVectorEnv包装成stable-baselines3兼容的VecEnv
    这是关键的兼容性包装器
    """
    
    def __init__(self, maniskill_vec_env):
        self.maniskill_env = maniskill_vec_env
        self.num_envs = maniskill_vec_env.num_envs
        
        # 获取观测和动作空间
        self.observation_space = maniskill_vec_env.single_observation_space
        self.action_space = maniskill_vec_env.single_action_space
        
        # 设置设备
        self.device = getattr(maniskill_vec_env, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 确保空间不为None
        if self.observation_space is None or self.action_space is None:
            raise ValueError(f"无法获取观测空间或动作空间: obs_space={self.observation_space}, action_space={self.action_space}")
        
        # 初始化VecEnv - 使用手动初始化避免stable-baselines3的严格检查
        # 这是因为ManiSkillVectorEnv的某些属性可能与stable-baselines3的期望不完全匹配
        self.num_envs = self.num_envs
        self.observation_space = self.observation_space
        self.action_space = self.action_space
        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None
        self.metadata = {}
        self.closed = False
        
        # 添加episode奖励跟踪
        self.episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        
        print(f"成功创建ManiSkillVecEnvWrapper: {self.num_envs}个环境, obs_shape={self.observation_space.shape}, action_shape={self.action_space.shape}")
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        """检查环境是否被指定的包装器包装"""
        return [False] * self.num_envs
        
    def reset(self):
        """重置环境"""
        obs_info = self.maniskill_env.reset()
        if isinstance(obs_info, tuple):
            obs, info = obs_info
        else:
            obs = obs_info
            info = [{}] * self.num_envs
        
        # 重置episode统计
        self.episode_rewards.fill(0.0)
        self.episode_lengths.fill(0)
        
        # 转换观测为numpy
        obs_np = self._convert_obs_to_numpy(obs)
        return obs_np
    
    def step_async(self, actions):
        """异步执行动作"""
        # 将numpy动作转换为tensor
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float().to(self.device)
        
        self.actions = actions
    
    def step_wait(self):
        """等待步骤完成并返回结果"""
        # 执行动作
        step_result = self.maniskill_env.step(self.actions)
        
        if len(step_result) == 5:
            obs, rewards, terminated, truncated, infos = step_result
            # 对于新版gymnasium，done = terminated | truncated
            dones = terminated | truncated
        else:
            obs, rewards, dones, infos = step_result
        
        # 转换为numpy格式
        obs_np = self._convert_obs_to_numpy(obs)
        rewards_np = self._convert_to_numpy(rewards)
        dones_np = self._convert_to_numpy(dones)
        
        # 确保infos是列表格式，并且每个元素都是字典
        if not isinstance(infos, list):
            infos = [{}] * self.num_envs
        else:
            # 确保每个info都是字典
            for i in range(len(infos)):
                if not isinstance(infos[i], dict):
                    infos[i] = {}
        
        # 确保infos列表长度正确
        while len(infos) < self.num_envs:
            infos.append({})
        
        # 更新episode统计
        self.episode_rewards += rewards_np
        self.episode_lengths += 1
        
        # 处理episode结束的情况
        for i in range(self.num_envs):
            if dones_np[i]:
                # 在episode结束时，将总奖励添加到info中
                # 确保info是字典类型
                if not isinstance(infos[i], dict):
                    infos[i] = {}
                
                infos[i]['r'] = float(self.episode_rewards[i])
                infos[i]['l'] = int(self.episode_lengths[i])
                
                # 调试信息
                print(f"Episode结束 - 环境{i}: 奖励={self.episode_rewards[i]:.3f}, 步数={self.episode_lengths[i]}")
                
                # 重置该环境的统计
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
        
        return obs_np, rewards_np, dones_np, infos
    
    def close(self):
        """关闭环境"""
        return self.maniskill_env.close()
    
    def get_attr(self, attr_name, indices=None):
        """获取环境属性"""
        return getattr(self.maniskill_env, attr_name)
    
    def set_attr(self, attr_name, value, indices=None):
        """设置环境属性"""
        setattr(self.maniskill_env, attr_name, value)
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """调用环境方法"""
        return getattr(self.maniskill_env, method_name)(*method_args, **method_kwargs)
    
    def render(self, mode="rgb_array"):
        """渲染环境"""
        return self.maniskill_env.render()
    
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
            return numpy_data
        elif isinstance(tensor_data, (list, tuple)):
            return np.array(tensor_data)
        else:
            return tensor_data

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
    """使用ManiSkill的批量环境创建功能，然后包装为stable-baselines3兼容格式"""
    print(f"使用ManiSkill批量创建 {config.num_envs} 个环境...")
    
    # 环境配置参数 - 针对可视化卡顿进行优化
    env_kwargs = {
        "obs_mode": "state",  # 使用state模式，避免RGBD问题
        "control_mode": "pd_joint_delta_pos",
        "render_mode": "human" if config.enable_render else "none",  # 强制使用rgb_array模式
        "max_objects": 6,  # 进一步减少物体数量
        "robot_uids": "panda",  # 明确指定机器人类型
        "sim_backend": "gpu",  # 使用GPU后端支持批量环境
        # 新增：针对可视化优化的参数
        # "sim_freq": 240,  # 降低仿真频率，减少计算负担
        # "control_freq": 20,  # 降低控制频率
        "shader_dir": "minimal",  # 使用最简着色器（如果支持）
    }
    
    # # 如果启用可视化，进一步优化渲染设置
    # if config.enable_render:
    #     print("检测到可视化模式，应用渲染优化...")
    #     # 添加渲染优化参数
    #     env_kwargs.update({
    #         "camera_width": 256,   # 降低渲染分辨率
    #         "camera_height": 256,  # 降低渲染分辨率
    #         "render_gpu_device_id": 0,  # 指定GPU设备
    #     })
    #     print("已应用可视化渲染优化")
    
    # 使用ManiSkill的批量环境创建 - 这比for循环快得多！
    envs = gym.make(
        "StackPickingManiSkill-v1",
        num_envs=config.num_envs,  # 关键参数：批量创建多个环境
        **env_kwargs
    )
    
    # 应用必要的包装器来处理tensor到numpy转换
    if config.enable_render:
        print("已启用渲染模式 - 使用rgb_array格式避免窗口冲突")
    
    # 使用ManiSkillVectorEnv包装器
    # 类型转换以消除linter警告，实际运行时是兼容的
    maniskill_vec_env = ManiSkillVectorEnv(
        envs,  # type: ignore  # gym.Env与BaseEnv在运行时兼容
        num_envs=config.num_envs, 
        ignore_terminations=True, 
        record_metrics=True
    )
    
    # 关键步骤：包装为stable-baselines3兼容的VecEnv
    sb3_compatible_env = ManiSkillVecEnvWrapper(maniskill_vec_env)
    
    print(f"成功创建 {config.num_envs} 个并行环境（批量创建）")
    print("已包装为stable-baselines3兼容格式")
    return sb3_compatible_env

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
    """训练PPO模型 - 使用stable-baselines3的PPO与批量环境创建"""
    print("开始PPO模型训练（使用stable-baselines3 PPO）...")
    
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
        "environment_creation_method": "batch_creation_with_sb3_wrapper",  # 标记使用批量创建+包装器
    }
    save_hyperparameters(hyperparams, current_run)
    
    try:
        # 使用批量环境创建 + stable-baselines3包装器 - 这是关键的性能优化！
        print("使用批量环境创建方式 + stable-baselines3兼容包装器...")
        vec_env = create_maniskill_envs(config)
        
        print(f"成功创建 {config.num_envs} 个并行环境（批量创建+包装器）")
        if config.enable_render:
            print("已启用批量渲染支持")
        
        # 使用stable-baselines3的PPO - 这是您要求的关键改动！
        print("使用stable-baselines3的PPO进行训练...")
        
        # 创建PPO模型
        model = PPO(
            "MlpPolicy",  # 使用多层感知机策略
            vec_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            policy_kwargs=config.policy_kwargs,
            tensorboard_log=tb_log_dir,
            device="auto",  # 自动选择设备
            verbose=1
        )
        
        print(f"PPO模型创建成功，使用设备: {model.device}")
        
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
        
        # 如果启用可视化，添加超轻量级可视化回调 - 彻底解决卡顿！
        if config.enable_render:
            # 根据环境数量选择不同的可视化策略
            if config.num_envs == 1:
                # 单环境：使用超轻量级可视化
                print("单环境模式 - 使用超轻量级可视化回调")
                viz_callback = create_ultra_lightweight_callback(
                    render_freq=config.render_freq,
                    max_fps=5,  # 限制最大5FPS，避免卡顿
                    enable_display=True
                )
                callbacks.append(viz_callback)
                print("已启用超轻量级可视化 - 专门解决单环境卡顿问题")
                
            elif config.num_envs <= 4:
                # 少量环境：使用优化的可视化回调
                print("少量环境模式 - 使用优化可视化回调")
                optimized_viz_callback = create_optimized_callback(
                    render_freq=config.render_freq,
                    max_envs=config.num_envs,
                    enable_async=True,
                    save_to_disk=False
                )
                callbacks.append(optimized_viz_callback)
                print(f"已启用优化可视化 - 异步处理，显示{config.num_envs}个环境")
                
            else:
                # 大量环境：使用最小化可视化
                print("大量环境模式 - 使用最小化可视化回调")
                minimal_viz_callback = create_minimal_callback(
                    render_freq=config.render_freq * 2  # 进一步降低频率
                )
                callbacks.append(minimal_viz_callback)
                print(f"已启用最小化可视化 - 仅监控训练状态，不显示图像")
        
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
        print(f"环境创建方式: 批量创建（性能优化）+ stable-baselines3包装器")
        print(f"PPO实现: stable-baselines3 PPO（官方实现）")
        print(f"可视化: {'优化异步可视化' if config.enable_render else '禁用'}")
        print(f"TensorBoard日志保存到: {tb_log_dir}")
        print(f"{'='*50}\n")
        
        # 开始训练 - 使用stable-baselines3的PPO！
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
    
    # 创建评估环境配置
    config = PPOTrainingConfig()
    config.num_envs = 1  # 评估时只需要1个环境
    config.enable_render = True  # 评估时启用渲染
    
    # 创建评估环境
    eval_env = create_maniskill_envs(config)
    
    # 加载模型
    model = PPO.load(model_path)
    print("模型加载成功")
    
    # 评估结果存储
    episode_rewards = []
    episode_lengths = []
    success_counts = []
    
    for episode in range(num_episodes):
        print(f"评估第 {episode + 1}/{num_episodes} 个episode")
        
        obs = eval_env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done and steps < 200:
            # 确保obs是正确的格式
            if isinstance(obs, tuple):
                obs = obs[0]  # 如果是元组，取第一个元素
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            # 处理奖励（现在应该已经是numpy格式）
            if isinstance(reward, (list, np.ndarray)) and len(reward) > 0:
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
    parser = argparse.ArgumentParser(description='ManiSkill PPO训练（使用stable-baselines3 PPO）')
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
        
        print(f"使用stable-baselines3 PPO + 批量环境创建方式，将创建 {config.num_envs} 个并行环境")
        train_ppo_model(config)
        
    elif args.mode == 'visualize':
        # 可视化训练模式 - 强制启用渲染
        config = PPOTrainingConfig()
        config.enable_render = True
        config.render_freq = 50  # 更频繁的渲染
        config.num_envs = 1  
        print("启动可视化训练模式（使用stable-baselines3 PPO）...")
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