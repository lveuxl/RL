"""
使用stable-baselines3训练EnvClutter环境
目标：学习最优的抓取顺序
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import torch
import time

# 尝试导入MaskablePPO，如果没有安装sb3-contrib则使用普通PPO
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    MASKABLE_AVAILABLE = True
    print("✅ 使用MaskablePPO进行动作掩码训练")
except ImportError:
    from stable_baselines3 import PPO
    MASKABLE_AVAILABLE = False
    print("⚠️ 未检测到sb3-contrib，将使用普通PPO（建议安装: pip install sb3-contrib）")

from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.logger import TensorBoardOutputFormat
import json

# 导入支持抓取顺序学习的环境版本
from env_clutter import EnvClutterEnv  # 新版本，支持use_ideal_oracle
print("✅ 使用env_clutter环境（抓取顺序学习）")
HAS_IDEAL_ORACLE = True

from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode
import mani_skill.envs

from wrappers.mask_wrapper import ExtractMaskWrapper, SB3CompatWrapper, ActionConversionWrapper
from wrappers.maskable_wrapper import create_maskable_env, MaskableVectorEnvWrapper


class TrainingMonitorCallback(BaseCallback):
    """
    训练监控回调 - 实时监控训练指标和梯度信息
    """
    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.best_mean_reward = -np.inf
        
    def _init_callback(self) -> None:
        # Create logs dir if needed
        if self.logger.get_dir() is not None:
            self.log_dir = self.logger.get_dir()
        else:
            self.log_dir = None
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 记录训练统计信息
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                # 从episode信息缓冲区获取统计数据
                rewards = [ep['r'] for ep in self.model.ep_info_buffer]
                lengths = [ep['l'] for ep in self.model.ep_info_buffer]
                
                if rewards:
                    mean_reward = np.mean(rewards)
                    mean_length = np.mean(lengths)
                    
                    # 记录到TensorBoard
                    self.logger.record("train/mean_episode_reward", mean_reward)
                    self.logger.record("train/mean_episode_length", mean_length)
                    
                    # 检查是否是最佳性能
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        self.logger.record("train/best_mean_reward", self.best_mean_reward)
            
            # 记录学习率
            if hasattr(self.model, 'learning_rate'):
                current_lr = self.model.learning_rate
                if callable(current_lr):
                    current_lr = current_lr(1.0)  # Get current learning rate
                self.logger.record("train/learning_rate", current_lr)
            
            # 记录梯度统计
            if hasattr(self.model.policy, 'parameters'):
                total_norm = 0.0
                param_count = 0
                for param in self.model.policy.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                
                if param_count > 0:
                    total_norm = total_norm ** (1. / 2)
                    self.logger.record("train/gradient_norm", total_norm)
            
        return True


class GraspSequenceAnalysisCallback(BaseCallback):
    """
    抓取序列分析回调 - 分析学习到的抓取策略
    """
    def __init__(self, eval_env, check_freq: int = 50000, n_eval_episodes: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.n_eval_episodes = n_eval_episodes
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self._analyze_grasp_sequences()
        return True
    
    def _analyze_grasp_sequences(self):
        """分析抓取序列模式"""
        try:
            sequences = []
            success_count = 0
            
            for episode in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                sequence = []
                episode_success = False
                
                for step in range(20):  # 最多20步
                    action, _ = self.model.predict(obs, deterministic=True)
                    sequence.append(int(action[0]) if isinstance(action, np.ndarray) else int(action))
                    
                    obs, reward, done, info = self.eval_env.step(action)
                    
                    if done[0] if isinstance(done, np.ndarray) else done:
                        if isinstance(info, list) and len(info) > 0:
                            episode_success = info[0].get('success', False)
                        elif isinstance(info, dict):
                            episode_success = info.get('success', False)
                        break
                
                sequences.append(sequence[:9])  # 只记录前9个动作
                if episode_success:
                    success_count += 1
            
            # 分析序列一致性
            if sequences:
                unique_sequences = len(set(tuple(seq) for seq in sequences))
                sequence_consistency = 1.0 - (unique_sequences - 1) / max(len(sequences) - 1, 1)
                
                # 记录分析结果
                self.logger.record("eval/sequence_consistency", sequence_consistency)
                self.logger.record("eval/unique_sequences", unique_sequences)
                self.logger.record("eval/success_rate", success_count / self.n_eval_episodes)
                
                # 记录最常见的序列
                from collections import Counter
                sequence_counter = Counter(tuple(seq) for seq in sequences)
                most_common_seq = sequence_counter.most_common(1)[0][0] if sequence_counter else []
                self.logger.record("eval/most_common_sequence_length", len(most_common_seq))
                
                if self.verbose >= 1:
                    print(f"\n🎯 抓取序列分析 (步数: {self.n_calls}):")
                    print(f"  序列一致性: {sequence_consistency:.2%}")
                    print(f"  独特序列数: {unique_sequences}")
                    print(f"  评估成功率: {success_count/self.n_eval_episodes:.2%}")
                    if most_common_seq:
                        print(f"  最常见序列: {list(most_common_seq)}")
                        
        except Exception as e:
            if self.verbose >= 1:
                print(f"抓取序列分析失败: {e}")


def create_learning_rate_schedule(initial_lr: float = 1e-4, final_lr_ratio: float = 0.1):
    """
    创建学习率衰减调度器
    """
    def lr_schedule(progress_remaining: float) -> float:
        """
        进度从1.0（开始）到0.0（结束）
        """
        return initial_lr * (final_lr_ratio + (1.0 - final_lr_ratio) * progress_remaining)
    
    return lr_schedule


def create_env(env_id="EnvClutter-v1", num_envs=128, record_video=False, video_dir="./videos", **env_kwargs):
    """
    创建训练环境
    """
    
    # 构建环境参数，根据版本决定是否包含use_ideal_oracle
    env_params = {
        "num_envs": num_envs,
        "obs_mode": "state",
        "control_mode": "pd_ee_delta_pose", 
        "reward_mode": "dense",
        "sim_backend": "gpu",
        "render_mode": "rgb_array" if record_video else None,
        "use_discrete_action": True,  # 启用离散动作
        **env_kwargs
    }
    
    # 只在支持的版本中添加use_ideal_oracle参数
    if HAS_IDEAL_ORACLE:
        env_params["use_ideal_oracle"] = True  # 使用理想化神谕抓取
    
    # 创建原始环境（直接创建多环境版本）
    env = gym.make(env_id, **env_params)
    
    # 获取最大物体数量
    max_n = env.unwrapped.MAX_N if hasattr(env.unwrapped, 'MAX_N') else 9
    
    if MASKABLE_AVAILABLE:
        # 使用MaskablePPO专用包装器
        print("🎯 配置MaskablePPO专用环境包装...")
        env = SB3CompatWrapper(env)
        env = ExtractMaskWrapper(env, max_n=max_n)
        # 转换为SB3向量环境
        vec_env = ManiSkillSB3VectorEnv(env)
        # 添加MaskablePPO向量环境包装器
        vec_env = MaskableVectorEnvWrapper(vec_env, max_n=max_n)
    else:
        # 使用普通PPO包装器
        print("⚠️ 配置普通PPO环境包装...")
        env = SB3CompatWrapper(env)
        env = ExtractMaskWrapper(env, max_n=max_n)
        env = ActionConversionWrapper(env)
        # 转换为SB3向量环境
        vec_env = ManiSkillSB3VectorEnv(env)
    
    return vec_env


def create_eval_env(env_id="EnvClutter-v1", num_envs=16, record_video=False, video_dir="./videos", **env_kwargs):
    """
    创建评估环境
    """
    
    # 构建环境参数，根据版本决定是否包含use_ideal_oracle
    env_params = {
        "num_envs": num_envs,
        "obs_mode": "state",
        "control_mode": "pd_ee_delta_pose",
        "reward_mode": "dense", 
        "sim_backend": "gpu",
        "render_mode": "rgb_array" if record_video else None,
        "use_discrete_action": True,
        **env_kwargs
    }
    
    # 只在支持的版本中添加use_ideal_oracle参数
    if HAS_IDEAL_ORACLE:
        env_params["use_ideal_oracle"] = True
    
    env = gym.make(env_id, **env_params)
    
    # 只有在评估环境且环境数量较少时才录制视频
    if record_video and num_envs <= 4:
        timestamp = int(time.time())
        unique_trajectory_name = f"eval_trajectory_{timestamp}"
        
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_video=True,
            trajectory_name=unique_trajectory_name,
            max_steps_per_video=2000,
            video_fps=30,
        )
    
    # 获取最大物体数量
    max_n = env.unwrapped.MAX_N if hasattr(env.unwrapped, 'MAX_N') else 9
    
    if MASKABLE_AVAILABLE:
        # 使用MaskablePPO专用包装器
        env = SB3CompatWrapper(env)
        env = ExtractMaskWrapper(env, max_n=max_n)
        vec_env = ManiSkillSB3VectorEnv(env)
        vec_env = MaskableVectorEnvWrapper(vec_env, max_n=max_n)
    else:
        # 使用普通PPO包装器
        env = SB3CompatWrapper(env)
        env = ExtractMaskWrapper(env, max_n=max_n)
        env = ActionConversionWrapper(env)
        vec_env = ManiSkillSB3VectorEnv(env)
    
    return vec_env


def train_ppo(args):
    """
    训练PPO智能体学习最优抓取顺序
    """
    print(f"开始训练EnvClutter环境 - 目标：学习自上而下的抓取顺序")
    print(f"环境版本: {'抓取顺序学习版本' if HAS_IDEAL_ORACLE else '基础版本（功能受限）'}")
    print(f"理想化神谕抓取: {'✅启用' if HAS_IDEAL_ORACLE else '❌未启用'}")
    print(f"并行环境数: {args.num_envs}")
    print(f"总训练步数: {args.total_timesteps}")
    print(f"每回合抓取次数: 9次（对应9个action）")
    
    # 创建训练环境
    print("创建训练环境...")
    vec_env = create_env(
        env_id="EnvClutter-v1",
        num_envs=args.num_envs,
        record_video=False,
        video_dir=os.path.join(args.log_dir, "train_videos"),
    )
    
    # 创建评估环境
    print("创建评估环境...")
    eval_env = create_eval_env(
        env_id="EnvClutter-v1", 
        num_envs=args.eval_envs,
        record_video=args.record_video,
        video_dir=os.path.join(args.log_dir, "eval_videos"),
    )
    
    # 创建PPO模型 - 支持动作掩码
    print("创建PPO模型...")
    
    # 🚀 精调超参数 - 专门为自上而下抓取顺序学习优化
    # 动态调整批次大小避免内存问题
    total_batch_size = args.num_envs * 2048  # n_steps * n_envs
    optimal_batch_size = min(512, max(64, total_batch_size // 8))
    
    # 创建学习率调度
    lr_schedule = create_learning_rate_schedule(
        initial_lr=args.learning_rate, 
        final_lr_ratio=0.1
    )
    
    model_kwargs = {
        "gamma": 0.99,              # 更高折扣因子，强调长期策略
        "gae_lambda": 0.95,         # GAE平滑优势估计
        "n_steps": 2048,            # 增加经验收集，更好学习序列决策
        "batch_size": optimal_batch_size,  # 动态批次大小
        "n_epochs": 12,             # 平衡训练效果和效率
        "ent_coef": 0.008,          # 稍低熵系数，减少后期的随机探索
        "learning_rate": lr_schedule,  # 使用学习率调度
        "clip_range": 0.15,         # 稍紧的裁剪，提高训练稳定性
        "clip_range_vf": 0.15,      # 价值函数裁剪，增强稳定性
        "max_grad_norm": 0.5,       # 梯度裁剪，防止梯度爆炸
        "vf_coef": 0.25,           # 价值函数损失系数
        "target_kl": 0.02,         # 早停KL散度阈值
        "verbose": 1,
        "tensorboard_log": args.log_dir,
        "policy_kwargs": {
            # 🎯 网络架构优化：适合处理复杂抓取序列决策
            "net_arch": dict(
                pi=[512, 512, 512, 256],   # 策略网络：更深层次理解抓取优先级
                vf=[512, 512, 256]         # 价值网络：准确估计长期回报
            ),
            "activation_fn": torch.nn.ReLU,
            "squash_output": True,      # 确保输出范围合适
            "normalize_images": False,   # 不使用图像，设为False
            "optimizer_class": torch.optim.AdamW,  # 使用AdamW优化器
            "optimizer_kwargs": dict(
                weight_decay=0.01,       # L2正则化
                eps=1e-8,               # 数值稳定性
            ),
        }
    }
    
    # 根据是否支持掩码选择模型类型
    if MASKABLE_AVAILABLE:
        # 为MaskablePPO调整策略参数
        maskable_kwargs = model_kwargs.copy()
        maskable_kwargs["policy_kwargs"]["features_extractor_class"] = None  # 使用默认特征提取器
        
        model = MaskablePPO("MlpPolicy", vec_env, **maskable_kwargs)
        print("✅ 使用MaskablePPO，支持动作掩码")
        print(f"   动作空间: {vec_env.action_space}")
        print(f"   观测空间: {vec_env.observation_space}")
    else:
        model = PPO("MlpPolicy", vec_env, **model_kwargs)
        print("⚠️ 使用普通PPO，不支持动作掩码")
        print("   建议安装sb3-contrib获得完整功能: pip install sb3-contrib")
    
    param_count = sum(p.numel() for p in model.policy.parameters())
    print(f"模型创建完成，参数量: {param_count:,}")
    print(f"动态批次大小: {optimal_batch_size}")
    print(f"初始学习率: {args.learning_rate:.2e}")
    
    # 创建增强的回调函数系统
    callbacks = []
    
    # 训练监控回调
    training_monitor = TrainingMonitorCallback(
        check_freq=1000,
        verbose=1
    )
    callbacks.append(training_monitor)
    
    # 评估回调
    if args.eval_freq > 0:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.model_dir, "best_model"),
            log_path=os.path.join(args.log_dir, "eval_logs"),
            eval_freq=args.eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=args.n_eval_episodes,
            verbose=1,
        )
        callbacks.append(eval_callback)
    
    # 抓取序列分析回调
    sequence_analysis = GraspSequenceAnalysisCallback(
        eval_env=eval_env,
        check_freq=args.eval_freq,
        n_eval_episodes=3,  # 快速分析
        verbose=1
    )
    callbacks.append(sequence_analysis)
    
    # 检查点回调
    if args.save_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=args.save_freq,
            save_path=args.model_dir,
            name_prefix="ppo_envclutter_topdown",
            verbose=1,
        )
        callbacks.append(checkpoint_callback)
    
    # 开始训练
    print("开始训练...")
    print("优化的奖励设计（按用户要求的优先级）：")
    if HAS_IDEAL_ORACLE:
        print("1. 【优先级1】成功抓取奖励:")
        print("   - 基础抓取奖励: +5.0")
        print("   - 高度奖励: +3.0 * normalized_height (鼓励自上而下)")
        print("   - 完成所有物体: +20.0")
        print("2. 【优先级2】位移惩罚: -1.5 * displacement (减少其他物体移动)")
        print("3. 【优先级3】时间惩罚: -0.1 (鼓励效率)")
        print("4. 失败惩罚: -1.0")
        print("")
        print("策略学习特性:")
        print("- 增强观测包含：物体高度、相对高度、抓取优先级")
        print("- 动作掩码确保不重复抓取")
        print("- 自上而下策略通过高度奖励强化")
    else:
        print("⚠️ 使用基础环境，奖励函数可能不包含所有抓取顺序学习特性")
        print("建议：使用env_clutter.py版本以获得完整功能")
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # 保存最终模型
    final_model_path = os.path.join(args.model_dir, "ppo_envclutter_final")
    model.save(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # 关闭环境
    vec_env.close()
    eval_env.close()
    
    print("训练完成！")


def evaluate_model(args):
    """
    评估训练好的模型
    """
    print(f"开始评估模型: {args.model_path}")
    
    # 创建评估环境
    eval_env = create_eval_env(
        env_id="EnvClutter-v1",
        num_envs=1,  # 评估时使用单环境
        record_video=True,
        video_dir=os.path.join(args.log_dir, "eval_videos"),
    )
    
    # 加载模型
    if MASKABLE_AVAILABLE:
        try:
            model = MaskablePPO.load(args.model_path)
            print("✅ 加载MaskablePPO模型")
        except:
            model = PPO.load(args.model_path)
            print("⚠️ 尝试加载为普通PPO模型")
    else:
        model = PPO.load(args.model_path)
    
    print("开始评估...")
    
    # 运行评估
    obs = eval_env.reset()
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    action_sequences = []  # 记录动作序列
    
    current_episode_reward = 0
    current_episode_length = 0
    current_action_sequence = []
    
    for step in range(args.eval_steps):
        # 预测动作
        action, _states = model.predict(obs, deterministic=True)
        current_action_sequence.append(action[0] if isinstance(action, np.ndarray) else action)
        
        # 执行动作
        obs, reward, done, info = eval_env.step(action)
        
        current_episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
        current_episode_length += 1
        
        if done[0] if isinstance(done, np.ndarray) else done:
            # Episode结束
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            action_sequences.append(current_action_sequence)
            
            # 检查成功率
            if isinstance(info, list) and len(info) > 0:
                success = info[0].get('success', False)
            elif isinstance(info, dict):
                success = info.get('success', False)
            else:
                success = False
            
            episode_successes.append(success)
            
            print(f"Episode完成: 奖励={current_episode_reward:.2f}, 长度={current_episode_length}, 成功={success}")
            print(f"动作序列: {current_action_sequence[:9]}")  # 显示前9个动作
            
            # 重置计数器
            current_episode_reward = 0
            current_episode_length = 0
            current_action_sequence = []
    
    # 打印评估结果
    if episode_rewards:
        print(f"\n评估结果 (共{len(episode_rewards)}个episode):")
        print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"平均长度: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"成功率: {np.mean(episode_successes):.2%}")
        
        # 分析动作序列模式
        if action_sequences:
            print("\n动作序列分析:")
            for i, seq in enumerate(action_sequences[:3]):  # 显示前3个episode的序列
                print(f"Episode {i+1} 动作序列: {seq[:9]}")
    
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description='训练EnvClutter环境学习最优抓取顺序')
    
    # 基本参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], 
                       help='运行模式：训练或评估')
    parser.add_argument('--total_timesteps', type=int, default=2_000_000, 
                       help='总训练步数 - 充分学习抓取顺序')
    parser.add_argument('--num_envs', type=int, default=32, 
                       help='并行训练环境数量 - 平衡效率与稳定性')
    parser.add_argument('--eval_envs', type=int, default=4, 
                       help='并行评估环境数量')
    
    # PPO超参数 - 已优化
    parser.add_argument('--gamma', type=float, default=0.95, help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--n_steps', type=int, default=512, help='每环境的步数')
    parser.add_argument('--batch_size', type=int, default=4096, help='批次大小')
    parser.add_argument('--n_epochs', type=int, default=10, help='PPO更新轮数')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='熵系数')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--clip_range', type=float, default=0.2, help='PPO裁剪范围')
    
    # 目录和日志
    parser.add_argument('--log_dir', type=str, default='./logs/sb3_topdown', 
                       help='日志目录')
    parser.add_argument('--model_dir', type=str, default='./models/sb3_topdown', 
                       help='模型保存目录')
    
    # 评估和保存 - 优化频率确保及时反馈
    parser.add_argument('--eval_freq', type=int, default=50000, 
                       help='评估频率（步数）- 较低频率减少训练中断')
    parser.add_argument('--n_eval_episodes', type=int, default=5, 
                       help='每次评估的episode数 - 快速评估')
    parser.add_argument('--save_freq', type=int, default=100000, 
                       help='模型保存频率（步数）')
    
    # 视频录制
    parser.add_argument('--record_video', action='store_true', 
                       help='是否录制评估视频')
    
    # 评估模式参数
    parser.add_argument('--model_path', type=str, 
                       help='评估模式下要加载的模型路径')
    parser.add_argument('--eval_steps', type=int, default=10000, 
                       help='评估模式下的总步数')
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    if args.mode == 'train':
        train_ppo(args)
    elif args.mode == 'eval':
        if not args.model_path:
            print("错误：评估模式需要指定--model_path参数")
            return
        evaluate_model(args)


if __name__ == "__main__":
    main()