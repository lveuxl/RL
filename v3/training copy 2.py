"""
使用stable-baselines3训练EnvClutter环境
目标：学习最优的抓取顺序
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch
import time

# 导入支持抓取顺序学习的环境版本
from env_clutter import EnvClutterEnv  # 新版本，支持use_ideal_oracle
print("✅ 使用env_clutter环境（抓取顺序学习）")
HAS_IDEAL_ORACLE = True

from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode
import mani_skill.envs

from wrappers.mask_wrapper import ExtractMaskWrapper, SB3CompatWrapper, ActionConversionWrapper


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
    
    # 添加SB3兼容性包装器
    env = SB3CompatWrapper(env)
    
    # 添加掩码提取包装器 - 确保MAX_N与环境一致
    max_n = env.unwrapped.MAX_N if hasattr(env.unwrapped, 'MAX_N') else 15
    env = ExtractMaskWrapper(env, max_n=max_n)
    
    # 使用动作转换包装器处理离散动作
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
    
    env = SB3CompatWrapper(env)
    max_n = env.unwrapped.MAX_N if hasattr(env.unwrapped, 'MAX_N') else 15
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
    
    # 创建PPO模型
    print("创建PPO模型...")
    
    # 优化后的超参数 - 适合学习抓取顺序
    model_kwargs = {
        "gamma": 0.95,              # 较高的折扣因子，重视长期奖励
        "gae_lambda": 0.95,
        "n_steps": 512,             # 增加步数以收集更多经验
        "batch_size": 4096,         # 大批次训练
        "n_epochs": 10,
        "ent_coef": 0.01,           # 保持探索
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "verbose": 1,
        "tensorboard_log": args.log_dir,
        "policy_kwargs": {
            "net_arch": dict(pi=[256, 256, 256], vf=[256, 256, 256]),
            "activation_fn": torch.nn.ReLU,
        }
    }
    
    model = PPO("MlpPolicy", vec_env, **model_kwargs)
    
    print(f"模型创建完成，参数量: {sum(p.numel() for p in model.policy.parameters())}")
    
    # 创建回调函数
    callbacks = []
    
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
        )
        callbacks.append(eval_callback)
    
    # 检查点回调
    if args.save_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=args.save_freq,
            save_path=args.model_dir,
            name_prefix="ppo_envclutter_topdown",
        )
        callbacks.append(checkpoint_callback)
    
    # 开始训练
    print("开始训练...")
    print("奖励设计：")
    if HAS_IDEAL_ORACLE:
        print("1. 成功抓取奖励: +3.0")
        print("2. 完成所有物体: +15.0")
        print("3. 其他物体位移惩罚: -0.8 * displacement")
        print("4. 失败惩罚: -0.5")
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
    parser.add_argument('--total_timesteps', type=int, default=10000, 
                       help='总训练步数')
    #1_000_000
    parser.add_argument('--num_envs', type=int, default=64, 
                       help='并行训练环境数量')
    parser.add_argument('--eval_envs', type=int, default=8, 
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
    
    # 评估和保存
    parser.add_argument('--eval_freq', type=int, default=10000, 
                       help='评估频率（步数）')
    parser.add_argument('--n_eval_episodes', type=int, default=10, 
                       help='每次评估的episode数')
    parser.add_argument('--save_freq', type=int, default=50000, 
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