"""
使用stable-baselines3的MaskablePPO训练EnvClutter环境
按照官方example.py的风格编写
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from env_clutter import EnvClutterEnv

# 尝试导入MaskablePPO，如果没有安装sb3-contrib则使用普通PPO
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    # 强制禁用MaskablePPO，因为ManiSkillSB3VectorEnv不兼容离散动作空间
    MASKABLE_AVAILABLE = False
except ImportError:
    print("⚠️ 未检测到sb3-contrib，将使用普通PPO（建议安装: pip install sb3-contrib）")
    MASKABLE_AVAILABLE = False

from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode
import mani_skill.envs  # 确保环境已注册

from wrappers.mask_wrapper import ExtractMaskWrapper, SB3CompatWrapper, ActionConversionWrapper
import torch
import time  # 添加时间模块用于生成唯一文件名


def mask_fn(obs):
    """
    从观测中提取动作掩码的回调函数
    用于ActionMasker包装器
    
    新的观测结构：[物体特征(total_objects * 8), 动作掩码(total_objects), 全局特征(3)]
    """
    total_objects = 9  # 用户要求每回合抓9次，对应9个物体
    
    if isinstance(obs, np.ndarray):
        if obs.ndim == 1:
            # 1D情况：观测结构为 [物体特征(9*8=72), 掩码(9), 全局特征(3)]
            mask_start = total_objects * 8  # 掩码开始位置
            mask_end = mask_start + total_objects  # 掩码结束位置
            mask = obs[mask_start:mask_end]
        else:
            # 2D情况 (batch_size, features)
            mask_start = total_objects * 8
            mask_end = mask_start + total_objects
            mask = obs[:, mask_start:mask_end]
            # 对于ActionMasker，需要返回1D掩码，取第一个环境的掩码
            mask = mask[0] if mask.shape[0] > 0 else mask
    elif hasattr(obs, 'cpu'):  # torch.Tensor
        if obs.dim() == 1:
            mask_start = total_objects * 8
            mask_end = mask_start + total_objects
            mask = obs[mask_start:mask_end].cpu().numpy()
        else:
            mask_start = total_objects * 8
            mask_end = mask_start + total_objects
            mask = obs[:, mask_start:mask_end].cpu().numpy()
            mask = mask[0] if mask.shape[0] > 0 else mask
    else:
        # 如果没有掩码，返回全1（所有动作都可用）
        mask = np.ones(total_objects, dtype=np.float32)
    
    # 确保掩码是布尔类型（ActionMasker期望的格式）
    return mask.astype(bool)


def create_env(env_id="EnvClutter-v1", num_envs=128, record_video=False, video_dir="./videos", trajectory_name="train_trajectory", **env_kwargs):
    """
    创建训练环境 - 针对抓取顺序学习优化
    
    Args:
        env_id: 环境ID
        num_envs: 并行环境数量
        record_video: 是否录制视频
        video_dir: 视频目录
        trajectory_name: 轨迹文件名
        **env_kwargs: 其他环境参数
    """
    
    # 创建原始环境（直接创建多环境版本）
    env = gym.make(
        env_id,
        num_envs=num_envs,  # 直接创建多环境
        obs_mode="state",
        control_mode="pd_ee_delta_pose", 
        reward_mode="dense",  # 使用dense奖励模式以支持复杂的抓取顺序奖励
        sim_backend="gpu",
        render_mode="rgb_array" if record_video else None,
        use_discrete_action=True,  # 启用离散动作 - 每个动作选择一个物体进行抓取
        use_ideal_oracle=True,  # 启用理想化神谕抓取，确保抓取成功率
        **env_kwargs
    )
    
    # 训练环境不录制视频，避免维度不匹配问题
    # 视频录制只在评估环境中进行
    
    # 添加SB3兼容性包装器
    env = SB3CompatWrapper(env)
    
    # 添加掩码提取包装器 - 设置为9个物体
    env = ExtractMaskWrapper(env, max_n=9)
    
    # 根据是否使用MaskablePPO决定是否添加动作转换包装器
    if MASKABLE_AVAILABLE:
        # 使用MaskablePPO时，保持离散动作空间，添加ActionMasker
        env = ActionMasker(env, mask_fn)
    else:
        # 使用普通PPO时，需要动作转换包装器
        env = ActionConversionWrapper(env)
    
    # 不添加Monitor包装器，因为它与多环境tensor格式不兼容
    # env = Monitor(env)
    
    # 转换为SB3向量环境
    vec_env = ManiSkillSB3VectorEnv(env)
    
    return vec_env


def create_eval_env(env_id="EnvClutter-v1", num_envs=16, record_video=False, video_dir="./videos", **env_kwargs):
    """
    创建评估环境 - 针对抓取顺序学习优化
    """
    
    env = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",  # 使用dense奖励模式
        sim_backend="gpu",  # 添加GPU后端
        render_mode="rgb_array" if record_video else None,
        use_discrete_action=True,  # 启用离散动作选择
        use_ideal_oracle=True,  # 启用理想化神谕抓取
        **env_kwargs
    )
    
    # 只有在评估环境且环境数量较少时才录制视频
    if record_video and num_envs <= 4:
        # 为评估环境使用不同的文件名
        timestamp = int(time.time())
        unique_trajectory_name = f"eval_trajectory_{timestamp}"
        
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_video=True,
            trajectory_name=unique_trajectory_name,  # 使用唯一名称
            max_steps_per_video=200,
            video_fps=30,
        )
    elif record_video and num_envs > 4:
        print(f"⚠️ 跳过视频录制：环境数量({num_envs})过多，可能导致维度不匹配")
    
    env = SB3CompatWrapper(env)
    env = ExtractMaskWrapper(env, max_n=9)  # 设置为9个物体
    
    # 根据是否使用MaskablePPO决定是否添加动作转换包装器
    if MASKABLE_AVAILABLE:
        # 使用MaskablePPO时，保持离散动作空间，添加ActionMasker
        env = ActionMasker(env, mask_fn)
    else:
        # 使用普通PPO时，需要动作转换包装器
        env = ActionConversionWrapper(env)
    
    # 不添加Monitor包装器
    # env = Monitor(env)
    
    vec_env = ManiSkillSB3VectorEnv(env)
    
    return vec_env


def train_ppo(args):
    """
    训练PPO智能体
    """
    print(f"开始训练EnvClutter环境 - 抓取顺序学习任务")
    print(f"任务目标: 通过强化学习挑选最适合的抓取顺序，每回合抓取9个物体")
    print(f"奖励优先级: 1.抓取成功 2.其他物体位移小 3.总时间短")
    print(f"并行环境数: {args.num_envs}")
    print(f"总训练步数: {args.total_timesteps}")
    print(f"使用模型: {'MaskablePPO' if MASKABLE_AVAILABLE else 'PPO'}")
    
    # 创建训练环境
    print("创建训练环境...")
    vec_env = create_env(
        env_id="EnvClutter-v1",
        num_envs=args.num_envs,
        record_video=False, # 训练环境不录制视频
        video_dir=os.path.join(args.log_dir, "eval_videos"),
        trajectory_name="train_trajectory",
    )
    
    # 创建评估环境
    print("创建评估环境...")
    eval_env = create_eval_env(
        env_id="EnvClutter-v1", 
        num_envs=args.eval_envs,
        record_video=args.record_video,
        video_dir=os.path.join(args.log_dir, "eval_videos"),
    )
    
    # 创建模型
    print("创建PPO模型...")
    
    # 模型超参数
    model_kwargs = {
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "ent_coef": args.ent_coef,
        "learning_rate": args.learning_rate,
        "clip_range": args.clip_range,
        "verbose": 1,
        "tensorboard_log": args.log_dir,
        "policy_kwargs": {
            "net_arch": dict(pi=[256, 256, 256], vf=[256, 256, 256]),
            "activation_fn": torch.nn.ReLU,
        }
    }
    
    # 选择模型类型
    if MASKABLE_AVAILABLE:
        model = MaskablePPO("MlpPolicy", vec_env, **model_kwargs)
    else:
        # 如果没有MaskablePPO，使用普通PPO
        # 注意：这种情况下掩码信息会作为普通特征输入，效果可能不如MaskablePPO
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
            name_prefix="ppo_envclutter",
        )
        callbacks.append(checkpoint_callback)
    
    # 开始训练
    print("开始训练...")
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
        model = MaskablePPO.load(args.model_path)
    else:
        model = PPO.load(args.model_path)
    
    print("开始评估...")
    
    # 运行评估
    obs = eval_env.reset()
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    current_episode_reward = 0
    current_episode_length = 0
    
    for step in range(args.eval_steps):
        # 预测动作
        action, _states = model.predict(obs, deterministic=True)
        
        # 执行动作
        obs, reward, done, info = eval_env.step(action)
        
        current_episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
        current_episode_length += 1
        
        if done[0] if isinstance(done, np.ndarray) else done:
            # Episode结束
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            
            # 检查成功率
            if isinstance(info, list) and len(info) > 0:
                success = info[0].get('success', False)
            elif isinstance(info, dict):
                success = info.get('success', False)
            else:
                success = False
            
            episode_successes.append(success)
            
            print(f"Episode完成: 奖励={current_episode_reward:.2f}, 长度={current_episode_length}, 成功={success}")
            
            # 重置计数器
            current_episode_reward = 0
            current_episode_length = 0
    
    # 打印评估结果
    if episode_rewards:
        print(f"\n评估结果 (共{len(episode_rewards)}个episode):")
        print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"平均长度: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"成功率: {np.mean(episode_successes):.2%}")
    
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description='使用SB3训练EnvClutter环境')
    
    # 基本参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], 
                       help='运行模式：训练或评估')
    parser.add_argument('--total_timesteps', type=int, default=10_000, 
                       help='总训练步数')
                       # default=1_000_000
    parser.add_argument('--num_envs', type=int, default=128, 
                       help='并行训练环境数量')
    parser.add_argument('--eval_envs', type=int, default=16, 
                       help='并行评估环境数量')
    
    # PPO超参数 - 针对抓取顺序学习任务优化
    parser.add_argument('--gamma', type=float, default=0.95, help='折扣因子 - 提高以重视长期奖励（抓取顺序）')
    parser.add_argument('--gae_lambda', type=float, default=0.9, help='GAE lambda - 降低以减少方差')
    parser.add_argument('--n_steps', type=int, default=128, help='每环境的步数 - 降低以适应短episode')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小 - 调整以匹配n_steps')
    parser.add_argument('--n_epochs', type=int, default=8, help='PPO更新轮数 - 适度降低防止过拟合')
    parser.add_argument('--ent_coef', type=float, default=0.02, help='熵系数 - 提高以鼓励探索不同抓取顺序')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='学习率 - 适度提高以加快收敛')
    parser.add_argument('--clip_range', type=float, default=0.2, help='PPO裁剪范围')
    
    # 目录和日志
    parser.add_argument('--log_dir', type=str, default='./logs/sb3_envclutter', 
                       help='日志目录')
    parser.add_argument('--model_dir', type=str, default='./models/sb3_envclutter', 
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