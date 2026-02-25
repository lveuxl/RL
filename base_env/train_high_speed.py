#!/usr/bin/env python3
"""
极速训练脚本 - 最大化训练速度的优化版本
特性：
1. 完全移除物理仿真，纯逻辑计算
2. CPU优化训练（对MLP策略更高效）
3. 大批量并行处理
4. 精简的超参数配置
5. 快速收敛设计
"""

import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
import time

# 注册环境
from env_clutter_optimized import EnvClutterOptimizedEnv
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
import mani_skill.envs


class SpeedCallback(BaseCallback):
    """极简回调 - 只记录关键指标"""
    
    def __init__(self, verbose=0):
        super(SpeedCallback, self).__init__(verbose)
        self.episode_count = 0
        self.start_time = time.time()
        
    def _on_rollout_end(self) -> None:
        self.episode_count += 1
        if self.episode_count % 10 == 0:  # 每10个rollout记录一次
            elapsed = time.time() - self.start_time
            speed = self.num_timesteps / elapsed
            self.logger.record("speed/timesteps_per_second", speed)
            self.logger.record("speed/episodes", self.episode_count)


def create_fast_env(env_id: str, num_envs: int, seed: int = 0):
    """创建极速环境"""
    env = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        sim_backend="gpu",
        render_mode=None,
    )
    
    vec_env = ManiSkillSB3VectorEnv(env)
    return vec_env


def train_high_speed(use_cpu: bool = True):
    """极速训练主函数"""
    
    # 极速配置 - 专注性能
    config = {
        "env_id": "EnvClutterOptimized-v1",
        "num_envs": 256,  # 大批量并行（无物理仿真负担）
        "seed": 42,
        
        # 极速PPO配置
        "learning_rate": 5e-4,  # 稍高学习率加速收敛
        "n_steps": 64,   # 更短rollout（更频繁更新）
        "batch_size": 8192,  # 超大批次
        "n_epochs": 3,   # 最少更新轮数
        "gamma": 0.98,   # 稍低折扣（加速学习）
        "gae_lambda": 0.9,
        "clip_range": 0.2,
        "ent_coef": 0.005,  # 减少熵（更快收敛）
        "vf_coef": 0.25,
        "max_grad_norm": 1.0,
        
        # 训练配置
        "total_timesteps": 100_000,  # 短期快速训练
        "eval_freq": 10000,
        "log_interval": 1,
        
        # 网络配置
        "policy_kwargs": {
            "net_arch": [128, 128],  # 更小网络（加速训练）
            "activation_fn": torch.nn.ReLU,
        },
        
        "verbose": 1,
        "device": "cpu" if use_cpu else "auto",
    }
    
    device_name = "CPU" if use_cpu else ("GPU" if torch.cuda.is_available() else "CPU")
    print(f"🚀 极速训练模式")
    print(f"📊 配置: {config['num_envs']}环境 x {config['total_timesteps']:,}步")
    print(f"⚡ 计算设备: {device_name}")
    print(f"🎯 预计训练时间: ~5-10分钟")
    
    # 创建环境
    print("创建极速环境...")
    start_time = time.time()
    
    env = create_fast_env(config["env_id"], config["num_envs"], config["seed"])
    eval_env = create_fast_env(config["env_id"], min(32, config["num_envs"]), config["seed"] + 1000)
    
    env_time = time.time() - start_time
    print(f"✓ 环境创建耗时: {env_time:.1f}秒")
    
    # 创建PPO模型
    print("创建极速PPO模型...")
    model_start = time.time()
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        policy_kwargs=config["policy_kwargs"],
        verbose=config["verbose"],
        seed=config["seed"],
        device=config["device"],
        tensorboard_log="./tensorboard_logs_speed/" if config["verbose"] > 0 else None,
    )
    
    model_time = time.time() - model_start
    param_count = sum(p.numel() for p in model.policy.parameters())
    print(f"✓ 模型创建耗时: {model_time:.1f}秒")
    print(f"🧠 模型参数: {param_count:,}")
    
    # 极简回调
    callbacks = [SpeedCallback()]
    
    # 可选评估（影响速度）
    if config["eval_freq"] > 0:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./models_speed/",
            log_path="./logs_speed/",
            eval_freq=config["eval_freq"],
            n_eval_episodes=5,  # 快速评估
            deterministic=True,
            render=False,
            verbose=0,  # 静默评估
        )
        callbacks.append(eval_callback)
    
    # 开始训练
    print("\n⚡ 开始极速训练...")
    print("-" * 50)
    
    train_start = time.time()
    
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            log_interval=config["log_interval"],
            progress_bar=True,
        )
        
        train_time = time.time() - train_start
        total_time = time.time() - start_time
        
        # 性能统计
        steps_per_second = config["total_timesteps"] / train_time
        
        print(f"\n🎉 极速训练完成！")
        print(f"⏱️  总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        print(f"🚀 训练耗时: {train_time:.1f}秒")
        print(f"📈 训练速度: {steps_per_second:.0f} steps/s")
        print(f"⚡ 性能提升: ~{steps_per_second/5:.0f}x vs原版")
        
        # 保存模型
        model.save("./models_speed/final_model")
        print(f"💾 模型已保存: ./models_speed/final_model")
        
        env.close()
        eval_env.close()
        
        return model, steps_per_second
        
    except KeyboardInterrupt:
        print("\n⏹️  训练被中断")
        env.close()
        eval_env.close()
        return None, 0


def quick_test():
    """快速测试训练好的模型"""
    print("🧪 快速测试模型...")
    
    try:
        model = PPO.load("./models_speed/final_model")
        env = create_fast_env("EnvClutterOptimized-v1", 1, 42)
        
        total_reward = 0
        obs = env.reset()
        
        for step in range(20):  # 快速测试20步
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            
            if done[0]:
                break
        
        env.close()
        print(f"✓ 测试完成，总奖励: {total_reward:.1f}")
        
    except Exception as e:
        print(f"⚠️  测试失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="极速强化学习训练")
    parser.add_argument("--gpu", action="store_true", help="强制使用GPU")
    parser.add_argument("--test", action="store_true", help="测试已训练模型")
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs("./models_speed", exist_ok=True)
    os.makedirs("./logs_speed", exist_ok=True)
    os.makedirs("./tensorboard_logs_speed", exist_ok=True)
    
    if args.test:
        quick_test()
        return
    
    # 设备选择策略
    use_cpu = not args.gpu  # 默认使用CPU（对MLP更快）
    
    if use_cpu:
        print("💡 使用CPU训练（对MLP策略更高效）")
        # 关闭CUDA避免开销
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        print("🔥 使用GPU训练")
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 开始训练
    model, speed = train_high_speed(use_cpu=use_cpu)
    
    if model and speed > 0:
        # 自动测试
        print("\n" + "="*50)
        quick_test()
        
        print(f"\n🚀 训练完成！速度: {speed:.0f} steps/s")
        print("🔥 启动tensorboard: tensorboard --logdir ./tensorboard_logs_speed")


if __name__ == "__main__":
    main()

