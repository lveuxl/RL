"""
最终训练脚本 - 完整的指标追踪
"""

import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from collections import deque
import time
import warnings
warnings.filterwarnings('ignore')

# 注册环境
from env_clutter_final import EnvClutterFinalEnv
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
import mani_skill.envs


class ComprehensiveMetricsCallback(BaseCallback):
    """综合指标追踪回调 - 记录reward、loss、success率"""
    
    def __init__(self, eval_env=None, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        
        # 追踪指标
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)
        self.training_rewards = deque(maxlen=1000)
        
        # Loss追踪
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.entropy_losses = deque(maxlen=100)
        
        # 训练统计
        self.best_mean_reward = -float('inf')
        self.episodes_count = 0
        self.success_count = 0
        
    def _on_step(self) -> bool:
        # 记录即时奖励
        if self.locals.get("rewards") is not None:
            rewards = self.locals["rewards"]
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.cpu().numpy()
            
            # 记录所有奖励
            for r in rewards:
                self.training_rewards.append(float(r))
            
            # 每100步记录平均奖励
            if self.num_timesteps % 100 == 0:
                if len(self.training_rewards) > 0:
                    mean_reward = np.mean(self.training_rewards)
                    std_reward = np.std(self.training_rewards)
                    
                    self.logger.record("train/instant_reward_mean", mean_reward)
                    self.logger.record("train/instant_reward_std", std_reward)
                    self.logger.record("train/instant_reward_max", np.max(self.training_rewards))
                    self.logger.record("train/instant_reward_min", np.min(self.training_rewards))
        
        # 检查episode结束
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            if isinstance(dones, torch.Tensor):
                dones = dones.cpu().numpy()
            
            infos = self.locals.get("infos", [])
            
            for i, done in enumerate(dones):
                if done and i < len(infos):
                    info = infos[i]
                    
                    # 记录episode奖励
                    if "episode" in info:
                        ep_reward = info["episode"].get("r", 0)
                        ep_length = info["episode"].get("l", 0)
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        self.episodes_count += 1
                    
                    # 记录成功率
                    if "success" in info:
                        success = float(info["success"])
                        self.episode_successes.append(success)
                        if success > 0:
                            self.success_count += 1
                    elif "success_rate" in info:
                        self.episode_successes.append(float(info["success_rate"]))
        
        # 每500步记录详细指标
        if self.num_timesteps % 500 == 0:
            self._log_metrics()
        
        return True
    
    def _on_rollout_end(self) -> None:
        """rollout结束时记录loss等指标"""
        
        # 尝试获取loss信息
        if hasattr(self.model, "logger") and self.model.logger is not None:
            # 从模型的logger中获取loss
            try:
                if hasattr(self.model, "_last_obs"):
                    # 获取一个批次进行前向传播以计算loss
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(self.model._last_obs).to(self.model.device)
                        values = self.model.policy.predict_values(obs_tensor)
                        if values is not None:
                            value_mean = float(values.mean().item())
                            self.logger.record("train/value_function_mean", value_mean)
            except:
                pass
        
        # 记录学习率
        if hasattr(self.model, "learning_rate"):
            if callable(self.model.learning_rate):
                current_lr = self.model.learning_rate(self.model._current_progress_remaining)
            else:
                current_lr = self.model.learning_rate
            self.logger.record("train/learning_rate", current_lr)
        
        # 记录探索率（熵系数）
        if hasattr(self.model, "ent_coef"):
            self.logger.record("train/entropy_coefficient", self.model.ent_coef)
        
        # 记录训练进度
        self.logger.record("train/progress", 1.0 - self.model._current_progress_remaining)
        self.logger.record("train/total_timesteps", self.num_timesteps)
        self.logger.record("train/episodes_count", self.episodes_count)
        
        self._log_metrics()
    
    def _log_metrics(self):
        """记录所有指标到tensorboard"""
        
        # Episode指标
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards)
            std_reward = np.std(self.episode_rewards)
            
            self.logger.record("episode/reward_mean", mean_reward)
            self.logger.record("episode/reward_std", std_reward)
            self.logger.record("episode/reward_max", np.max(self.episode_rewards))
            self.logger.record("episode/reward_min", np.min(self.episode_rewards))
            self.logger.record("episode/length_mean", np.mean(self.episode_lengths))
            
            # 检查是否有改进
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.logger.record("episode/best_mean_reward", self.best_mean_reward)
        
        # 成功率
        if len(self.episode_successes) > 0:
            success_rate = np.mean(self.episode_successes) * 100
            self.logger.record("success/rate", success_rate)
            self.logger.record("success/total_count", self.success_count)
            
            # 最近10个episode的成功率
            recent_successes = list(self.episode_successes)[-10:]
            if len(recent_successes) > 0:
                recent_success_rate = np.mean(recent_successes) * 100
                self.logger.record("success/recent_rate", recent_success_rate)
        
        # 奖励趋势
        if len(self.training_rewards) > 100:
            # 计算奖励趋势（斜率）
            x = np.arange(len(self.training_rewards))
            y = np.array(self.training_rewards)
            z = np.polyfit(x, y, 1)
            reward_trend = z[0]  # 斜率
            self.logger.record("trend/reward_slope", reward_trend)
            
            # 奖励改进率
            early_rewards = list(self.training_rewards)[:100]
            recent_rewards = list(self.training_rewards)[-100:]
            improvement = (np.mean(recent_rewards) - np.mean(early_rewards)) / (abs(np.mean(early_rewards)) + 1e-8)
            self.logger.record("trend/improvement_rate", improvement * 100)


def create_env(env_id, num_envs, seed=0):
    """创建环境"""
    env = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        sim_backend="gpu" if torch.cuda.is_available() else "cpu",
        render_mode=None,
    )
    
    # 设置种子
    env.unwrapped.seed(seed)
    
    # 包装为SB3环境
    vec_env = ManiSkillSB3VectorEnv(env)
    return vec_env


def linear_schedule(initial_value: float):
    """线性学习率调度"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def train():
    """主训练函数"""
    
    # 配置
    config = {
        "env_id": "EnvClutterFinal-v1",
        "num_envs": 64,  # 并行环境数
        "total_timesteps": 500_000,
        "seed": 42,
        
        # PPO参数 - 优化后保证收敛
        "learning_rate": linear_schedule(3e-4),  # 线性衰减
        "n_steps": 128,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": linear_schedule(0.2),
        "clip_range_vf": None,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.01,
        
        # 路径
        "log_dir": "./logs/final_training",
        "tb_log_dir": "./logs/final_training/tensorboard",
        "model_dir": "./models/final_training",
    }
    
    # 创建目录
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["tb_log_dir"], exist_ok=True)
    os.makedirs(config["model_dir"], exist_ok=True)
    
    # 设置种子
    set_random_seed(config["seed"])
    
    print("="*60)
    print("🚀 最终版训练 - 完整指标追踪")
    print("="*60)
    print(f"环境: {config['env_id']}")
    print(f"并行数: {config['num_envs']}")
    print(f"总步数: {config['total_timesteps']:,}")
    print("-"*60)
    
    # 创建训练环境
    print("创建训练环境...")
    train_env = create_env(config["env_id"], config["num_envs"], config["seed"])
    print("✓ 训练环境就绪")
    
    # 创建评估环境
    print("创建评估环境...")
    eval_env = create_env(config["env_id"], num_envs=4, seed=config["seed"]+1000)
    print("✓ 评估环境就绪")
    
    # 创建模型
    print("初始化PPO模型...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        clip_range_vf=config["clip_range_vf"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        target_kl=config["target_kl"],
        tensorboard_log=config["tb_log_dir"],
        policy_kwargs={
            "net_arch": dict(pi=[128, 128, 64], vf=[128, 128, 64]),
            "activation_fn": torch.nn.Tanh,
            "normalize_images": False,
        },
        verbose=1,
        seed=config["seed"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    print(f"✓ 模型就绪 (device: {model.device})")
    
    # 设置logger
    logger = configure(config["log_dir"], ["stdout", "tensorboard"])
    model.set_logger(logger)
    
    # 创建回调
    callbacks = []
    
    # 综合指标回调
    metrics_callback = ComprehensiveMetricsCallback(eval_env=eval_env)
    callbacks.append(metrics_callback)
    
    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config["model_dir"], "best_model"),
        log_path=os.path.join(config["log_dir"], "evaluations"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks.append(eval_callback)
    
    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=config["model_dir"],
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks.append(checkpoint_callback)
    
    # 开始训练
    print("\n" + "="*60)
    print("📊 开始训练 - 监控以下指标:")
    print("  • Reward: 应该从负值逐渐上升到正值")
    print("  • Success Rate: 应该从0%逐渐上升到90%+")
    print("  • Loss: 应该逐渐下降并稳定")
    print("="*60)
    print("\n预期里程碑:")
    print("  5,000步: 看到奖励开始上升")
    print("  20,000步: 成功率达到20%+")
    print("  50,000步: 奖励稳定为正值")
    print("  100,000步: 成功率达到50%+")
    print("  200,000步: 学会自上而下策略")
    print("  500,000步: 成功率达到90%+")
    print("-"*60 + "\n")
    
    start_time = time.time()
    
    try:
        # 训练
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            log_interval=1,
            progress_bar=True,
            reset_num_timesteps=True,
            tb_log_name="PPO_final",
        )
        
        # 保存最终模型
        final_path = os.path.join(config["model_dir"], "final_model")
        model.save(final_path)
        print(f"\n✅ 训练完成！最终模型: {final_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练中断")
        interrupted_path = os.path.join(config["model_dir"], "interrupted_model")
        model.save(interrupted_path)
        print(f"中断模型已保存: {interrupted_path}")
    
    finally:
        # 统计
        elapsed = time.time() - start_time
        print(f"\n" + "="*60)
        print("📈 训练统计:")
        print(f"  总时间: {elapsed/3600:.2f}小时")
        print(f"  完成步数: {model.num_timesteps:,}")
        print(f"  训练速度: {model.num_timesteps/elapsed:.0f} steps/秒")
        
        # 最终指标
        if len(metrics_callback.episode_rewards) > 0:
            print(f"\n📊 最终性能:")
            print(f"  平均奖励: {np.mean(metrics_callback.episode_rewards):.2f}")
            print(f"  最佳奖励: {metrics_callback.best_mean_reward:.2f}")
            
        if len(metrics_callback.episode_successes) > 0:
            final_success_rate = np.mean(list(metrics_callback.episode_successes)[-20:]) * 100
            print(f"  最终成功率: {final_success_rate:.1f}%")
        
        # 清理
        train_env.close()
        eval_env.close()
        
        print(f"\n" + "="*60)
        print("🎯 下一步:")
        print(f"1. 查看训练曲线: tensorboard --logdir {config['tb_log_dir']}")
        print(f"2. 测试模型: python test_final.py")
        print(f"3. 评估最佳模型: python evaluate_final.py {os.path.join(config['model_dir'], 'best_model', 'best_model.zip')}")
        print("="*60)


if __name__ == "__main__":
    train()