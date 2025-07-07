"""
复杂堆叠杂乱环境训练脚本
使用Stable Baselines3的PPO算法进行强化学习训练
包含课程学习、回调函数、模型保存等功能
"""

import os
import time
import argparse
from typing import Dict, Any, Optional, Callable
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, 
    CallbackList, ProgressBarCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
import wandb
from wandb.integration.sb3 import WandbCallback

from config import (
    get_env_config, get_training_config, get_ppo_config,
    get_curriculum_config, get_objects_by_difficulty, DIFFICULTY_GROUPS
)
from env_clutter import ComplexStackingClutterEnv
from utils import create_directories, save_training_info, load_training_info


class CurriculumCallback(BaseCallback):
    """课程学习回调函数"""
    
    def __init__(self, curriculum_config: Dict, verbose: int = 1):
        super().__init__(verbose)
        self.curriculum_config = curriculum_config
        self.current_stage = 0
        self.stage_start_timestep = 0
        self.stages = curriculum_config["stages"]
        self.enable_curriculum = curriculum_config["enable"]
        
    def _on_training_start(self) -> None:
        """训练开始时初始化课程学习"""
        if self.enable_curriculum:
            self._update_curriculum_stage()
            if self.verbose >= 1:
                print(f"课程学习已启动，当前阶段: {self.stages[self.current_stage]['name']}")
    
    def _on_step(self) -> bool:
        """每步检查是否需要更新课程学习阶段"""
        if not self.enable_curriculum:
            return True
            
        # 检查是否需要进入下一阶段
        current_timestep = self.num_timesteps
        stage_timesteps = current_timestep - self.stage_start_timestep
        
        if (self.current_stage < len(self.stages) - 1 and 
            stage_timesteps >= self.stages[self.current_stage]["timesteps"]):
            
            # 检查成功率是否达到阈值
            if self._check_success_threshold():
                self.current_stage += 1
                self.stage_start_timestep = current_timestep
                self._update_curriculum_stage()
                
                if self.verbose >= 1:
                    print(f"\n进入下一课程阶段: {self.stages[self.current_stage]['name']}")
                    print(f"当前时间步: {current_timestep}")
        
        return True
    
    def _check_success_threshold(self) -> bool:
        """检查当前阶段的成功率是否达到阈值"""
        # 这里可以从训练环境获取成功率
        # 简化实现，假设达到阈值
        return True
    
    def _update_curriculum_stage(self):
        """更新课程学习阶段"""
        current_stage_config = self.stages[self.current_stage]
        
        # 更新环境配置
        difficulty_groups = current_stage_config["difficulty_groups"]
        max_objects = current_stage_config["max_objects"]
        
        # 这里需要更新环境的物体选择范围
        # 实际实现中需要通过环境接口来更新
        print(f"更新课程阶段: {current_stage_config['name']}")
        print(f"难度组: {difficulty_groups}")
        print(f"最大物体数: {max_objects}")


class TrainingMetricsCallback(BaseCallback):
    """训练指标回调函数"""
    
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        """记录训练指标"""
        # 获取环境信息
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            
            # 记录episode奖励和长度
            if 'r' in ep_info:
                self.episode_rewards.append(ep_info['r'])
            if 'l' in ep_info:
                self.episode_lengths.append(ep_info['l'])
            
            # 计算移动平均
            if len(self.episode_rewards) >= 100:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                
                # 记录到日志
                self.logger.record("train/mean_reward_100", mean_reward)
                self.logger.record("train/mean_length_100", mean_length)
                
                # 更新最佳奖励
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.logger.record("train/best_mean_reward", self.best_mean_reward)
        
        return True


class ComplexStackingTrainer:
    """复杂堆叠杂乱环境训练器"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.env_config = get_env_config()
        self.training_config = get_training_config()
        self.ppo_config = get_ppo_config()
        self.curriculum_config = get_curriculum_config()
        
        # 创建必要的目录
        create_directories()
        
        # 设置随机种子
        if args.seed is not None:
            set_random_seed(args.seed)
        
        # 初始化wandb
        if args.use_wandb:
            self._init_wandb()
        
        # 创建环境
        self.env = self._create_env()
        
        # 创建模型
        self.model = self._create_model()
        
        # 创建回调函数
        self.callbacks = self._create_callbacks()
    
    def _init_wandb(self):
        """初始化wandb"""
        wandb.init(
            project="complex-stacking-clutter",
            name=f"ppo-{int(time.time())}",
            config={
                "env_config": self.env_config,
                "training_config": self.training_config,
                "ppo_config": self.ppo_config,
                "args": vars(self.args)
            },
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
    
    def _create_env(self):
        """创建训练环境"""
        def make_env(rank: int, seed: int = 0) -> Callable:
            def _init() -> gym.Env:
                env = ComplexStackingClutterEnv(
                    max_objects=self.args.max_objects,
                    reward_mode=self.args.reward_mode,
                    enable_intelligent_selection=self.args.enable_intelligent_selection,
                    render_mode=None,  # 训练时不渲染
                )
                env.seed(seed + rank)
                env = Monitor(env, f"./logs/monitor_{rank}")
                return env
            return _init
        
        # 创建向量化环境
        if self.args.num_envs > 1:
            env = SubprocVecEnv([
                make_env(i, self.args.seed) for i in range(self.args.num_envs)
            ])
        else:
            env = DummyVecEnv([make_env(0, self.args.seed)])
        
        return env
    
    def _create_model(self):
        """创建PPO模型"""
        # 配置策略网络
        policy_kwargs = dict(
            net_arch=[
                dict(
                    pi=self.env_config["network_config"]["policy_layers"],
                    vf=self.env_config["network_config"]["value_layers"]
                )
            ],
            activation_fn=torch.nn.Tanh,
            ortho_init=self.env_config["network_config"]["ortho_init"],
        )
        
        # 创建PPO模型
        model = PPO(
            policy=self.ppo_config["policy"],
            env=self.env,
            learning_rate=self.ppo_config["learning_rate"],
            n_steps=self.ppo_config["n_steps"],
            batch_size=self.ppo_config["batch_size"],
            n_epochs=self.ppo_config["n_epochs"],
            gamma=self.ppo_config["gamma"],
            gae_lambda=self.ppo_config["gae_lambda"],
            clip_range=self.ppo_config["clip_range"],
            ent_coef=self.ppo_config["ent_coef"],
            vf_coef=self.ppo_config["vf_coef"],
            max_grad_norm=self.ppo_config["max_grad_norm"],
            target_kl=self.ppo_config["target_kl"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.ppo_config["tensorboard_log"],
            verbose=self.ppo_config["verbose"],
            device=self.args.device,
        )
        
        return model
    
    def _create_callbacks(self):
        """创建回调函数列表"""
        callbacks = []
        
        # 课程学习回调
        if self.curriculum_config["enable"]:
            curriculum_callback = CurriculumCallback(
                self.curriculum_config, verbose=1
            )
            callbacks.append(curriculum_callback)
        
        # 训练指标回调
        metrics_callback = TrainingMetricsCallback(verbose=1)
        callbacks.append(metrics_callback)
        
        # 模型保存回调
        checkpoint_callback = CheckpointCallback(
            save_freq=self.env_config["logging_config"]["save_freq"],
            save_path="./models/checkpoints/",
            name_prefix="ppo_complex_stacking",
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # 评估回调
        eval_env = self._create_eval_env()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./models/",
            log_path="./logs/eval/",
            eval_freq=self.env_config["eval_config"]["eval_freq"],
            n_eval_episodes=self.env_config["eval_config"]["n_eval_episodes"],
            deterministic=self.env_config["eval_config"]["eval_deterministic"],
            render=self.env_config["eval_config"]["render_eval"],
            verbose=1
        )
        callbacks.append(eval_callback)
        
        # wandb回调
        if self.args.use_wandb:
            wandb_callback = WandbCallback(
                gradient_save_freq=1000,
                model_save_path=f"./models/wandb_{wandb.run.id}",
                verbose=2,
            )
            callbacks.append(wandb_callback)
        
        # 进度条回调
        if self.args.progress_bar:
            progress_callback = ProgressBarCallback()
            callbacks.append(progress_callback)
        
        return CallbackList(callbacks)
    
    def _create_eval_env(self):
        """创建评估环境"""
        eval_env = ComplexStackingClutterEnv(
            max_objects=self.args.max_objects,
            reward_mode=self.args.reward_mode,
            enable_intelligent_selection=self.args.enable_intelligent_selection,
            render_mode=None,
        )
        eval_env = Monitor(eval_env, "./logs/eval_monitor")
        eval_env = DummyVecEnv([lambda: eval_env])
        return eval_env
    
    def train(self):
        """开始训练"""
        print("=== 开始训练复杂堆叠杂乱环境 ===")
        print(f"总训练步数: {self.args.total_timesteps}")
        print(f"并行环境数: {self.args.num_envs}")
        print(f"设备: {self.args.device}")
        print(f"课程学习: {'启用' if self.curriculum_config['enable'] else '禁用'}")
        
        # 保存训练信息
        training_info = {
            "args": vars(self.args),
            "env_config": self.env_config,
            "training_config": self.training_config,
            "ppo_config": self.ppo_config,
            "curriculum_config": self.curriculum_config,
            "start_time": time.time(),
        }
        save_training_info(training_info, "./logs/training_info.json")
        
        # 开始训练
        start_time = time.time()
        try:
            self.model.learn(
                total_timesteps=self.args.total_timesteps,
                callback=self.callbacks,
                log_interval=self.env_config["logging_config"]["log_interval"],
                tb_log_name="ppo_complex_stacking",
                reset_num_timesteps=not self.args.continue_training,
                progress_bar=self.args.progress_bar,
            )
        except KeyboardInterrupt:
            print("\n训练被用户中断")
        
        training_time = time.time() - start_time
        print(f"\n训练完成，总用时: {training_time:.2f} 秒")
        
        # 保存最终模型
        final_model_path = f"./models/ppo_complex_stacking_final_{int(time.time())}.zip"
        self.model.save(final_model_path)
        print(f"最终模型已保存到: {final_model_path}")
        
        # 更新训练信息
        training_info.update({
            "end_time": time.time(),
            "training_time": training_time,
            "final_model_path": final_model_path,
        })
        save_training_info(training_info, "./logs/training_info.json")
        
        # 关闭wandb
        if self.args.use_wandb:
            wandb.finish()
        
        # 关闭环境
        self.env.close()
    
    def resume_training(self, model_path: str):
        """恢复训练"""
        print(f"从 {model_path} 恢复训练")
        self.model = PPO.load(model_path, env=self.env)
        self.train()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="复杂堆叠杂乱环境训练")
    
    # 基本参数
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="总训练步数")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="并行环境数量")
    parser.add_argument("--device", type=str, default="auto",
                        help="训练设备 (cpu/cuda/auto)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    
    # 环境参数
    parser.add_argument("--max-objects", type=int, default=16,
                        help="最大物体数量")
    parser.add_argument("--reward-mode", type=str, default="dense",
                        choices=["sparse", "dense"],
                        help="奖励模式")
    parser.add_argument("--enable-intelligent-selection", action="store_true",
                        help="启用智能物体选择")
    
    # 训练参数
    parser.add_argument("--continue-training", action="store_true",
                        help="继续训练")
    parser.add_argument("--model-path", type=str, default=None,
                        help="继续训练的模型路径")
    parser.add_argument("--progress-bar", action="store_true",
                        help="显示进度条")
    
    # 日志参数
    parser.add_argument("--use-wandb", action="store_true",
                        help="使用wandb记录")
    parser.add_argument("--wandb-project", type=str, default="complex-stacking-clutter",
                        help="wandb项目名称")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建训练器
    trainer = ComplexStackingTrainer(args)
    
    # 开始训练或恢复训练
    if args.continue_training and args.model_path:
        trainer.resume_training(args.model_path)
    else:
        trainer.train()


if __name__ == "__main__":
    main() 