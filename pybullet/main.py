import pybullet as p
import pybullet_data
import argparse
import os
import time
import random
import csv
import math
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Lock
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import gym
import queue
import json
import threading
from collections import deque
import gym.spaces as spaces

from config import ENV_CONFIG
from pick_place_env import PickAndPlaceEnv, RewardMetricsCallback
from utils import random_objects, pick
from simulation import setup_scene, generate_objects

def make_env(rank, seed=0):
    """
    创建单个环境的辅助函数
    """
    def _init():
        env = PickAndPlaceEnv()
        env.seed(seed + rank)  # 设置不同的随机种子
        return env
    return _init

class ImprovedAsyncEnv:
    """异步环境类，解决数据混乱和同步问题"""
    
    def __init__(self, env_id, shared_action_queue, shared_result_queue, 
                 ready_event, stop_event, info_queue):
        self.env_id = env_id
        self.shared_action_queue = shared_action_queue
        self.shared_result_queue = shared_result_queue
        self.ready_event = ready_event
        self.stop_event = stop_event
        self.info_queue = info_queue
        
        # 本地状态缓存
        self.last_obs = None
        self.last_info = {}
        self.pending_action = None
        
        # 启动环境进程
        self.process = Process(target=self._run_env, daemon=True)
        self.process.start()
        
        # 等待环境初始化完成
        self._wait_for_initialization()
    
    def _wait_for_initialization(self):
        """等待环境初始化完成"""
        timeout = 30  # 30秒超时
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = self.shared_result_queue.get(timeout=0.1)
                if result[0] == self.env_id and result[1] == "INIT_COMPLETE":
                    self.last_obs = result[2]
                    print(f"环境 {self.env_id} 初始化完成")
                    return
                else:
                    # 不是我们的数据，放回队列
                    self.shared_result_queue.put(result)
            except queue.Empty:
                continue
        
        raise RuntimeError(f"环境 {self.env_id} 初始化超时")
    
    def _run_env(self):
        """环境进程主循环"""
        try:
            print(f"启动环境进程 {self.env_id}")
            
            # 修改环境创建逻辑 - 仿照main_副本2.py的渲染方式
            render_mode = self.env_id == 0  # 只有环境0使用GUI渲染
            
            # 设置环境变量来控制GUI连接
            if render_mode:
                os.environ['PYBULLET_ALLOW_GUI'] = '1'
                print(f"环境 {self.env_id}: 允许GUI连接")
            else:
                os.environ['PYBULLET_ALLOW_GUI'] = '0'
                print(f"环境 {self.env_id}: 禁止GUI连接，强制DIRECT模式")
            
            # 创建环境时添加额外处理以避免GUI冲突
            if render_mode:
                # 检查是否已有GUI连接
                try:
                    # 确保没有已存在的GUI连接
                    try:
                        # 尝试断开任何现有连接
                        for i in range(10):
                            try:
                                p.disconnect(i)
                            except:
                                pass
                    except:
                        pass
                except:
                    pass
            else:
                # 非渲染环境，确保断开任何可能的连接
                try:
                    for i in range(10):
                        try:
                            p.disconnect(i)
                        except:
                            pass
                except:
                    pass
            
            # 创建环境，根据env_id决定是否渲染
            env = PickAndPlaceEnv(render=render_mode)
            env.seed(self.env_id * 1000)
            
            # 初始化环境
            obs_result = env.reset()
            if isinstance(obs_result, tuple):
                obs = obs_result[0]
            else:
                obs = obs_result
                
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)
            obs = obs.flatten()
            
            # 发送初始化完成信号
            self.shared_result_queue.put((self.env_id, "INIT_COMPLETE", obs))
            
            step_count = 0
            episode_count = 0
            
            while not self.stop_event.is_set():
                try:
                    # 检查是否有动作
                    action_data = self.shared_action_queue.get(timeout=0.1)
                    
                    if action_data is None or action_data[0] != self.env_id:
                        # 不是给我们的动作，放回队列
                        if action_data is not None:
                            self.shared_action_queue.put(action_data)
                        continue
                    
                    action = action_data[1]
                    step_count += 1
                    
                    # 执行动作
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        new_obs, reward, done, _, info = step_result
                    else:
                        new_obs, reward, done, info = step_result
                    
                    if not isinstance(new_obs, np.ndarray):
                        new_obs = np.array(new_obs, dtype=np.float32)
                    new_obs = new_obs.flatten()
                    
                    # 发送info数据
                    info_data = {
                        'episode_reward': env.current_episode_reward,
                        'success_count': env.success_count,
                        'total_distance': env.total_distance,
                        'is_episode_done': done
                    }
                    self.info_queue.put((self.env_id, info_data))
                    
                    # 如果回合结束，重置环境
                    if done:
                        episode_count += 1
                        print(f"环境 {self.env_id} 第 {episode_count} 个episode结束")
                        
                        reset_result = env.reset()
                        if isinstance(reset_result, tuple):
                            new_obs = reset_result[0]
                        else:
                            new_obs = reset_result
                            
                        if not isinstance(new_obs, np.ndarray):
                            new_obs = np.array(new_obs, dtype=np.float32)
                        new_obs = new_obs.flatten()
                    
                    # 发送结果
                    self.shared_result_queue.put((
                        self.env_id, "STEP_RESULT", 
                        new_obs, reward, done, info
                    ))
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"环境 {self.env_id} 发生错误: {e}")
                    
                    # 添加恢复机制 
                    if "GUI connection" in str(e) or "Only one local" in str(e):
                        print(f"环境 {self.env_id} 尝试以DIRECT模式恢复")
                        try:
                            # 强制切换到DIRECT模式
                            env.close()
                            os.environ['PYBULLET_ALLOW_GUI'] = '0'  # 强制禁用GUI
                            env = PickAndPlaceEnv(render=False)
                            env.seed(self.env_id * 1000)
                            
                            # 重置环境
                            reset_result = env.reset()
                            if isinstance(reset_result, tuple):
                                new_obs = reset_result[0]
                            else:
                                new_obs = reset_result
                                
                            if not isinstance(new_obs, np.ndarray):
                                new_obs = np.array(new_obs, dtype=np.float32)
                            new_obs = new_obs.flatten()
                            
                            # 发送恢复后的观察
                            self.shared_result_queue.put((
                                self.env_id, "STEP_RESULT", 
                                new_obs, 0.0, False, {'recovered': True}
                            ))
                            continue
                        except Exception as reset_e:
                            print(f"环境 {self.env_id} 恢复失败: {reset_e}")
                    
                    # 发送错误信号
                    self.shared_result_queue.put((
                        self.env_id, "ERROR", str(e)
                    ))
                    break
            
            env.close()
            print(f"环境 {self.env_id} 已关闭")
            
        except Exception as e:
            print(f"环境 {self.env_id} 初始化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def step(self, action):
        """执行动作并获取结果"""
        # 发送动作
        self.shared_action_queue.put((self.env_id, action))
        
        # 等待结果
        timeout = 2.0  # 超时时间
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = self.shared_result_queue.get(timeout=0.01)
                
                if result[0] == self.env_id and result[1] == "STEP_RESULT":
                    obs, reward, done, info = result[2], result[3], result[4], result[5]
                    self.last_obs = obs
                    self.last_info = info
                    return obs, reward, done, info
                else:
                    # 不是我们的数据，放回队列
                    self.shared_result_queue.put(result)
                    
            except queue.Empty:
                time.sleep(0.001)
                continue
        
        # 超时处理
        print(f"环境 {self.env_id} 步骤超时")
        return self.last_obs, 0.0, False, {'timeout': True, 'env_id': self.env_id}
    
    def reset(self):
        """重置环境"""
        return self.last_obs
    
    def close(self):
        """关闭环境"""
        self.stop_event.set()
        if self.process.is_alive():
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
    
    def wait_for_ready(self):
        """等待环境准备就绪"""
        # 环境在初始化时已经等待完成，所以这里直接返回
        return True

class ImprovedAsyncEnvWrapper(gym.Env):
    """环境包装器"""
    
    def __init__(self, env_id, shared_action_queue, shared_result_queue, 
                 ready_event, stop_event, info_queue):
        super().__init__()
        self.async_env = ImprovedAsyncEnv(
            env_id, shared_action_queue, shared_result_queue,
            ready_event, stop_event, info_queue
        )
        
        # 直接定义动作和观察空间，避免创建dummy_env
        self.action_space = spaces.Discrete(16)  # 16个可能的动作（选择物体）
        
        # 观察空间计算：
        # - 每个物体：4个位置特征 + 4个类别特征 + 1个暴露度 + 1个失败次数 = 10个特征
        # - 16个物体：16 × 10 = 160个特征  
        # - 全局特征：3个（步数进度、剩余物体数、成功率）
        # - 总计：160 + 3 = 163个特征
        features_per_obj = 4 + 4 + 1 + 1  # 位置(4) + 类别(4) + 暴露度(1) + 失败次数(1) = 10
        global_features = 3
        obs_dim = 16 * features_per_obj + global_features  # 16*10+3 = 163
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        print(f"预期观察空间大小: {obs_dim}")
        print(f"注意：如果category_map发生变化，需要更新特征数量计算")
    
    def step(self, action):
        return self.async_env.step(action)
    
    def reset(self):
        return self.async_env.reset()
    
    def close(self):
        self.async_env.close()
    
    def wait_for_ready(self):
        """等待异步环境准备就绪"""
        return self.async_env.wait_for_ready()

def improved_learner(num_envs, checkpoint_dir, tb_log_dir, model_path, info_queue):
    """
    学习器函数，管理多个异步环境
    
    Args:
        num_envs: 环境数量
        checkpoint_dir: 检查点目录
        tb_log_dir: TensorBoard日志目录
        model_path: 模型保存路径
        info_queue: 信息队列，用于向主进程发送训练统计信息
    """
    print("开始启动改进的学习器进程...")
    print(f"渲染配置: 环境0使用GUI渲染，环境1-{num_envs-1}使用DIRECT模式")
    
    # 创建共享队列
    shared_action_queue = Queue()
    shared_result_queue = Queue()
    ready_event = Event()
    stop_event = Event()
    
    # 创建环境包装器
    env_wrappers = []
    for i in range(num_envs):
        wrapper = ImprovedAsyncEnvWrapper(
            env_id=i,
            shared_action_queue=shared_action_queue,
            shared_result_queue=shared_result_queue,
            ready_event=ready_event,
            stop_event=stop_event,
            info_queue=info_queue
        )
        env_wrappers.append(wrapper)
        print(f"环境 {i} 创建完成 {'(GUI渲染)' if i == 0 else '(DIRECT模式)'}")
    
    # 等待所有环境准备就绪
    print("等待所有环境准备就绪...")
    for wrapper in env_wrappers:
        wrapper.wait_for_ready()
    print("所有环境已准备就绪")
    
    try:
        # 从tb_log_dir中提取current_run
        import os
        current_run = 1
        try:
            # 从tb_log_dir路径中提取运行序号
            # tb_log_dir格式类似: "ppo_model/tensorboard/run_1"
            if "run_" in tb_log_dir:
                current_run = int(tb_log_dir.split("run_")[-1])
        except:
            current_run = 1
        
        # 向info_queue发送当前运行序号信息
        info_queue.put((0, {"current_run": current_run}))
        
        # 创建向量化环境
        vec_env = DummyVecEnv([lambda i=i: env_wrappers[i] for i in range(num_envs)])
        
        # 定义模型超参数 - 优化版本
        ppo_params = {
            "policy": "MlpPolicy",
            "n_steps": 256,        # 增加步数，获得更多经验
            "batch_size": 128,     # 增加批次大小，提高训练稳定性
            "learning_rate": 5e-4, # 适中的学习率，平衡收敛速度和稳定性
            "gamma": 0.995,        # 稍高的折扣因子，重视长期奖励
            "n_epochs": 8,         # 适中的训练轮数
            "clip_range": 0.15,    # 稍小的剪切范围，提高稳定性
            "ent_coef": 0.01,      # 适度的熵系数，平衡探索和利用
            "vf_coef": 0.25,       # 降低价值函数系数
            "max_grad_norm": 0.3,  # 梯度剪切
            "gae_lambda": 0.95,    # GAE lambda参数
            "verbose": 1,
            "tensorboard_log": tb_log_dir,
            # 添加网络架构配置
            "policy_kwargs": {
                "net_arch": [128, 128, 64],  # 简化网络结构
                "activation_fn": "tanh",      # 使用tanh激活函数
            }
        }
        
        # 创建PPO模型
        model = PPO(
            ppo_params["policy"], 
            vec_env, 
            verbose=ppo_params["verbose"], 
            tensorboard_log=ppo_params["tensorboard_log"],
            n_steps=ppo_params["n_steps"],
            batch_size=ppo_params["batch_size"],
            learning_rate=ppo_params["learning_rate"],
            gamma=ppo_params["gamma"],
            n_epochs=ppo_params["n_epochs"],
            clip_range=ppo_params["clip_range"],
            ent_coef=ppo_params["ent_coef"],
            vf_coef=ppo_params["vf_coef"],
            max_grad_norm=ppo_params["max_grad_norm"],
            gae_lambda=ppo_params["gae_lambda"],
        )
        
        # 保存超参数到log/hyperparameters文件夹
        save_hyperparameters(ppo_params, current_run)
        
        # 设置检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=2000,         # 确保多次保存
            save_path=checkpoint_dir,
            name_prefix="ppo_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        
        # 在model.learn之前添加
        print(f"开始训练，总步数设置为: {50000}，并行环境数: {num_envs}")
        print(f"每{2000}步保存一次checkpoint到: {checkpoint_dir}")
        
        # 创建自定义回调实例
        metrics_callback = RewardMetricsCallback(verbose=1, flush_freq=100)  # 降低flush频率，更频繁刷新
        
        # 添加增强的TensorBoard记录回调
        class EnhancedTensorBoardCallback(BaseCallback):
            def __init__(self, verbose=0):
                super(EnhancedTensorBoardCallback, self).__init__(verbose)
                self.episode_count = 0
                self.step_count = 0
                
            def _on_step(self):
                self.step_count += 1
                
                # 记录基础训练指标
                if len(self.locals['rewards']) > 0:
                    current_reward = self.locals['rewards'][0]
                    self.logger.record('train/step_reward', current_reward)
                
                # 记录学习率
                if hasattr(self.model, 'learning_rate'):
                    if callable(self.model.learning_rate):
                        lr = self.model.learning_rate(1.0)  # 传入进度参数
                    else:
                        lr = self.model.learning_rate
                    self.logger.record('train/learning_rate', lr)
                
                # 记录策略损失和价值损失（如果可用）
                if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                    for key, value in self.model.logger.name_to_value.items():
                        if 'loss' in key.lower():
                            self.logger.record(f'train/{key}', value)
                
                # 处理环境信息
                if self.locals.get('infos'):
                    for i, info in enumerate(self.locals['infos']):
                        if info:
                            # 确保success相关数据被记录
                            if 'success' in info:
                                self.logger.record('success/step_success_rate', float(info['success']))
                            
                            if 'success_count' in info:
                                self.logger.record('success/cumulative_successes', info['success_count'])
                            
                            if 'remaining_objects' in info:
                                self.logger.record('environment/objects_remaining', info['remaining_objects'])
                            
                            # 只处理第一个环境的数据
                            break
                
                # 检查回合结束
                dones = self.locals.get('dones')
                if dones is not None and np.any(dones):
                    self.episode_count += 1
                    self.logger.record('training/episodes_completed', self.episode_count)
                
                # 每50步强制刷新一次
                if self.step_count % 50 == 0:
                    self.logger.dump(self.step_count)
                
                return True
        
        enhanced_tb_callback = EnhancedTensorBoardCallback()
        
        class ProgressCallback(BaseCallback):
            def __init__(self, verbose=0):
                super(ProgressCallback, self).__init__(verbose)
                self.step_count = 0
            
            def _on_step(self):
                self.step_count += 1
                if self.step_count % 100 == 0:  # 每100步打印一次
                    print(f"已完成 {self.step_count} 步训练 (目标: 4)")
                return True
        
        progress_callback = ProgressCallback()
        
        # 修改SaveModelCallback以同步保存TensorBoard数据
        class SaveModelCallback(BaseCallback):
            def __init__(self, save_path, verbose=0, tb_log_dir=None, save_freq=100, current_run=1):
                super(SaveModelCallback, self).__init__(verbose)
                self.save_path = save_path
                self.step_count = 0
                self.tb_log_dir = tb_log_dir
                self.save_freq = save_freq
                self.current_run = current_run
            
            def _on_step(self):
                self.step_count += 1
                
                # 记录每步的基本信息到TensorBoard
                try:
                    self.model.logger.record("training/global_step", self.step_count)
                    # Dump每一步，确保数据进入内存
                    self.model.logger.dump(self.step_count)
                except Exception as e:
                    print(f"记录步数信息时出错: {e}")
                
                # 每save_freq步保存一次模型和额外数据
                if self.step_count % self.save_freq == 0:
                    # 保存模型
                    try:
                        save_path = f"{self.save_path}_step_{self.step_count}"
                        self.model.save(save_path)
                        print(f"手动保存模型到: {save_path}")
                        
                        # 在保存点添加额外的TensorBoard数据点
                        self.model.logger.record("checkpoints/saved_model", 1)
                        self.model.logger.record("checkpoints/step", self.step_count)
                        self.model.logger.dump(self.step_count)
                        
                        # 将当前运行序号添加到info_queue
                        try:
                            info_queue.put((0, {"current_run": self.current_run}))
                        except:
                            pass
                        
                        # 尝试强制刷新TensorBoard
                        self._flush_tensorboard()
                    except Exception as e:
                        print(f"保存模型或TensorBoard检查点时出错: {e}")
                
                return True
            
            def _flush_tensorboard(self):
                """强制刷新TensorBoard数据到磁盘"""
                try:
                    if hasattr(self.model.logger, 'writer') and self.model.logger.writer is not None:
                        self.model.logger.writer.flush()
                except:
                    pass
                
                try:
                    # 创建一个小文件触发文件系统刷新
                    import os
                    if self.tb_log_dir and os.path.exists(self.tb_log_dir):
                        touch_file = os.path.join(self.tb_log_dir, f"save_{self.step_count}.tmp")
                        with open(touch_file, "w") as f:
                            f.write(f"Save at step {self.step_count}")
                        os.remove(touch_file)
                except:
                    pass
        
        save_callback = SaveModelCallback(model_path, tb_log_dir=tb_log_dir, save_freq=100, current_run=current_run)
        
        # 增加步数监控回调
        class StepProgressCallback(BaseCallback):
            def __init__(self, verbose=0, total_steps=2048, tb_logger=None):
                super(StepProgressCallback, self).__init__(verbose)
                self.total_steps = total_steps
                self.tb_logger = tb_logger
                self.last_reported = -1
            
            def _on_step(self):
                # 每10步报告一次进度
                if self.n_calls % 10 == 0 and self.n_calls != self.last_reported:
                    progress = (self.n_calls / self.total_steps) * 100
                    print(f"训练进度: {self.n_calls}/{self.total_steps} 步 ({progress:.1f}%)")
                    self.last_reported = self.n_calls
                    
                    # 记录到自定义logger
                    if self.tb_logger:
                        self.tb_logger.log_step(self.n_calls, {
                            "training/progress": progress,
                            "training/step": self.n_calls
                        })
                return True
        
        # 更新回调列表
        callbacks = CallbackList([
            checkpoint_callback, 
            progress_callback, 
            save_callback, 
            metrics_callback,
            enhanced_tb_callback
        ])

        # 训练循环前记录初始步骤
        print("\n" + "="*50)
        print(f"开始训练，总步数: {50000}，实际设置: {50000}，并行环境: {num_envs}")
        print(f"TensorBoard日志保存到: {tb_log_dir}")
        print("="*50 + "\n")

        
        # 训练模型
        try:
            # 设置自定义logger直接指向目标目录
            from stable_baselines3.common.logger import configure
            custom_logger = configure(tb_log_dir, ["stdout", "tensorboard"])
            model.set_logger(custom_logger)
            
            # 训练
            model.learn(total_timesteps=50000, callback=callbacks, progress_bar=True)
            
            print("\n" + "="*50)
            print(f"训练完成! 最终步数: {model.num_timesteps}")
            print("="*50 + "\n")
            
            # 确保最终模型被保存
            model.save(model_path)
            print(f"最终模型已保存到: {model_path}")
            
            # 最后一次刷新TensorBoard
            try:
                model.logger.record("training/finished", 1.0)
                model.logger.dump(model.num_timesteps)
            
                
                print("已确保TensorBoard日志完全写入")
                
                # 检查TensorBoard日志文件是否存在
                import glob
                log_files = glob.glob(f"{tb_log_dir}/*tfevents*")
                print(f"在 {tb_log_dir} 中找到 {len(log_files)} 个TensorBoard日志文件:")
                for log_file in log_files:
                    print(f"  - {os.path.basename(log_file)}")
                    
            except Exception as e:
                print(f"写入最终TensorBoard日志时出错: {e}")
            
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 尝试保存当前模型
            try:
                model.save(f"{model_path}_interrupted")
                print(f"中断的模型已保存到: {model_path}_interrupted")
            except:
                print("无法保存中断的模型")
        
        # 关闭环境
        vec_env.close()
        
        
    except Exception as e:
        print(f"改进的学习器进程发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 发送训练结束信号给logger进程
        try:
            info_queue.put(("END_TRAINING", {"is_training_done": True}))
            print("已发送训练结束信号给logger进程")
        except Exception as e:
            print(f"发送训练结束信号时出错: {e}")
        
        stop_event.set()
        for wrapper in env_wrappers:
            wrapper.close()

def save_hyperparameters(params, run_number):
    """保存超参数到log/hyperparameters文件夹"""
    # 确保目录存在
    hyperparams_dir = os.path.join('log', 'hyperparameters')
    os.makedirs(hyperparams_dir, exist_ok=True)
    
    # 创建文件名
    filename = os.path.join(hyperparams_dir, f'hyperparameters_{run_number}.json')
    
    # 添加环境和训练相关参数
    params.update({
        "total_timesteps": 50000,
        "num_envs": 4,  
        "max_steps_per_episode": 16,
        "env_reward_config": {
            "success_reward": 10.0,
            "failure_penalty": -2.0,
            "displacement_penalty_factor": -0.5,
            "time_penalty": -0.1
        }
    })
    
    # 保存为JSON文件
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"超参数已保存到: {filename}")
    
    # 同时保存为CSV便于查看
    csv_filename = os.path.join(hyperparams_dir, f'hyperparameters_{run_number}.csv')
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

def logger(info_queue, results_queue=None):
    """
    日志记录进程：记录训练统计信息
    """
    episode_rewards = []
    success_counts = []
    distances = []
    episodes = 0
    
    # 为每个环境单独记录episode数据
    env_episodes = {0: 0, 1: 0, 2: 0, 3: 0}  # 4个环境的episode计数
    total_steps_received = 0  # 记录接收到的总步数
    
    # 存储结果以便返回
    collected_results = {
        "objnumlist": [],
        "sucktimelist": [],
        "successobjlist": [],
        "failobjlist": [],
        "remainobjlist": [],
        "successratelist": [],
        "totaldistancelist": [],
        "averagedistancelist": [],
        "timelist": []
    }
    
    try:
        print("日志记录进程启动")
        start_time = time.time()
        
        while True:
            try:
                result = info_queue.get(timeout=60)
                
                # 检查结果格式
                if isinstance(result, tuple) and len(result) == 2:
                    env_id, info = result
                else:
                    continue
                
                # 检查是否是训练结束信号
                if env_id == "END_TRAINING" and info.get('is_training_done', False):
                    print("日志进程收到训练结束信号，准备发送结果")
                    break
                
                # 记录接收到的步数
                total_steps_received += 1
                
                # 检查这是否是回合结束的信息
                is_episode_done = info.get('is_episode_done', False)
                
                # 如果回合结束，则记录完整的回合数据
                if is_episode_done:
                    episode_rewards.append(info['episode_reward'])
                    success_counts.append(info['success_count'])
                    distances.append(info['total_distance'])
                    episodes += 1
                    
                    # 更新对应环境的episode计数
                    if env_id in env_episodes:
                        env_episodes[env_id] += 1
                    
                    # 每次生成新的时间戳，而不是使用start_time
                    current_time = time.time() - start_time
                    
                    # 计算训练统计
                    recent_rewards = episode_rewards[-min(10, len(episode_rewards)):]
                    recent_successes = success_counts[-min(10, len(success_counts)):]
                    recent_distances = distances[-min(10, len(distances)):]
                    
                    # 每个回合打印统计信息
                    print(f"\n=== 训练统计 [总回合 {episodes}] ===")
                    print(f"环境 {env_id} 完成第 {env_episodes.get(env_id, 0)} 个episode")
                    print(f"各环境episode数: {env_episodes}")
                    print(f"训练时间: {current_time:.2f} 秒")
                    print(f"总接收步数: {total_steps_received}")
                    print(f"平均奖励: {np.mean(recent_rewards):.2f}")
                    print(f"平均成功抓取: {np.mean(recent_successes):.2f}")
                    print(f"平均位移: {np.mean(recent_distances):.2f}")
                    
                    # 获取当前运行的序号 (从info_queue中获取)
                    current_run = info.get('current_run', 1)
                    
                    # 每10个回合保存一次CSV，并包含当前时间戳
                    if episodes % 10 == 0:
                        import pandas as pd
                        data = {
                            'episode': list(range(1, len(episode_rewards) + 1)),
                            'reward': episode_rewards,
                            'success_count': success_counts,
                            'total_distance': distances,
                            'time': [current_time] * len(episode_rewards)  # 为每条记录添加当前时间戳
                        }
                        df = pd.DataFrame(data)
                        os.makedirs('log', exist_ok=True)  # 使用log文件夹
                        df.to_csv(f'log/training_stats_{current_run}.csv', index=False)  # 添加运行序号
                        print(f"已保存中间统计数据，共 {episodes} 个episodes")
                
                # 无论是否回合结束，都显示实时统计（减少输出频率）
                if env_id == 0 and total_steps_received % 100 == 0:  # 每100步显示一次
                    print(f"\r步数 {total_steps_received} - 环境 {env_id} - 当前奖励: {info['episode_reward']:.2f}, 成功: {info['success_count']}, 位移: {info['total_distance']:.2f}", end="")
                    
            except queue.Empty:
                # 超时检查 - 如果长时间没有数据，可能训练已结束
                print(f"\n日志进程：长时间未收到数据，当前已记录 {episodes} 个episodes，总步数 {total_steps_received}")
                print(f"各环境episode数: {env_episodes}")
                continue
            except Exception as e:
                print(f"日志处理错误: {e}")
                import traceback
                traceback.print_exc()
                break
                
    except KeyboardInterrupt:
        print("日志记录进程收到中断信号")
    
    # 确保最终数据保存
    print(f"\n日志进程结束，共记录 {episodes} 个episodes，总步数 {total_steps_received}")
    print(f"各环境episode数: {env_episodes}")
    
    # 最终保存统计数据
    try:
        import pandas as pd
        
        # 确保最终记录包含时间信息
        current_time = time.time() - start_time
        times = [current_time] * len(episode_rewards)
        
        # 获取当前运行的序号 (设置为默认值，info中可能没有此值)
        current_run = 1
        if 'info' in locals() and isinstance(info, dict):
            current_run = info.get('current_run', 1)
        
        data = {
            'episode': list(range(1, len(episode_rewards) + 1)),
            'reward': episode_rewards,
            'success_count': success_counts,
            'total_distance': distances,
            'time': times  # 添加时间戳
        }
        df = pd.DataFrame(data)
        os.makedirs('log', exist_ok=True)  # 使用log文件夹
        df.to_csv(f'log/final_training_stats_{current_run}.csv', index=False)  # 添加运行序号
        
        # 保存详细的调试信息
        debug_info = {
            'total_episodes': episodes,
            'total_steps_received': total_steps_received,
            'env_episodes': env_episodes,
            'training_time': current_time,
            'expected_episodes_per_env': 50000 // 4 // 16,  # 总步数 / 环境数 / 每episode步数
            'expected_total_episodes': (50000 // 4 // 16) * 4
        }
        
        with open(f'log/debug_info_{current_run}.json', 'w') as f:
            import json
            json.dump(debug_info, f, indent=2)
        
        print(f"已保存最终统计数据和调试信息")
        
    except Exception as e:
        print(f"保存最终统计数据时出错: {e}")
        
    print("日志记录进程结束")

    # 在函数结束前，确保结果被发送到队列
    if results_queue is not None:
        try:
            # 填充结果数据
            for episode_idx in range(len(episode_rewards)):
                collected_results["objnumlist"].append(success_counts[episode_idx] + 4)  # 假设初始有4个物体
                collected_results["sucktimelist"].append(16)  # 使用实际的max_steps
                collected_results["successobjlist"].append(success_counts[episode_idx])
                collected_results["failobjlist"].append(16 - success_counts[episode_idx])
                collected_results["remainobjlist"].append(4 - success_counts[episode_idx])
                collected_results["successratelist"].append(success_counts[episode_idx] / 16 if success_counts[episode_idx] > 0 else 0)
                collected_results["totaldistancelist"].append(distances[episode_idx])
                collected_results["averagedistancelist"].append(distances[episode_idx] / 16)
                collected_results["timelist"].append(current_time / episodes if episodes > 0 else current_time)  # 平均每episode时间
            
            # 如果没有数据，至少创建一条默认记录
            if len(episode_rewards) == 0:
                collected_results["objnumlist"].append(4)  # 假设初始有4个物体
                collected_results["sucktimelist"].append(0)
                collected_results["successobjlist"].append(0)
                collected_results["failobjlist"].append(0)
                collected_results["remainobjlist"].append(4)
                collected_results["successratelist"].append(0)
                collected_results["totaldistancelist"].append(0)
                collected_results["averagedistancelist"].append(0)
                collected_results["timelist"].append(time.time() - start_time)
                print("日志进程：没有回合数据，创建默认记录")
            
            # 发送结果并确保成功
            results_queue.put(collected_results)
            print(f"日志进程：已发送结果，包含 {len(collected_results['objnumlist'])} 条记录")
        except Exception as e:
            print(f"结果队列发送失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['height', 'reverse', 'random', 'forward', 'distance', 'ppo'], 
                       default='ppo', help='选择一个模式')
    args = parser.parse_args()
    
    # 转变视角
    width = 1408  # 图像宽度
    height = 1024   # 图像高度
    fov = 50  # 相机视角
    aspect = width / height  # 宽高比
    near = 0.01
    far = 20
    cam_distance = 0.58
    cam_yaw = -90
    cam_pitch = -89
    cam_roll = 0
    cam_targetPos = [1.0 , -0.2 ,1.3]
    cam_up_axis_idx = 2
    # viewMatrix = p.computeViewMatrixFromYawPitchRoll(
    #     cam_targetPos, cam_distance, cam_yaw, cam_pitch, cam_roll, cam_up_axis_idx
    # )  # 计算视角矩阵       [R, t, 0, 1]

    # proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)  # 计算投影矩阵
    # # 设置视角
    # p.resetDebugVisualizerCamera(cameraDistance=0.58,cameraYaw=-90,cameraPitch=-89,cameraTargetPosition=[1.0, -0.2, 1.3])

    # 创建用于保存数据的目录
    output_dir = "camera_outputs_franka"
    os.makedirs(output_dir, exist_ok=True)
    rgb_dir = os.path.join(output_dir, 'images')
    os.makedirs(rgb_dir, exist_ok = True)
    anno_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(anno_dir, exist_ok = True)
    depth_dir = os.path.join(output_dir, 'depths')
    os.makedirs(depth_dir, exist_ok = True)

    # 初始化一个字典来存储所有帧的分割信息
    camera_data = {
        #"categories": [{"id":1, "name":"CrackerBox"}, {"id":2, "name":"GelatinBox"}, {"id":3, "name":"PottedMeatCan"}, {"id":4, "name":"MustardBottle"}, {"id":5, "name":"SugarBox"}, {"id":6, "name":"PuddingBox"}],
        "categories": [{"id":1, "name":"CrackerBox"}, {"id":2, "name":"GelatinBox"}, {"id":3, "name":"PottedMeatCan"}, {"id":4, "name":"MustardBottle"}],
        "images": [],    #{"id": , "width": , "height": ,"file_name": }
        "annotations": []   #{"id":, "image_id", "category_id", "segmentation", "bbox"}
    }

    category_map = ENV_CONFIG["category_map"]

    # 循环次数和物体数量
    cycle_num = 50
    obj_num = ENV_CONFIG["obj_num"]
    
    # 初始化结果记录
    results = {
        "objnumlist": [],
        "sucktimelist": [],
        "successobjlist": [],
        "failobjlist": [],
        "remainobjlist": [],
        "successratelist": [],
        "totaldistancelist": [],
        "averagedistancelist": [],
        "timelist": []
    }
    
    # PPO模式处理
    if args.mode == "ppo":
        # 创建主目录结构
        main_dir = "ppo_model"
        os.makedirs(main_dir, exist_ok=True)
        
        # 创建子目录
        checkpoint_dir = os.path.join(main_dir, "Checkpoint")
        tensorboard_dir = os.path.join(main_dir, "tensorboard")
        trained_model_dir = os.path.join(main_dir, "trained_model")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(trained_model_dir, exist_ok=True)
        
        # 创建log目录和子目录
        os.makedirs("log", exist_ok=True)
        os.makedirs(os.path.join("log", "hyperparameters"), exist_ok=True)
        
        # 确定当前训练的序号
        existing_checkpoints = [d for d in os.listdir(checkpoint_dir) 
                        if os.path.isdir(os.path.join(checkpoint_dir, d)) 
                        and d.startswith("ppo_model_")]
        
        existing_models = [f for f in os.listdir(trained_model_dir) 
                        if f.startswith("model_") and f.endswith(".zip")]
        
        # 计算新的训练序号
        if existing_checkpoints:
            checkpoint_nums = [int(d.split("_")[-1]) for d in existing_checkpoints]
            current_run = max(checkpoint_nums) + 1
        elif existing_models:
            model_nums = [int(f.split("_")[-1].split(".")[0]) for f in existing_models]
            current_run = max(model_nums) + 1
        else:
            current_run = 1
        
        # 创建唯一的时间戳
        #timestamp = int(time.time())
        run_name = f"run_{current_run}"
        
        # 创建本次训练的checkpoint目录
        current_checkpoint_dir = os.path.join(checkpoint_dir, f"ppo_model_{current_run}")
        os.makedirs(current_checkpoint_dir, exist_ok=True)
        
        # 创建唯一的tensorboard日志目录
        tb_log_dir = os.path.join(tensorboard_dir, run_name)
        os.makedirs(tb_log_dir, exist_ok=True)
        
        # 添加验证代码
        print(f"TensorBoard日志目录: {tb_log_dir}")
        print(f"TensorBoard目录权限: {os.access(tb_log_dir, os.W_OK)}")
        
        # 尝试写入测试文件验证权限
        test_file = os.path.join(tb_log_dir, "test_write.txt")
        try:
            with open(test_file, "w") as f:
                f.write("Test write access")
            print(f"成功写入测试文件: {test_file}")
            os.remove(test_file)  # 删除测试文件
        except Exception as e:
            print(f"无法写入测试文件: {e}")
        
        # 设置模型路径
        model_path = os.path.join(trained_model_dir, f"model_{current_run}")
        
        # 是否训练新模型
        #train_new_model = False
        # 是否训练新模型
        train_new_model = True
        
        # 初始化模型变量为None
        model = None
        eval_env = None
        
        if train_new_model:
            try:
                # 使用多进程实现并行环境
                num_envs = 4  # 恢复4个环境，但禁用GUI
                
                # 创建通信队列
                info_queue = Queue()
                results_queue = Queue()  # 新增结果队列
                
                # 启动一个学习器进程来管理所有环境
                learner_process = Process(
                    target=improved_learner, 
                    args=(num_envs, current_checkpoint_dir, tb_log_dir, model_path, info_queue)
                )
                learner_process.start()
                print(f"学习器进程已启动，管理 {num_envs} 个环境")
                
                # 启动日志记录进程，添加results_queue参数
                logger_process = Process(target=logger, args=(info_queue, results_queue), daemon=True)
                logger_process.start()
                print("日志记录进程已启动")
                
                # 等待学习器完成
                learner_process.join()
                
                # 然后尝试获取结果
                try:
                    if not results_queue.empty():
                        training_results = results_queue.get(timeout=10)
                        # 合并训练结果
                        for key in results:
                            if key in training_results:
                                results[key].extend(training_results[key])
                        print("训练结果已成功收集")
                    else:
                        print("结果队列为空，等待更长时间...")
                        # 给日志进程更多时间处理结果
                        time.sleep(5)
                        if not results_queue.empty():
                            training_results = results_queue.get(timeout=10)
                            # 合并训练结果
                            for key in results:
                                if key in training_results:
                                    results[key].extend(training_results[key])
                            print("训练结果已成功收集")
                        else:
                            print("无法获取训练结果，队列仍为空")
                            # 创建默认结果以避免"没有结果可保存"
                            training_results = {
                                "objnumlist": [4],  # 假设有4个物体
                                "sucktimelist": [1],
                                "successobjlist": [0],
                                "failobjlist": [1],
                                "remainobjlist": [4],
                                "successratelist": [0],
                                "totaldistancelist": [0],
                                "averagedistancelist": [0],
                                "timelist": [time.time()]
                            }
                            for key in results:
                                if key in training_results:
                                    results[key].extend(training_results[key])
                            print("使用默认结果")
                except Exception as e:
                    print(f"获取训练结果时出错: {e}")
                    # 创建默认结果
                    training_results = {
                        "objnumlist": [4],
                        "sucktimelist": [1],
                        "successobjlist": [0],
                        "failobjlist": [1],
                        "remainobjlist": [4],
                        "successratelist": [0],
                        "totaldistancelist": [0],
                        "averagedistancelist": [0],
                        "timelist": [time.time()]
                    }
                    for key in results:
                        if key in training_results:
                            results[key].extend(training_results[key])
                    print("出错后使用默认结果")
                
                # 最后再终止其他进程
                if learner_process.is_alive():
                    print("终止学习器进程...")
                    learner_process.terminate()
                
                if logger_process.is_alive():
                    print("终止日志记录进程...")
                    logger_process.terminate()
                
                print("所有进程已终止")
                
            except Exception as e:
                print(f"多进程训练中发生错误: {e}")
                import traceback
                traceback.print_exc()
        else:
            # 加载已有模型
            try:
                # 首先确认没有已存在的连接
                try:
                    p.disconnect()
                except:
                    pass
                
                # 创建一个单一的GUI连接
                physics_client = p.connect(p.GUI)
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setGravity(0, 0, -50)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
                p.setRealTimeSimulation(0)
                p.setTimeStep(1./120.)
                
                # 设置视角
                p.resetDebugVisualizerCamera(cameraDistance=0.58, cameraYaw=-90, cameraPitch=-89,
                                           cameraTargetPosition=[1.0, -0.2, 1.3])
                
                # 计算视图和投影矩阵
                viewMatrix = p.computeViewMatrixFromYawPitchRoll(
                    cam_targetPos, cam_distance, cam_yaw, cam_pitch, cam_roll, cam_up_axis_idx
                )
                proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
                
                # 修改2：传递已存在的physics_client以避免重复连接
                eval_env = PickAndPlaceEnv(render=True, viewMatrix=viewMatrix, proj_matrix=proj_matrix, 
                                           physics_client=physics_client)
                
                # 加载模型
                #model = PPO.load("./ppo_model/trained_model/model_34_step_4500", env=eval_env)
                model = PPO.load("./improved_checkpoints/improved_ppo_model", env=eval_env)
                #print(f"已加载模型: ./past/ppo_pick_place_model")
                print(f"已加载模型: ./ppo_model/trained_model/model_26")
            except Exception as e:
                print(f"加载模型时出现错误: {e}")
                model = None
                eval_env = None
                
                # 确保断开任何可能的连接
                try:
                    p.disconnect()
                except:
                    pass
        
        # 模型测试和评估
        if model is not None:
            print(f"开始测试训练模型")
            results_file = os.path.join(main_dir, f"evaluation_results_{current_run}.txt")

            # 循环评估多个环境
            for i in range(cycle_num):
                print(f"评估第 {i+1}/{cycle_num} 个场景")
                
                # 重置环境
                obs, _ = eval_env.reset()  # 兼容新版Gym API
                done = False
                total_reward = 0
                
                # 使用模型进行预测
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = eval_env.step(action)  # 兼容新版Gym API
                    total_reward += reward
                
                # 记录本次评估结果
                cycle_results = {
                    "objnumlist": [len(eval_env.obj_ids) + eval_env.success_count],
                    "sucktimelist": [eval_env.step_counter],
                    "successobjlist": [eval_env.success_count],
                    "failobjlist": [eval_env.step_counter - eval_env.success_count],
                    "remainobjlist": [len(eval_env.obj_ids)],
                    "successratelist": [eval_env.success_count / max(1, eval_env.step_counter)],
                    "totaldistancelist": [eval_env.total_distance],
                    "averagedistancelist": [eval_env.total_distance / max(1, eval_env.step_counter)],
                    "timelist": [time.time() - eval_env.start_time]
                }
                
                # 合并结果
                for key in results:
                    results[key].extend(cycle_results[key])
                
                print(f"总奖励: {total_reward:.2f}")
                print(f"成功抓取: {eval_env.success_count}/{eval_env.step_counter}")
                print(f"总位移: {eval_env.total_distance:.2f}")
                print(f"总时间: {time.time() - eval_env.start_time:.2f}秒")
                # 保存每次评估结果
                with open(results_file, "w") as f:
                    f.write(f"总奖励: {total_reward:.2f}\n")
                    f.write(f"成功抓取: {eval_env.success_count}/{eval_env.step_counter}\n")
                    f.write(f"总位移: {eval_env.total_distance:.2f}\n")
                    f.write(f"总时间: {time.time() - eval_env.start_time:.2f}秒\n")

            # 关闭评估环境
            eval_env.close()

    else:
        # 非PPO模式才在主线程连接物理引擎
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 确保能找到模型文件
        p.setGravity(0, 0, -50)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setRealTimeSimulation(0)
        p.setTimeStep(1./120.)
        
        # 设置视角
        p.resetDebugVisualizerCamera(cameraDistance=0.58, cameraYaw=-90, cameraPitch=-89,
                                    cameraTargetPosition=[1.0, -0.2, 1.3])
                                    
        # 计算视图矩阵                           
        viewMatrix = p.computeViewMatrixFromYawPitchRoll(
            cam_targetPos, cam_distance, cam_yaw, cam_pitch, cam_roll, cam_up_axis_idx
        )
        proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        
        # 循环次数和物体数量
        cycle_num = 50
        obj_num = ENV_CONFIG["obj_num"]
        
        for i in range(cycle_num):
            print(f"开始第 {i+1}/{cycle_num} 个场景")
            
            # 设置场景
            scene = setup_scene()
            robot_id = scene["robot_id"]
            num_joints = scene["num_joints"]
            kuka_end_effector_idx = scene["kuka_end_effector_idx"]
            kuka_cid = scene["kuka_cid"]
            
            # 获取墙壁ID
            wall_ids = scene["wall_ids"]
            wall_id1 = wall_ids["wall_id1"]
            wall_id2 = wall_ids["wall_id2"]
            wall_id3 = wall_ids["wall_id3"]
            wall_id4 = wall_ids["wall_id4"]
            wall_id5 = wall_ids["wall_id5"]
            wall_id6 = wall_ids["wall_id6"]
            wall_id7 = wall_ids["wall_id7"]
            
            # 置随机数，一个环境一个随机数便于复现
            random_seed = i + 1
            random.seed(random_seed)
            
            obj_ids = generate_objects(obj_num)
            obj_number = len(obj_ids)
            
            # 移除墙壁
            p.removeBody(wall_id1)
            p.removeBody(wall_id2)
            p.removeBody(wall_id3)
            p.removeBody(wall_id5)
            p.removeBody(wall_id7)
            
            #=================拍照，存数据==========================
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(1408, 1024, viewMatrix, projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)   #   rgbImg (1408, 1024, 4) rgba depthImg ()
            # save_data(obj_ids, camera_data, random_seed, rgbImg, depthImg, segImg, rgb_dir, depth_dir)
            #=====================================================

            # 记录初始位置
            initial_positions = {}
            for obj in obj_ids:
                initial_positions[obj['id']] = p.getBasePositionAndOrientation(obj['id'])[0]
            
            # 根据模式排序物体
            if args.mode == "height":
                obj_ids.sort(key=lambda obj: p.getBasePositionAndOrientation(obj['id'])[0][2], reverse=True)
            elif args.mode == "reverse":
                obj_ids.reverse()
            elif args.mode == "random":
                random.shuffle(obj_ids)
            elif args.mode == "distance":
                circle_center = ENV_CONFIG["circle_center"]
                for obj in obj_ids:
                    pos = p.getBasePositionAndOrientation(obj['id'])[0]
                    obj_ids.sort(key=lambda obj: math.sqrt((pos[0]-circle_center[0])**2 + (pos[1]-circle_center[1])**2))
            
            # 执行抓取操作
            cycle_results = pick(obj_ids, initial_positions, kuka_cid, robot_id, num_joints, i, obj_number)
            
            # 合并结果
            for key in results:
                results[key].extend(cycle_results[key])
    
    # 保存CSV结果
    save_results_to_csv(results, args.mode, cycle_num)

def save_results_to_csv(results, mode, cycle_num):
    """保存结果到CSV文件"""
    # 确保有结果可以保存
    if not any(results.values()):  # 检查所有结果列表是否都为空
        print("没有结果可保存，创建默认结果")
        # 创建默认结果
        results = {
            "objnumlist": [4],
            "sucktimelist": [0],
            "successobjlist": [0],
            "failobjlist": [0],
            "remainobjlist": [4],
            "successratelist": [0],
            "totaldistancelist": [0],
            "averagedistancelist": [0],
            "timelist": [time.time()]
        }
    
    # 调整循环次数，确保不超过结果列表长度
    actual_cycles = min(cycle_num, len(results["objnumlist"]))
    
    # 提取运行序号（如果是ppo模式）
    current_run = 1
    if mode == "ppo":
        # 尝试从文件名中提取运行序号
        try:
            # 查找存在的最高序号
            import glob
            tensorboard_files = glob.glob("ppo_model/tensorboard/run_*")
            if tensorboard_files:
                run_numbers = [int(os.path.basename(f).split("_")[1]) for f in tensorboard_files]
                current_run = max(run_numbers, default=1)
        except:
            pass
    
    # 确保log目录存在
    os.makedirs('log', exist_ok=True)
    
    # 使用运行序号保存到log文件夹
    output_file = f'log/evaluation_results_{current_run}.csv'
    
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # 写入表头
        header = ['环境编号', '物体个数', '总吸取次数', '成功吸取物体个数', '失败物体个数', 
                 '残留物体个数', '吸取成功率', '物体位移总距离', '物体位移平均距离', '抓取时间']
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

# 修改2: 确保安全计算平均值
def safe_mean(xs):
    """安全计算平均值，避免空列表"""
    return float(sum(xs)) / max(len(xs), 1)

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序异常: {e}")
        import traceback
        traceback.print_exc() 