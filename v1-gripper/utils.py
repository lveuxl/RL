"""
EnvClutter环境的工具函数
包含可视化、数据处理、评估等辅助功能
"""

import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
import json
from collections import defaultdict, deque
import time
import gymnasium as gym
from pathlib import Path

def setup_seed(seed: int = 42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device: str = "auto") -> torch.device:
    """获取计算设备"""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device)

def flatten_observation(obs: Any) -> np.ndarray:
    """展平观测数据"""
    if isinstance(obs, dict):
        flattened = []
        for key in sorted(obs.keys()):
            if key == 'sensor_data':
                continue  # 跳过图像数据
            value = obs[key]
            if isinstance(value, torch.Tensor):
                flattened.append(value.cpu().numpy().flatten())
            elif isinstance(value, np.ndarray):
                flattened.append(value.flatten())
            elif isinstance(value, (list, tuple)):
                flattened.append(np.array(value).flatten())
            else:
                flattened.append(np.array([value]).flatten())
        return np.concatenate(flattened)
    else:
        return np.array(obs).flatten()

def unflatten_observation(flattened_obs: np.ndarray, obs_structure: Dict) -> Dict:
    """将展平的观测数据恢复为原始结构"""
    # 这是一个简化版本，实际实现需要根据具体的观测结构来定制
    result = {}
    idx = 0
    for key, shape in obs_structure.items():
        if key == 'sensor_data':
            continue
        size = np.prod(shape)
        result[key] = flattened_obs[idx:idx+size].reshape(shape)
        idx += size
    return result

class RewardTracker:
    """奖励跟踪器"""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rewards = deque(maxlen=window_size)
        self.success_rates = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.displacement_penalties = deque(maxlen=window_size)
        
    def add_episode(self, reward: float, success: bool, length: int, displacement: float = 0.0):
        """添加一个episode的数据"""
        self.rewards.append(reward)
        self.success_rates.append(float(success))
        self.episode_lengths.append(length)
        self.displacement_penalties.append(displacement)
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        if len(self.rewards) == 0:
            return {}
        
        return {
            'mean_reward': np.mean(self.rewards),
            'std_reward': np.std(self.rewards),
            'success_rate': np.mean(self.success_rates),
            'mean_episode_length': np.mean(self.episode_lengths),
            'mean_displacement_penalty': np.mean(self.displacement_penalties),
            'num_episodes': len(self.rewards)
        }

class VideoRecorder:
    """视频录制器"""
    def __init__(self, save_dir: str, fps: int = 30):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frames = []
        self.recording = False
        
    def start_recording(self):
        """开始录制"""
        self.frames = []
        self.recording = True
        
    def add_frame(self, frame: np.ndarray):
        """添加帧"""
        if self.recording:
            self.frames.append(frame)
    
    def stop_recording(self, filename: str):
        """停止录制并保存"""
        if not self.recording or len(self.frames) == 0:
            return
        
        filepath = self.save_dir / filename
        height, width = self.frames[0].shape[:2]
        
        # 使用OpenCV保存视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(filepath), fourcc, self.fps, (width, height))
        
        for frame in self.frames:
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        self.recording = False
        print(f"视频已保存到: {filepath}")

class PerformanceProfiler:
    """性能分析器"""
    def __init__(self):
        self.timers = defaultdict(list)
        self.start_times = {}
        
    def start_timer(self, name: str):
        """开始计时"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str):
        """结束计时"""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.timers[name].append(elapsed)
            del self.start_times[name]
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """获取性能统计"""
        stats = {}
        for name, times in self.timers.items():
            stats[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times),
                'count': len(times)
            }
        return stats
    
    def print_stats(self):
        """打印性能统计"""
        stats = self.get_stats()
        print("\n=== 性能统计 ===")
        for name, stat in stats.items():
            print(f"{name}:")
            print(f"  平均: {stat['mean']:.4f}s")
            print(f"  标准差: {stat['std']:.4f}s")
            print(f"  最小: {stat['min']:.4f}s")
            print(f"  最大: {stat['max']:.4f}s")
            print(f"  总计: {stat['total']:.4f}s")
            print(f"  次数: {stat['count']}")

def visualize_training_progress(log_dir: str, save_path: str = None):
    """可视化训练进度"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # 读取tensorboard日志
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        # 获取标量数据
        scalar_keys = ea.Tags()['scalars']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练进度', fontsize=16)
        
        # 绘制奖励曲线
        if 'Training/Episode_Reward' in scalar_keys:
            rewards = ea.Scalars('Training/Episode_Reward')
            steps = [r.step for r in rewards]
            values = [r.value for r in rewards]
            axes[0, 0].plot(steps, values)
            axes[0, 0].set_title('Episode Reward')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        # 绘制成功率曲线
        if 'Training/Success_Rate' in scalar_keys:
            success_rates = ea.Scalars('Training/Success_Rate')
            steps = [s.step for s in success_rates]
            values = [s.value for s in success_rates]
            axes[0, 1].plot(steps, values)
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].grid(True)
        
        # 绘制Actor损失
        if 'Training/Actor_Loss' in scalar_keys:
            actor_losses = ea.Scalars('Training/Actor_Loss')
            steps = [a.step for a in actor_losses]
            values = [a.value for a in actor_losses]
            axes[1, 0].plot(steps, values)
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # 绘制Critic损失
        if 'Training/Critic_Loss' in scalar_keys:
            critic_losses = ea.Scalars('Training/Critic_Loss')
            steps = [c.step for c in critic_losses]
            values = [c.value for c in critic_losses]
            axes[1, 1].plot(steps, values)
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练进度图已保存到: {save_path}")
        
        plt.show()
        
    except ImportError:
        print("需要安装tensorboard来可视化训练进度")
    except Exception as e:
        print(f"可视化训练进度时出错: {e}")

def evaluate_model(env, agent, num_episodes: int = 100, render: bool = False, 
                  video_recorder: Optional[VideoRecorder] = None) -> Dict[str, Any]:
    """评估模型性能"""
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    displacement_penalties = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_success = False
        episode_displacement = 0
        
        if video_recorder and episode < 5:  # 只录制前5个episode
            video_recorder.start_recording()
        
        while True:
            # 获取动作
            flattened_obs = flatten_observation(obs)
            action, _ = agent.get_action(flattened_obs)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if info.get('success', False):
                episode_success = True
            
            # 记录位移惩罚
            if hasattr(env, '_calculate_other_objects_displacement'):
                displacement = env._calculate_other_objects_displacement()
                episode_displacement += displacement.item() if hasattr(displacement, 'item') else displacement
            
            if render:
                env.render()
            
            if video_recorder and episode < 5:
                frame = env.render()
                if frame is not None:
                    video_recorder.add_frame(frame)
            
            if terminated or truncated:
                break
        
        if video_recorder and episode < 5:
            video_recorder.stop_recording(f"eval_episode_{episode}.mp4")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        displacement_penalties.append(episode_displacement)
        
        if episode_success:
            success_count += 1
    
    # 计算统计信息
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': success_count / num_episodes,
        'mean_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'mean_displacement_penalty': np.mean(displacement_penalties),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'displacement_penalties': displacement_penalties,
    }
    
    return results

def save_evaluation_results(results: Dict[str, Any], filepath: str):
    """保存评估结果"""
    # 转换numpy数组为列表以便JSON序列化
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_results[key] = value.item()
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"评估结果已保存到: {filepath}")

def load_evaluation_results(filepath: str) -> Dict[str, Any]:
    """加载评估结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def print_evaluation_summary(results: Dict[str, Any]):
    """打印评估摘要"""
    print("\n=== 评估结果摘要 ===")
    print(f"平均奖励: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"成功率: {results['success_rate']:.4f}")
    print(f"平均episode长度: {results['mean_episode_length']:.2f} ± {results['std_episode_length']:.2f}")
    print(f"平均位移惩罚: {results['mean_displacement_penalty']:.4f}")

def create_comparison_plot(results_list: List[Dict[str, Any]], 
                          labels: List[str], 
                          save_path: str = None):
    """创建多个模型的比较图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('模型性能比较', fontsize=16)
    
    # 准备数据
    mean_rewards = [r['mean_reward'] for r in results_list]
    std_rewards = [r['std_reward'] for r in results_list]
    success_rates = [r['success_rate'] for r in results_list]
    mean_lengths = [r['mean_episode_length'] for r in results_list]
    mean_displacements = [r['mean_displacement_penalty'] for r in results_list]
    
    x = np.arange(len(labels))
    
    # 平均奖励
    axes[0, 0].bar(x, mean_rewards, yerr=std_rewards, capsize=5)
    axes[0, 0].set_title('平均奖励')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 成功率
    axes[0, 1].bar(x, success_rates)
    axes[0, 1].set_title('成功率')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=45)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 平均episode长度
    axes[1, 0].bar(x, mean_lengths)
    axes[1, 0].set_title('平均Episode长度')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 平均位移惩罚
    axes[1, 1].bar(x, mean_displacements)
    axes[1, 1].set_title('平均位移惩罚')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"比较图已保存到: {save_path}")
    
    plt.show()

def check_environment_setup():
    """检查环境设置"""
    print("=== 环境设置检查 ===")
    
    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
    
    # 检查ManiSkill
    try:
        import mani_skill
        print(f"ManiSkill版本: {mani_skill.__version__}")
    except ImportError:
        print("ManiSkill未安装")
    
    # 检查其他依赖
    dependencies = ['numpy', 'cv2', 'matplotlib', 'gymnasium']
    for dep in dependencies:
        try:
            module = __import__(dep)
            if hasattr(module, '__version__'):
                print(f"{dep}版本: {module.__version__}")
            else:
                print(f"{dep}: 已安装")
        except ImportError:
            print(f"{dep}: 未安装")

if __name__ == "__main__":
    # 测试工具函数
    check_environment_setup()
    
    # 创建示例数据
    tracker = RewardTracker()
    for i in range(10):
        tracker.add_episode(
            reward=np.random.normal(5, 2),
            success=np.random.choice([True, False]),
            length=np.random.randint(50, 200),
            displacement=np.random.uniform(0, 1)
        )
    
    stats = tracker.get_stats()
    print("\n=== 奖励跟踪器测试 ===")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    
    print("\n工具函数测试完成！") 