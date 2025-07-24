import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical  # 改为Categorical分布
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from collections import deque
import mani_skill.envs
from env_clutter import EnvClutterEnv
from utils import CsvLogger  # 导入CsvLogger
# 新增：导入视频录制相关模块
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import warnings
warnings.filterwarnings("ignore")

class PPOActor(nn.Module):
    """PPO Actor网络 - 支持离散动作"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.logits_layer = nn.Linear(hidden_dim, action_dim)  # 输出logits
        
    def forward(self, state, mask=None):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.logits_layer(x)
        
        # 不在forward中应用掩码，只返回原始logits
        return logits
    
    def get_action(self, state, mask=None):
        logits = self.forward(state, mask)
        
        # 屏蔽非法动作
        if mask is not None:
            # 确保掩码是0/1值
            if not torch.all((mask == 0) | (mask == 1)):
                # 如果掩码不是0/1值，将其转换为0/1值
                # 假设非零值表示有效动作
                mask = (mask != 0).float()
            
            logits = torch.where(mask.bool(), logits, torch.tensor(-1e8, device=logits.device))
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def evaluate_action(self, state, action, mask=None):
        logits = self.forward(state, mask)
        
        # 屏蔽非法动作
        if mask is not None:
            # 确保掩码是0/1值
            if not torch.all((mask == 0) | (mask == 1)):
                # 如果掩码不是0/1值，将其转换为0/1值
                # 假设非零值表示有效动作
                mask = (mask != 0).float()
            
            logits = torch.where(mask.bool(), logits, torch.tensor(-1e8, device=logits.device))
        
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy

class PPOCritic(nn.Module):
    """PPO Critic网络"""
    def __init__(self, state_dim, hidden_dim=256):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value_layer(x)
        return value

class PPOAgent:
    """PPO智能体"""
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络
        self.actor = PPOActor(state_dim, action_dim).to(self.device)
        self.critic = PPOCritic(state_dim).to(self.device)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
    def get_action(self, state, mask=None):
        state = torch.FloatTensor(state).to(self.device)
        if mask is not None:
            mask = torch.FloatTensor(mask).to(self.device)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state, mask)
        return action.cpu().numpy(), log_prob.item()
    
    def get_value(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            value = self.critic(state)
        return value.item()
    
    def compute_gae(self, rewards, values, next_values, dones):
        """计算广义优势估计"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, states, actions, old_log_probs, rewards, values, dones, masks=None, epochs=10):
        """更新网络 - 适配离散动作"""
        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)  # 离散动作用LongTensor
        
        # 处理masks
        if masks is not None:
            masks = torch.FloatTensor(np.array(masks)).to(self.device)
        
        # 处理old_log_probs
        old_log_probs_array = []
        for log_prob in old_log_probs:
            if isinstance(log_prob, torch.Tensor):
                old_log_probs_array.append(log_prob.item() if log_prob.numel() == 1 else log_prob.cpu().numpy())
            elif isinstance(log_prob, np.ndarray):
                old_log_probs_array.append(log_prob.item() if log_prob.size == 1 else log_prob)
            else:
                old_log_probs_array.append(float(log_prob))
        old_log_probs = torch.FloatTensor(old_log_probs_array).to(self.device)
        
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        # 处理values
        values_array = []
        for value in values:
            if isinstance(value, torch.Tensor):
                values_array.append(value.item() if value.numel() == 1 else value.cpu().numpy())
            elif isinstance(value, np.ndarray):
                values_array.append(value.item() if value.size == 1 else value)
            else:
                values_array.append(float(value))
        values = torch.FloatTensor(values_array).to(self.device)
        
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算优势
        next_values = self.critic(states[-1:]).squeeze()
        advantages = self.compute_gae(rewards, values, next_values, dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + values
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 更新网络
        for _ in range(epochs):
            # Actor损失
            current_mask = masks if masks is not None else None
            new_log_probs, entropy = self.actor.evaluate_action(states, actions, current_mask)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
            
            # Critic损失
            new_values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(new_values, returns)
            
            # 更新
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

def flatten_obs(obs):
    """展平观测 - 支持离散动作观测"""
    if isinstance(obs, dict):
        flattened = []
        for key in sorted(obs.keys()):
            if key in ['sensor_data']:
                continue  # 跳过图像数据
            value = obs[key]
            if isinstance(value, torch.Tensor):
                flattened.append(value.flatten())
            elif isinstance(value, np.ndarray):
                flattened.append(torch.from_numpy(value).flatten())
            elif isinstance(value, (list, tuple)):
                flattened.append(torch.tensor(value).flatten())
            else:
                flattened.append(torch.tensor([value]).flatten())
        return torch.cat(flattened)
    else:
        # 处理非字典类型的观测
        if isinstance(obs, torch.Tensor):
            return obs.flatten()
        elif isinstance(obs, np.ndarray):
            return torch.from_numpy(obs).flatten()
        else:
            return torch.tensor(obs).flatten()

def extract_mask(obs):
    """从观测中提取动作掩码"""
    MAX_N = 15  # 与env_clutter.py中的MAX_N保持一致
    
    if isinstance(obs, dict) and 'discrete_action_obs' in obs:
        discrete_obs = obs['discrete_action_obs']
        if isinstance(discrete_obs, torch.Tensor):
            # discrete_action_obs的结构：[action_mask(MAX_N), object_features(MAX_N*7), step_count(1)]
            # 我们需要提取前MAX_N个元素作为action_mask
            mask = discrete_obs[:MAX_N] if discrete_obs.dim() == 1 else discrete_obs[:, :MAX_N]
            mask = mask.cpu().numpy()
            # 确保返回1D数组
            if mask.ndim > 1:
                mask = mask.flatten()
            return mask
        elif isinstance(discrete_obs, np.ndarray):
            mask = discrete_obs[:MAX_N] if discrete_obs.ndim == 1 else discrete_obs[:, :MAX_N]
            # 确保返回1D数组
            if mask.ndim > 1:
                mask = mask.flatten()
            return mask
    elif isinstance(obs, torch.Tensor):
        # 如果obs直接是张量，说明是修改后的环境返回的展平观测
        # 新的观测结构：按字母顺序排列的键
        # discrete_action_obs是第一个键，包含121个元素
        # 其中前15个是掩码
        
        if obs.dim() == 1:
            # 1D张量，直接提取前15个元素作为掩码
            mask = obs[:MAX_N]
        else:
            # 2D张量，取第一个batch的前15个元素
            mask = obs[0, :MAX_N]
        
        mask = mask.cpu().numpy()
        # 确保返回1D数组
        if mask.ndim > 1:
            mask = mask.flatten()
        return mask
    return None

def train_ppo(args):
    """训练PPO智能体"""
    # 创建环境
    env = gym.make(
        "EnvClutter-v1",
        num_envs=args.num_envs,
        obs_mode="rgb" if args.record_video else "state",  # 录制视频时使用rgb模式
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        render_mode="rgb_array" if args.record_video else ("human" if args.render else None),
        use_discrete_action=True,  # 启用离散动作
        # 录制视频时增加传感器配置
        **(dict(sensor_configs=dict(width=args.video_width, height=args.video_height)) if args.record_video else {})
    )
    
    # 新增：视频录制包装器
    if args.record_video:
        video_output_dir = os.path.join(args.log_dir, "training_videos")
        os.makedirs(video_output_dir, exist_ok=True)
        print(f"训练视频将保存到: {video_output_dir}")
        
        # 设置视频录制触发器：每隔指定间隔录制一次
        def video_trigger(episode_count):
            return episode_count % args.video_record_interval == 0
        
        env = RecordEpisode(
            env,
            output_dir=video_output_dir,
            save_trajectory=args.save_trajectory,
            save_video=True,
            trajectory_name="training_trajectory",
            max_steps_per_video=args.max_video_steps,
            video_fps=args.video_fps,
            render_substeps=True,  # 启用子步渲染以获得更流畅的视频
            info_on_video=True,  # 在视频上显示信息
            save_video_trigger=video_trigger,  # 使用触发器控制录制时机
            avoid_overwriting_video=True,  # 避免覆盖已有视频
        )
        print("✓ 视频录制包装器添加成功")
    
    # 新增：向量化包装器（如果启用了视频录制）
    if args.record_video:
        env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=False, record_metrics=True)
        print("✓ 向量化包装器添加成功")
    
    # 获取状态和动作维度
    obs, _ = env.reset()
    flattened_obs = flatten_obs(obs)
    state_dim = flattened_obs.shape[0]
    
    # 获取动作维度
    if hasattr(env, 'discrete_action_space') and env.discrete_action_space is not None:
        action_dim = env.discrete_action_space.n
        print(f"使用离散动作空间，动作维度: {action_dim}")
    elif hasattr(env.unwrapped, 'discrete_action_space') and env.unwrapped.discrete_action_space is not None:
        action_dim = env.unwrapped.discrete_action_space.n
        print(f"使用离散动作空间，动作维度: {action_dim}")
    else:
        action_dim = env.action_space.shape[0]
        print(f"使用连续动作空间，动作维度: {action_dim}")
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"渲染模式: {'视频录制' if args.record_video else ('人类观察' if args.render else '关闭')}")
    
    # 创建智能体
    agent = PPOAgent(state_dim, action_dim)
    
    # 创建日志记录器
    writer = SummaryWriter(log_dir=args.log_dir)
    csv_logger = CsvLogger(os.path.join(args.log_dir, "training_log.csv"))
    
    # 训练循环
    episode_rewards = deque(maxlen=100)
    episode_success_rates = deque(maxlen=100)
    
    total_steps = 0
    episode_count = 0
    
    for epoch in range(args.epochs):
        # 收集数据
        states, actions, log_probs, rewards, values, dones, masks = [], [], [], [], [], [], []
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_success = 0
        total_displacement = 0
        
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        for step in range(args.steps_per_epoch):
            # 展平观测
            flattened_obs = flatten_obs(obs)
            state = flattened_obs.cpu().numpy() if isinstance(flattened_obs, torch.Tensor) else flattened_obs
            
            # 提取掩码
            mask = extract_mask(obs)
            
            # 获取动作
            action, log_prob = agent.get_action(state, mask)
            value = agent.get_value(state)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 渲染环境
            if args.render and not args.record_video:  # 避免重复渲染
                env.render()
                time.sleep(0.01)
            
            # 处理奖励和信息
            if isinstance(reward, torch.Tensor):
                reward = reward.item() if reward.numel() == 1 else reward.mean().item()
            elif isinstance(reward, np.ndarray):
                reward = reward.item() if reward.size == 1 else reward.mean()
            
            # 处理done标志
            if isinstance(done, torch.Tensor):
                done = done.item() if done.numel() == 1 else done.any().item()
            elif isinstance(done, np.ndarray):
                done = done.item() if done.size == 1 else done.any()
            
            # 处理成功信息
            success = False
            displacement = 0.0
            if isinstance(info, dict):
                success = info.get('success', False)
                displacement = info.get('displacement', 0.0)
                if isinstance(success, torch.Tensor):
                    success = success.item() if success.numel() == 1 else success.any().item()
                elif isinstance(success, np.ndarray):
                    success = success.item() if success.size == 1 else success.any()
            
            # 存储数据
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            if mask is not None:
                masks.append(mask)
            
            episode_reward += reward
            total_displacement += displacement
            if success:
                episode_success = 1
            
            obs = next_obs
            total_steps += 1
            
            # 打印步骤信息（录制视频时更详细）
            if (args.render or args.record_video) and step % 10 == 0:
                print(f"步骤 {step}: 动作={action}, 奖励={reward:.3f}, 成功={success}, 完成={done}")
            
            # 如果episode结束
            if done:
                episode_rewards.append(episode_reward)
                episode_success_rates.append(episode_success)
                episode_count += 1
                
                print(f"Episode {episode_count} 结束: 奖励={episode_reward:.3f}, 成功={episode_success}")
                
                # 视频录制完成提示
                if args.record_video and (episode_count - 1) % args.video_record_interval == 0:
                    print(f"📹 Episode {episode_count} 的训练视频已录制完成")
                
                # 重置环境
                obs, _ = env.reset()
                episode_reward = 0
                episode_success = 0
                total_displacement = 0
        
        # 更新智能体
        if len(states) > 0:
            print(f"更新智能体，数据量: {len(states)}")
            mask_data = masks if len(masks) > 0 else None
            actor_loss, critic_loss = agent.update(
                states, actions, log_probs, rewards, values, dones, mask_data
            )
            
            # 记录日志
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_success_rate = np.mean(episode_success_rates) if episode_success_rates else 0
            
            writer.add_scalar('Training/Episode_Reward', avg_reward, epoch)
            writer.add_scalar('Training/Success_Rate', avg_success_rate, epoch)
            writer.add_scalar('Training/Actor_Loss', actor_loss, epoch)
            writer.add_scalar('Training/Critic_Loss', critic_loss, epoch)
            
            # 新增：视频录制相关日志
            if args.record_video:
                writer.add_scalar('Training/Episodes_Recorded', episode_count // args.video_record_interval, epoch)
            
            # CSV日志
            csv_logger.log({
                'epoch': epoch,
                'episode': episode_count,
                'avg_reward': avg_reward,
                'success_rate': avg_success_rate,
                'total_displacement': total_displacement,
                'steps': total_steps,
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'video_recorded': args.record_video and (episode_count - 1) % args.video_record_interval == 0
            })
            
            if epoch % args.log_interval == 0:
                log_msg = (f"Epoch {epoch}, "
                          f"平均奖励: {avg_reward:.2f}, "
                          f"成功率: {avg_success_rate:.2f}, "
                          f"Actor损失: {actor_loss:.4f}, "
                          f"Critic损失: {critic_loss:.4f}")
                
                if args.record_video:
                    recorded_episodes = episode_count // args.video_record_interval
                    log_msg += f", 已录制视频: {recorded_episodes} 个episode"
                
                print(log_msg)
        
        # 保存模型
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.model_dir, f"ppo_model_epoch_{epoch}.pth")
            agent.save(save_path)
            print(f"模型已保存到: {save_path}")
    
    # 保存最终模型
    final_save_path = os.path.join(args.model_dir, "ppo_model_final.pth")
    agent.save(final_save_path)
    print(f"最终模型已保存到: {final_save_path}")
    
    env.close()
    writer.close()

def main():
    parser = argparse.ArgumentParser(description='训练EnvClutter环境的PPO智能体')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--steps_per_epoch', type=int, default=2048, help='每轮步数')
    parser.add_argument('--num_envs', type=int, default=1, help='并行环境数量')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--model_dir', type=str, default='./models/env_clutter', help='模型保存目录')
    parser.add_argument('--log_interval', type=int, default=10, help='日志记录间隔')
    parser.add_argument('--save_interval', type=int, default=100, help='模型保存间隔')
    parser.add_argument('--render', action='store_true', help='是否渲染')
    
    # 新增：视频录制相关参数
    parser.add_argument('--record_video', action='store_true', help='是否录制训练视频')
    parser.add_argument('--save_trajectory', action='store_true', help='是否保存轨迹数据')
    parser.add_argument('--video_record_interval', type=int, default=50, help='视频录制间隔（每多少个episode录制一次）')
    parser.add_argument('--max_video_steps', type=int, default=1000, help='每个视频的最大步数')
    parser.add_argument('--video_fps', type=int, default=30, help='视频帧率')
    parser.add_argument('--video_width', type=int, default=256, help='视频宽度')
    parser.add_argument('--video_height', type=int, default=256, help='视频高度')
    parser.add_argument('--settle_steps', type=int, default=30, help='物体稳定等待步数（录制视频时）')
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 开始训练
    train_ppo(args)

if __name__ == "__main__":
    main() 