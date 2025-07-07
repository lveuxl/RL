import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from collections import deque
import mani_skill.envs
from env_clutter import EnvClutterEnv
import warnings
warnings.filterwarnings("ignore")

class PPOActor(nn.Module):
    """PPO Actor网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std.clamp(-5, 2))
        return mean, std
    
    def get_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def evaluate_action(self, state, action):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
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
        
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state)
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
    
    def update(self, states, actions, old_log_probs, rewards, values, dones, epochs=10):
        """更新网络"""
        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        
        # 处理old_log_probs，确保是正确的形状
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
        
        # 处理values，确保是正确的形状
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
            new_log_probs, entropy = self.actor.evaluate_action(states, actions)
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
    """展平观测"""
    if isinstance(obs, dict):
        flattened = []
        for key in sorted(obs.keys()):
            if key == 'sensor_data':
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

def train_ppo(args):
    """训练PPO智能体"""
    # 创建环境
    env = gym.make(
        "EnvClutter-v1",
        num_envs=args.num_envs,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        reward_mode="dense",
        render_mode="human" if args.render else None,
    )
    
    # 获取状态和动作维度
    obs, _ = env.reset()
    flattened_obs = flatten_obs(obs)
    state_dim = flattened_obs.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"渲染模式: {'开启' if args.render else '关闭'}")
    
    # 创建智能体
    agent = PPOAgent(state_dim, action_dim)
    
    # 创建日志记录器
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 训练循环
    episode_rewards = deque(maxlen=100)
    episode_success_rates = deque(maxlen=100)
    
    total_steps = 0
    episode_count = 0
    
    for epoch in range(args.epochs):
        # 收集数据
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_success = 0
        
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        for step in range(args.steps_per_epoch):
            # 展平观测
            flattened_obs = flatten_obs(obs)
            state = flattened_obs.cpu().numpy() if isinstance(flattened_obs, torch.Tensor) else flattened_obs
            
            # 获取动作
            action, log_prob = agent.get_action(state)
            value = agent.get_value(state)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 渲染环境
            if args.render:
                env.render()
                # 添加小延迟以便观察
                import time
                time.sleep(0.01)
            
            # 处理奖励和信息（确保是标量）
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
            if isinstance(info, dict):
                success = info.get('success', False)
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
            
            episode_reward += reward
            if success:
                episode_success = 1
            
            obs = next_obs
            total_steps += 1
            
            # 打印步骤信息
            if args.render and step % 10 == 0:
                print(f"步骤 {step}: 奖励={reward:.3f}, 成功={success}, 完成={done}")
            
            # 如果episode结束
            if done:
                episode_rewards.append(episode_reward)
                episode_success_rates.append(episode_success)
                episode_count += 1
                
                print(f"Episode {episode_count} 结束: 奖励={episode_reward:.3f}, 成功={episode_success}")
                
                # 重置环境
                obs, _ = env.reset()
                episode_reward = 0
                episode_success = 0
        
        # 更新智能体
        if len(states) > 0:
            print(f"更新智能体，数据量: {len(states)}")
            actor_loss, critic_loss = agent.update(
                states, actions, log_probs, rewards, values, dones
            )
            
            # 记录日志
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_success_rate = np.mean(episode_success_rates) if episode_success_rates else 0
            
            writer.add_scalar('Training/Episode_Reward', avg_reward, epoch)
            writer.add_scalar('Training/Success_Rate', avg_success_rate, epoch)
            writer.add_scalar('Training/Actor_Loss', actor_loss, epoch)
            writer.add_scalar('Training/Critic_Loss', critic_loss, epoch)
            
            if epoch % args.log_interval == 0:
                print(f"Epoch {epoch}, "
                      f"平均奖励: {avg_reward:.2f}, "
                      f"成功率: {avg_success_rate:.2f}, "
                      f"Actor损失: {actor_loss:.4f}, "
                      f"Critic损失: {critic_loss:.4f}")
        
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
    parser.add_argument('--log_dir', type=str, default='./logs/env_clutter', help='日志目录')
    parser.add_argument('--model_dir', type=str, default='./models/env_clutter', help='模型保存目录')
    parser.add_argument('--log_interval', type=int, default=10, help='日志记录间隔')
    parser.add_argument('--save_interval', type=int, default=100, help='模型保存间隔')
    parser.add_argument('--render', action='store_true', help='是否渲染')
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 开始训练
    train_ppo(args)

if __name__ == "__main__":
    main() 