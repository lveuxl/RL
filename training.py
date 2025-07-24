import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from collections import deque
import mani_skill.envs
from env_clutter import EnvClutterEnv
import warnings
from tqdm import tqdm  # 添加进度条库
warnings.filterwarnings("ignore")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOActor(nn.Module):
    """PPO Actor网络 - 修改为支持离散动作空间"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 输出动作概率
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs
    
    def get_action(self, state):
        """获取动作和对数概率"""
        state = torch.FloatTensor(state).to(device)
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()
    
    def evaluate_action(self, state, action):
        """评估动作的对数概率和熵"""
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
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
    """PPO智能体 - 修改为支持离散动作空间"""
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # 创建网络
        self.actor = PPOActor(state_dim, action_dim).to(device)
        self.critic = PPOCritic(state_dim).to(device)
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        print(f"PPO智能体初始化完成: 状态维度={state_dim}, 动作维度={action_dim}")
    
    def get_action(self, state):
        """获取动作"""
        return self.actor.get_action(state)
    
    def get_value(self, state):
        """获取状态价值"""
        state = torch.FloatTensor(state).to(device)
        return self.critic(state).item()
    
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
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        
        # 处理old_log_probs，确保是正确的形状
        old_log_probs_array = []
        for log_prob in old_log_probs:
            if isinstance(log_prob, torch.Tensor):
                old_log_probs_array.append(log_prob.item() if log_prob.numel() == 1 else log_prob.cpu().numpy())
            elif isinstance(log_prob, np.ndarray):
                old_log_probs_array.append(log_prob.item() if log_prob.size == 1 else log_prob)
            else:
                old_log_probs_array.append(float(log_prob))
        old_log_probs = torch.FloatTensor(old_log_probs_array).to(device)
        
        rewards = torch.FloatTensor(rewards).to(device)
        
        # 处理values，确保是正确的形状
        values_array = []
        for value in values:
            if isinstance(value, torch.Tensor):
                values_array.append(value.item() if value.numel() == 1 else value.cpu().numpy())
            elif isinstance(value, np.ndarray):
                values_array.append(value.item() if value.size == 1 else value)
            else:
                values_array.append(float(value))
        values = torch.FloatTensor(values_array).to(device)
        
        dones = torch.FloatTensor(dones).to(device)
        
        # 计算优势
        next_values = self.critic(states[-1:]).squeeze()
        advantages = self.compute_gae(rewards, values, next_values, dones)
        advantages = torch.FloatTensor(advantages).to(device)
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
        checkpoint = torch.load(filepath, map_location=device)
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
    print("开始PPO训练...")
    
    # 创建环境
    env = EnvClutterEnv(
        render_mode='human' if args.render else None,
        use_discrete_action=True,  # 使用离散动作空间
        control_mode="pd_ee_pose",  # 使用绝对位姿控制
        num_envs=args.num_envs
    )
    
    print(f"环境创建完成: {env}")
    
    # 获取环境信息
    obs, _ = env.reset()
    
    # 检查是否使用离散动作空间
    discrete_action_space = env.discrete_action_space
    if discrete_action_space is not None:
        action_dim = discrete_action_space.n
        print(f"使用离散动作空间，动作维度: {action_dim}")
    else:
        # 连续动作空间 - pd_ee_pose控制器通常是6维 (x, y, z, rx, ry, rz)
        action_dim = 6
        print(f"使用连续动作空间，动作维度: {action_dim}")
    
    # 提取状态特征
    state = flatten_obs(obs)
    state_dim = len(state)
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 创建PPO智能体
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef
    )
    
    # 创建日志记录器
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 训练统计
    episode_rewards = deque(maxlen=100)
    episode_success_rates = deque(maxlen=100)
    
    total_steps = 0
    episode_count = 0
    
    print("开始训练循环...")
    
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        epoch_rewards = []
        epoch_success = []
        
        # 收集经验
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        for step in tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}"):
            state = flatten_obs(obs)
            
            # 选择动作
            if discrete_action_space is not None:
                # 离散动作
                action, log_prob = agent.get_action(state)
            else:
                # 连续动作 - 这里需要修改为支持连续动作
                # 暂时使用随机动作作为占位符
                action = np.random.uniform(-0.1, 0.1, 6)
                log_prob = 0.0
            
            # 获取状态价值
            value = agent.get_value(state)
            
            # 执行动作
            try:
                next_obs, reward, done, truncated, info = env.step(action)
                
                # 记录数据
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                dones.append(done or truncated)
                
                # 更新观测
                obs = next_obs
                total_steps += 1
                
                # 记录奖励
                if isinstance(reward, (int, float)):
                    epoch_rewards.append(reward)
                elif hasattr(reward, 'item'):
                    epoch_rewards.append(reward.item())
                else:
                    epoch_rewards.append(float(reward))
                
                # 记录成功率
                if 'success' in info:
                    epoch_success.append(info['success'])
                else:
                    epoch_success.append(False)
                
                # 如果环境结束，重置
                if done or truncated:
                    obs, _ = env.reset()
                    episode_count += 1
                    
                    if len(epoch_rewards) > 0:
                        episode_rewards.append(sum(epoch_rewards[-100:]) / min(len(epoch_rewards), 100))
                    if len(epoch_success) > 0:
                        episode_success_rates.append(sum(epoch_success[-100:]) / min(len(epoch_success), 100))
                
            except Exception as e:
                print(f"步骤 {step} 执行失败: {e}")
                # 重置环境
                obs, _ = env.reset()
                continue
        
        # 更新智能体
        if len(states) > 0:
            try:
                actor_loss, critic_loss = agent.update(
                    states, actions, log_probs, rewards, values, dones, 
                    epochs=args.ppo_epochs
                )
                
                # 记录日志
                avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
                success_rate = np.mean(epoch_success) if epoch_success else 0
                
                writer.add_scalar('Loss/Actor', actor_loss, epoch)
                writer.add_scalar('Loss/Critic', critic_loss, epoch)
                writer.add_scalar('Reward/Average', avg_reward, epoch)
                writer.add_scalar('Success/Rate', success_rate, epoch)
                writer.add_scalar('Training/TotalSteps', total_steps, epoch)
                
                # 打印进度
                if epoch % args.log_interval == 0:
                    print(f"Epoch {epoch:4d} | "
                          f"平均奖励: {avg_reward:7.2f} | "
                          f"成功率: {success_rate:6.2%} | "
                          f"Actor损失: {actor_loss:8.4f} | "
                          f"Critic损失: {critic_loss:8.4f} | "
                          f"总步数: {total_steps}")
                
                # 保存模型
                if epoch % args.save_interval == 0 and epoch > 0:
                    model_path = os.path.join(args.model_dir, f'model_epoch_{epoch}.pt')
                    agent.save(model_path)
                    print(f"模型已保存到: {model_path}")
                
            except Exception as e:
                print(f"更新智能体时出错: {e}")
                continue
    
    # 训练完成
    print("训练完成！")
    
    # 保存最终模型
    final_model_path = os.path.join(args.model_dir, 'final_model.pt')
    agent.save(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # 关闭日志记录器
    writer.close()
    
    return agent

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
    
    # PPO超参数
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Actor学习率')
    parser.add_argument('--lr_critic', type=float, default=3e-4, help='Critic学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO剪切参数')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='熵系数')
    parser.add_argument('--value_coef', type=float, default=0.5, help='价值函数系数')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='PPO更新轮数')
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 开始训练
    train_ppo(args)

if __name__ == "__main__":
    main() 