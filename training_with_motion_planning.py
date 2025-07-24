"""
集成运动规划的训练脚本
RL智能体负责选择目标对象，运动规划器负责执行抓取动作
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env_clutter import EnvClutterEnv
from motion_planner import GraspPlanner, SimpleMotionPlanner, TrajectoryPlayer

class ObjectSelectionAgent:
    """
    对象选择智能体
    输入：环境状态
    输出：选择哪个对象进行抓取（离散动作）
    """
    
    def __init__(self, state_dim, num_objects, hidden_dim=256, lr=3e-4):
        self.state_dim = state_dim
        self.num_objects = num_objects
        self.hidden_dim = hidden_dim
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objects),
            nn.Softmax(dim=-1)
        )
        
        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # 训练参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        
    def get_action(self, state):
        """获取动作（选择对象）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item()
    
    def get_value(self, state):
        """获取状态价值"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            value = self.value_net(state_tensor)
            
        return value.item()
    
    def evaluate_action(self, state, action):
        """评估动作"""
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.LongTensor(action)
        
        action_probs = self.policy_net(state_tensor)
        dist = Categorical(action_probs)
        
        log_prob = dist.log_prob(action_tensor)
        entropy = dist.entropy()
        value = self.value_net(state_tensor)
        
        return log_prob, entropy, value
    
    def update(self, states, actions, old_log_probs, rewards, values, dones, epochs=10):
        """更新网络参数"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        dones = torch.BoolTensor(dones)
        
        # 计算优势函数
        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + values
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 更新网络
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(epochs):
            # 评估当前动作
            log_probs, entropy, new_values = self.evaluate_action(states, actions)
            
            # 计算比率
            ratio = torch.exp(log_probs - old_log_probs)
            
            # 计算策略损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 添加熵正则化
            policy_loss -= self.entropy_coef * entropy.mean()
            
            # 计算价值损失
            value_loss = nn.MSELoss()(new_values.squeeze(), returns)
            
            # 更新策略网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_optimizer.step()
            
            # 更新价值网络
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        return total_policy_loss / epochs, total_value_loss / epochs
    
    def compute_gae(self, rewards, values, dones):
        """计算广义优势估计"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            if dones[i]:
                next_value = 0
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[i])
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages)
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])

class HybridController:
    """
    混合控制器
    结合RL对象选择和运动规划执行
    """
    
    def __init__(self, env):
        self.env = env
        self.motion_planner = SimpleMotionPlanner(env)
        self.trajectory_player = TrajectoryPlayer(env)
        self.grasp_planner = GraspPlanner(env)
        
    def execute_grasp(self, selected_object_idx):
        """
        执行抓取动作
        
        Args:
            selected_object_idx: 选择的对象索引
            
        Returns:
            success: 是否成功
            reward: 执行奖励
        """
        try:
            # 获取选择的对象
            if selected_object_idx >= len(self.env.selectable_objects[0]):
                print(f"无效的对象索引: {selected_object_idx}")
                return False, -1.0
            
            target_object = self.env.selectable_objects[0][selected_object_idx]
            
            # 生成目标放置位置
            place_pos = np.array([0.1, 0.1, 0.2])  # 简单的固定位置
            
            # 执行抓取规划
            success = self.grasp_planner.plan_and_execute_grasp(target_object, place_pos)
            
            if success:
                print(f"成功抓取对象 {selected_object_idx}")
                return True, 10.0  # 成功奖励
            else:
                print(f"抓取对象 {selected_object_idx} 失败")
                return False, -1.0  # 失败惩罚
                
        except Exception as e:
            print(f"执行抓取时发生错误: {e}")
            return False, -1.0

def extract_object_features(env, obs):
    """
    提取对象特征用于RL决策
    
    Args:
        env: 环境实例
        obs: 观测数据
        
    Returns:
        features: 对象特征向量
    """
    features = []
    
    # 提取每个对象的特征
    for i, obj_info in enumerate(env.object_info[0]):  # 假设单环境
        # 对象位置
        obj_pos = obj_info['center']
        features.extend(obj_pos)
        
        # 对象大小
        obj_size = obj_info['size']
        features.extend(obj_size)
        
        # 暴露面积
        exposed_area = obj_info.get('exposed_area', 1.0)
        features.append(exposed_area)
        
        # 对象类型（one-hot编码）
        obj_type = obj_info['type']
        type_encoding = [0] * len(env.BOX_OBJECTS)
        if obj_type in env.BOX_OBJECTS:
            type_encoding[env.BOX_OBJECTS.index(obj_type)] = 1
        features.extend(type_encoding)
    
    # 添加机器人状态
    if isinstance(obs, dict) and 'agent' in obs:
        robot_state = obs['agent']
        if isinstance(robot_state, torch.Tensor):
            features.extend(robot_state.cpu().numpy().flatten())
        else:
            features.extend(np.array(robot_state).flatten())
    elif isinstance(obs, torch.Tensor):
        # 如果obs是张量，直接使用其数据
        features.extend(obs.cpu().numpy().flatten())
    
    return np.array(features, dtype=np.float32)

def train_hybrid_system(args):
    """训练混合系统"""
    print("初始化混合控制系统...")
    
    # 创建环境
    env = EnvClutterEnv(
        robot_uids="panda_suction",
        num_envs=1,  # 混合系统暂时只支持单环境
        obs_mode="state",
        control_mode="pd_joint_pos",  # 使用关节位置控制
        render_mode="human" if args.render else "rgb_array",
        sim_backend="auto"
    )
    
    print(f"环境创建完成: {env}")
    
    # 获取状态和动作维度
    obs, _ = env.reset()
    
    # 计算对象数量
    num_objects = len(env.selectable_objects[0])
    print(f"可选择对象数量: {num_objects}")
    
    # 提取特征
    features = extract_object_features(env, obs)
    state_dim = len(features)
    
    print(f"状态维度: {state_dim}, 对象数量: {num_objects}")
    
    # 创建智能体和控制器
    agent = ObjectSelectionAgent(state_dim, num_objects)
    controller = HybridController(env)
    
    # 创建日志记录器
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 训练循环
    episode_rewards = deque(maxlen=100)
    episode_success_rates = deque(maxlen=100)
    
    total_episodes = 0
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        # 收集数据
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_success = 0
        
        for step in range(args.steps_per_epoch):
            # 提取特征
            state = extract_object_features(env, obs)
            
            # 选择对象
            action, log_prob = agent.get_action(state)
            value = agent.get_value(state)
            
            print(f"步骤 {step}: 选择对象 {action}")
            
            # 执行抓取
            success, reward = controller.execute_grasp(action)
            
            # 检查episode是否结束
            done = success or step >= args.max_steps_per_episode
            
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
            
            # 渲染环境
            if args.render:
                env.render()
                time.sleep(0.1)
            
            # 打印步骤信息
            print(f"步骤 {step}: 奖励={reward:.3f}, 成功={success}, 完成={done}")
            
            # 如果episode结束
            if done:
                episode_rewards.append(episode_reward)
                episode_success_rates.append(episode_success)
                total_episodes += 1
                
                print(f"Episode {total_episodes} 结束: 奖励={episode_reward:.3f}, 成功={episode_success}")
                
                # 重置环境
                obs, _ = env.reset()
                episode_reward = 0
                episode_success = 0
                break
        
        # 更新智能体
        if len(states) > 0:
            print(f"更新智能体，数据量: {len(states)}")
            policy_loss, value_loss = agent.update(
                states, actions, log_probs, rewards, values, dones
            )
            
            # 记录日志
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_success_rate = np.mean(episode_success_rates) if episode_success_rates else 0
            
            writer.add_scalar('Training/Episode_Reward', avg_reward, epoch)
            writer.add_scalar('Training/Success_Rate', avg_success_rate, epoch)
            writer.add_scalar('Training/Policy_Loss', policy_loss, epoch)
            writer.add_scalar('Training/Value_Loss', value_loss, epoch)
            
            if epoch % args.log_interval == 0:
                print(f"Epoch {epoch}, "
                      f"平均奖励: {avg_reward:.2f}, "
                      f"成功率: {avg_success_rate:.2f}, "
                      f"策略损失: {policy_loss:.4f}, "
                      f"价值损失: {value_loss:.4f}")
        
        # 保存模型
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.model_dir, f"hybrid_model_epoch_{epoch}.pth")
            agent.save(save_path)
            print(f"模型已保存到: {save_path}")
    
    # 保存最终模型
    final_save_path = os.path.join(args.model_dir, "hybrid_model_final.pth")
    agent.save(final_save_path)
    print(f"最终模型已保存到: {final_save_path}")
    
    env.close()
    writer.close()

def main():
    parser = argparse.ArgumentParser(description='训练混合RL+运动规划系统')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--steps_per_epoch', type=int, default=10, help='每轮最大步数')
    parser.add_argument('--max_steps_per_episode', type=int, default=5, help='每个episode最大步数')
    parser.add_argument('--log_dir', type=str, default='./logs/hybrid_system', help='日志目录')
    parser.add_argument('--model_dir', type=str, default='./models/hybrid_system', help='模型保存目录')
    parser.add_argument('--log_interval', type=int, default=10, help='日志记录间隔')
    parser.add_argument('--save_interval', type=int, default=50, help='模型保存间隔')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 开始训练
    train_hybrid_system(args)

if __name__ == "__main__":
    main() 