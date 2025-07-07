import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import tqdm
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical, Normal

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs
from stack_picking_env_example import StackPickingEnv  # 导入你的自定义环境

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_name = None
seed: int = 1
torch_deterministic: bool = True
cuda: bool = True
track: bool = False
wandb_project_name: str = "ManiSkill"
wandb_group: str = "PPO"
capture_video: bool = True
save_trajectory: bool = False
save_model: bool = True
evaluate: bool = False
checkpoint = None

env_id: str = "StackPicking-v1"  # 使用你的自定义环境
obs_mode: str = "rgb"
include_state: bool = True
num_envs: int = 32
num_eval_envs: int = 4
num_steps: int = 100
num_eval_steps: int = 100
control_mode: str = "pd_ee_delta_pos"
render_mode: str = "all"

# PPO specific hyperparameters
total_timesteps: int = 500_000
learning_rate: float = 3e-4
num_steps_per_rollout: int = 2048
num_minibatches: int = 32
update_epochs: int = 10
norm_adv: bool = True
clip_coef: float = 0.2
clip_vloss: bool = True
ent_coef: float = 0.01
vf_coef: float = 0.5
max_grad_norm: float = 0.5
target_kl: Optional[float] = None
gamma: float = 0.99
gae_lambda: float = 0.95
camera_width: int = 128
camera_height: int = 128

# Network architectures
class PlainConv(nn.Module):
    """CNN encoder for RGB observations"""
    def __init__(self, in_channels=3, out_dim=256, image_size=[128, 128]):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *image_size)
            cnn_output_size = self.cnn(dummy_input).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size, out_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)

class EncoderObsWrapper(nn.Module):
    """Observation encoder that handles both RGB and state information"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
    def forward(self, obs):
        # Handle RGB observation
        rgb = obs["rgb"].float() / 255.0
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        
        # Transpose to (batch, channels, height, width)
        rgb = rgb.permute(0, 3, 1, 2)
        visual_feature = self.encoder(rgb)
        
        # Handle state information
        state_features = []
        if "state" in obs:
            state_features.append(obs["state"])
        
        # Add object-specific features
        for key, value in obs.items():
            if key.startswith("object_") and key.endswith(("_pos", "_dimensions", "_exposure", "_graspability")):
                if isinstance(value, torch.Tensor):
                    if value.dim() == 1:
                        state_features.append(value.unsqueeze(0))
                    else:
                        state_features.append(value)
                else:
                    state_features.append(torch.tensor([value], dtype=torch.float32))
        
        if state_features:
            state_tensor = torch.cat(state_features, dim=-1)
            return torch.cat([visual_feature, state_tensor], dim=-1)
        else:
            return visual_feature

class PPOAgent(nn.Module):
    """PPO Agent with both actor and critic networks"""
    def __init__(self, envs, sample_obs):
        super().__init__()
        
        # Visual encoder
        visual_encoder = PlainConv(
            in_channels=3,
            out_dim=256,
            image_size=[camera_height, camera_width]
        )
        self.encoder = EncoderObsWrapper(visual_encoder)
        
        # Get feature dimension
        with torch.no_grad():
            sample_features = self.encoder(sample_obs)
            feature_dim = sample_features.shape[-1]
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, np.prod(envs.single_action_space.shape)),
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
        # Action distribution parameters
        self.action_dim = np.prod(envs.single_action_space.shape)
        self.action_scale = torch.FloatTensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0)
        
    def get_features(self, obs):
        return self.encoder(obs)
    
    def get_value(self, obs):
        features = self.get_features(obs)
        return self.critic(features)
    
    def get_action_and_value(self, obs, action=None):
        features = self.get_features(obs)
        
        # Actor forward pass
        action_mean = self.actor(features)
        action_logstd = torch.zeros_like(action_mean)
        action_std = torch.exp(action_logstd)
        
        # Create action distribution
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        # Scale action to environment bounds
        scaled_action = torch.tanh(action) * self.action_scale.to(action.device) + self.action_bias.to(action.device)
        
        # Critic forward pass
        value = self.critic(features)
        
        return scaled_action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

class PPOBuffer:
    """PPO rollout buffer"""
    def __init__(self, obs_space, action_space, num_envs, num_steps, device):
        self.obs = {key: torch.zeros((num_steps, num_envs) + space.shape, dtype=torch.float32, device=device) 
                    for key, space in obs_space.items()}
        self.actions = torch.zeros((num_steps, num_envs) + action_space.shape, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        
    def add(self, step, obs, action, logprob, reward, done, value):
        for key in self.obs:
            if key in obs:
                self.obs[key][step] = obs[key]
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.rewards[step] = reward
        self.dones[step] = done
        self.values[step] = value.flatten()
    
    def compute_returns_and_advantages(self, next_value, gamma, gae_lambda):
        """Compute returns and advantages using GAE"""
        advantages = torch.zeros_like(self.rewards)
        lastgaelam = 0
        
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - self.dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
                
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            
        returns = advantages + self.values
        return returns, advantages

def main():
    if exp_name is None:
        exp_name = f"{env_id}__ppo__{seed}__{int(time.time())}"
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    
    # Environment setup
    env_kwargs = dict(
        obs_mode=obs_mode,
        render_mode=render_mode,
        sim_backend="gpu",
        sensor_configs=dict(width=camera_width, height=camera_height),
        control_mode=control_mode,
        num_objects=5  # 自定义环境参数
    )
    
    envs = gym.make(env_id, num_envs=num_envs, **env_kwargs)
    eval_envs = gym.make(env_id, num_envs=num_eval_envs, **env_kwargs)
    
    # Apply wrappers
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=include_state)
    
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    
    # Video recording
    if capture_video:
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=f"runs/{exp_name}/videos",
            save_video=True,
            max_steps_per_video=num_eval_steps,
        )
    
    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=True, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, num_eval_envs, ignore_terminations=True, record_metrics=True)
    
    # Initialize agent
    obs, _ = envs.reset(seed=seed)
    agent = PPOAgent(envs, obs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    # Initialize buffer
    buffer = PPOBuffer(
        envs.single_observation_space,
        envs.single_action_space,
        num_envs,
        num_steps_per_rollout,
        device
    )
    
    # Logging
    writer = SummaryWriter(f"runs/{exp_name}")
    
    # Training loop
    global_step = 0
    num_updates = total_timesteps // (num_steps_per_rollout * num_envs)
    
    for update in range(1, num_updates + 1):
        # Rollout phase
        for step in range(num_steps_per_rollout):
            global_step += num_envs
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)
            
            next_obs, reward, done, truncated, info = envs.step(action)
            
            buffer.add(step, obs, action, logprob, reward, done, value)
            obs = next_obs
        
        # Compute returns and advantages
        with torch.no_grad():
            next_value = agent.get_value(obs).flatten()
            returns, advantages = buffer.compute_returns_and_advantages(
                next_value, gamma, gae_lambda
            )
        
        # Update phase
        batch_size = num_envs * num_steps_per_rollout
        minibatch_size = batch_size // num_minibatches
        
        for epoch in range(update_epochs):
            indices = torch.randperm(batch_size, device=device)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                # Prepare minibatch data
                mb_obs = {key: buffer.obs[key].reshape(-1, *buffer.obs[key].shape[2:])[mb_indices] 
                         for key in buffer.obs}
                mb_actions = buffer.actions.reshape(-1, *buffer.actions.shape[2:])[mb_indices]
                mb_logprobs = buffer.logprobs.reshape(-1)[mb_indices]
                mb_returns = returns.reshape(-1)[mb_indices]
                mb_advantages = advantages.reshape(-1)[mb_indices]
                mb_values = buffer.values.reshape(-1)[mb_indices]
                
                # Normalize advantages
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Forward pass
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                
                # Policy loss
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(newvalue - mb_values, -clip_coef, clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
        
        # Logging
        if update % 10 == 0:
            print(f"Update {update}/{num_updates}, Global Step: {global_step}")
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/total_loss", loss.item(), global_step)
        
        # Save model
        if save_model and update % 100 == 0:
            model_path = f"runs/{exp_name}/ckpt_{global_step}.pt"
            torch.save({
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
            }, model_path)
            print(f"Model saved to {model_path}")
    
    envs.close()
    eval_envs.close()
    writer.close()

if __name__ == "__main__":
    main() 