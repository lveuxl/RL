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

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs

# Hyperparameters  
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


exp_name = None
seed: int = 1
torch_deterministic: bool = True
cuda: bool = True
track: bool = False
wandb_project_name: str = "ManiSkill"
wandb_group: str = "SAC"
capture_video: bool = True
save_trajectory: bool = False
save_model: bool = True
evaluate: bool = False
checkpoint = None
log_freq: int = 1000

env_id: str = "PickCube-v1"
obs_mode: str = "rgb"
include_state: bool = True
env_vectorization: str = "cuda"
num_envs: int = 64 #256
num_eval_envs: int = 8
partial_reset: bool = False
eval_partial_reset: bool = False
num_steps: int = 50
num_eval_steps: int = 50
reconfiguration_freq: int = None
eval_reconfiguration_freq: int = 1
eval_freq: int = 10000
save_train_video_freq: int = None
control_mode: str = "pd_ee_delta_pos"
render_mode: str = "all"

total_timesteps: int = 400_000#1000000
buffer_size: int = 300_000
buffer_device: str = "cuda"
gamma: float = 0.8
tau: float = 0.01
batch_size: int = 512
learning_starts: int = 4000
policy_lr: float = 3e-4
q_lr: float = 3e-4
policy_frequency: int = 1
target_network_frequency: int = 1
alpha: float = 0.2
autotune: bool = True
training_freq: int = 64
utd: float = 0.5
bootstrap_at_done: str = "always"
camera_width: int = 64
camera_height: int = 64

grad_steps_per_iteration: int = 0
steps_per_env: int = 0


# Replay buffer 
class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (
                        torch.float32 if v.dtype in (np.float32, np.float64)
                        else torch.uint8 if v.dtype == np.uint8
                        else torch.int16 if v.dtype == np.int16
                        else torch.int32 if v.dtype == np.int32
                        else v.dtype
                    )
                    self.data[k] = torch.zeros(
                        buffer_shape + v.shape, dtype=dtype, device=device
                    )

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[: len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        env,
        num_envs: int,
        buffer_size: int,
        storage_device: torch.device,
        sample_device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs

        self.obs = DictArray(
            (self.per_env_buffer_size, num_envs),
            env.single_observation_space,
            device=storage_device,
        )
        self.next_obs = DictArray(
            (self.per_env_buffer_size, num_envs),
            env.single_observation_space,
            device=storage_device,
        )
        self.actions = torch.zeros(
            (self.per_env_buffer_size, num_envs) + env.single_action_space.shape,
            device=storage_device,
        )
        self.logprobs = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.values = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)

    def add(
        self,
        obs: dict,
        next_obs: dict,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        if self.storage_device == torch.device("cpu"):
            obs = {k: v.cpu() for k, v in obs.items()}
            next_obs = {k: v.cpu() for k, v in next_obs.items()}
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        # self.obs[self.pos] = obs.to(self.device)
        # self.next_obs[self.pos] = next_obs.to(self.device)
        # self.actions[self.pos] = action.to(self.device)
        # self.rewards[self.pos] = reward.to(self.device)
        # self.dones[self.pos] = done.to(self.device)

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size,))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size,))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,))

        obs_sample = self.obs[batch_inds, env_inds]
        next_obs_sample = self.next_obs[batch_inds, env_inds]

        # Move dict data to sample_device
        obs_sample = {k: v.to(self.sample_device) for k, v in obs_sample.items()}
        next_obs_sample = {k: v.to(self.sample_device) for k, v in next_obs_sample.items()}

        return ReplayBufferSample(
            obs=obs_sample,
            next_obs=next_obs_sample,
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )


# Nerual Network definitions 
def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


class PlainConv(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_dim=256,
        pool_feature_map=False,
        last_act=True,
        image_size=[128, 128],
    ):
        super().__init__()
        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4)
            if image_size[0] == 128 and image_size[1] == 128
            else nn.MaxPool2d(2, 2),  # [32, 32] or [16,16] etc.
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        if pool_feature_map:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=last_act)
        else:
            self.pool = None
            self.fc = make_mlp(64 * 4 * 4, [out_dim], last_act=last_act)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class EncoderObsWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, obs):
        if "rgb" in obs:
            rgb = obs["rgb"].float() / 255.0  # (B,H,W,3)
        if "depth" in obs:
            depth = obs["depth"].float()  # (B,H,W,1)

        if "rgb" and "depth" in obs:
            img = torch.cat([rgb, depth], dim=3)  # (B,H,W,C)
        elif "rgb" in obs:
            img = rgb
        elif "depth" in obs:
            img = depth
        else:
            raise ValueError("Observation dict must contain 'rgb' or 'depth'")

        
        img = img.permute(0, 3, 1, 2)
        return self.encoder(img)


class SoftQNetwork(nn.Module):
    def __init__(self, envs, encoder: EncoderObsWrapper):
        super().__init__()
        self.encoder = encoder
        action_dim = np.prod(envs.single_action_space.shape)
        state_dim = envs.single_observation_space["state"].shape[0]
        
        self.mlp = make_mlp(
            self.encoder.encoder.out_dim + action_dim + state_dim,
            [512, 256, 1],
            last_act=False,
        )

    def forward(self, obs, action, visual_feature=None, detach_encoder=False):
        if visual_feature is None:
            visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()

        x = torch.cat([visual_feature, obs["state"], action], dim=1)
        return self.mlp(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, envs, sample_obs):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape)
        state_dim = envs.single_observation_space["state"].shape[0]
        in_channels = 0

        # figure out image shape
        if "rgb" in sample_obs:
            in_channels += sample_obs["rgb"].shape[-1]
            image_size = sample_obs["rgb"].shape[1:3]
        if "depth" in sample_obs:
            in_channels += sample_obs["depth"].shape[-1]
            image_size = sample_obs["depth"].shape[1:3]

        self.encoder = EncoderObsWrapper(
            PlainConv(in_channels=in_channels, out_dim=256, image_size=image_size)
        )
        self.mlp = make_mlp(256 + state_dim, [512, 256], last_act=True)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

        # action rescaling
        self.action_scale = torch.FloatTensor(
            (envs.single_action_space.high - envs.single_action_space.low) / 2.0
        )
        self.action_bias = torch.FloatTensor(
            (envs.single_action_space.high + envs.single_action_space.low) / 2.0
        )

    def get_feature(self, obs, detach_encoder=False):
        visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        x = torch.cat([visual_feature, obs["state"]], dim=1)
        return self.mlp(x), visual_feature

    def forward(self, obs, detach_encoder=False):
        x, visual_feature = self.get_feature(obs, detach_encoder)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std, visual_feature

    def get_eval_action(self, obs):
        mean, log_std, _ = self(obs)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, obs, detach_encoder=False):
        mean, log_std, visual_feature = self(obs, detach_encoder)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, visual_feature

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class Logger:
    def __init__(self, log_wandb=False, tensorboard: Optional[SummaryWriter] = None):
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()



# Main code
if __name__ == "__main__":
    grad_steps_per_iteration = int(training_freq * utd)
    steps_per_env = training_freq // num_envs
    if exp_name is None:
        exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"
    else:
        run_name = exp_name

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    env_kwargs = dict(obs_mode=obs_mode, render_mode=render_mode, sim_backend="gpu", sensor_configs=dict())
    if control_mode is not None:
        env_kwargs["control_mode"] = control_mode
    if camera_width is not None:
        env_kwargs["sensor_configs"]["width"] = camera_width
    if camera_height is not None:
        env_kwargs["sensor_configs"]["height"] = camera_height

    envs = gym.make(
        env_id,
        num_envs=num_envs if not evaluate else 1,
        reconfiguration_freq=reconfiguration_freq,
        **env_kwargs,
    )
    eval_envs = gym.make(
        env_id,
        num_envs=num_eval_envs,
        reconfiguration_freq=eval_reconfiguration_freq,
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs,
    )

    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=include_state)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if capture_video or save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if evaluate:
            eval_output_dir = f"{os.path.dirname(checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // num_steps) % save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=num_steps,
                video_fps=30,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=save_trajectory,
            save_video=capture_video,
            trajectory_name="trajectory",
            max_steps_per_video=num_eval_steps,
            video_fps=30,
        )

    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=not partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, num_eval_envs, ignore_terminations=not eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not evaluate:
        print("Running training")
        if track:
            import wandb

            config = {}
            config["env_cfg"] = dict(
                **env_kwargs,
                num_envs=num_envs,
                env_id=env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=partial_reset,
            )
            config["eval_env_cfg"] = dict(
                **env_kwargs,
                num_envs=num_eval_envs,
                env_id=env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=False,
            )
            wandb.init(
                project=wandb_project_name,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=wandb_group,
                tags=["sac", "walltime_efficient"],
            )

        writer = SummaryWriter(f"runs/{run_name}")
        logger = Logger(log_wandb=track, tensorboard=writer)
    else:
        print("Running evaluation")

    envs.single_observation_space.dtype = np.float32

    expert_buffer_file = "expert_buffer_sparse_64_nf.pt"
    if os.path.exists(expert_buffer_file):
        print(f"Found replay buffer file {expert_buffer_file}. Loading with weights_only=False ...")
        rb = torch.load(expert_buffer_file, weights_only=False)
        rb.sample_device = device
        rb.storage_device=device

        print(rb.rewards[-10:])
    
    else:
        print("No expert buffer found, creating a new ReplayBuffer...")
        rb = ReplayBuffer(
            env=envs,
            num_envs=num_envs,
            buffer_size=buffer_size,
            storage_device=device, #torch.device(buffer_device),
            sample_device=device,
        )

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=seed)
    eval_obs, _ = eval_envs.reset(seed=seed)

    actor = Actor(envs, sample_obs=obs).to(device)
    qf1 = SoftQNetwork(envs, actor.encoder).to(device)
    qf2 = SoftQNetwork(envs, actor.encoder).to(device)
    qf1_target = SoftQNetwork(envs, actor.encoder).to(device)
    qf2_target = SoftQNetwork(envs, actor.encoder).to(device)

    # chkpt_file = "runs/PickCube-v1__sac_replay__1__1739111619/ckpt_240000.pt"
    # ckpt = torch.load(chkpt_file)
    # actor.load_state_dict(ckpt["actor"])
    # qf1.load_state_dict(ckpt["qf1"])
    # qf2.load_state_dict(ckpt["qf2"])  # watch for typos
    
    if checkpoint is not None:
        ckpt = torch.load(checkpoint)
        actor.load_state_dict(ckpt["actor"])
        qf1.load_state_dict(ckpt["qf1"])
        qf2.load_state_dict(ckf["qf2"])  # watch for typos
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(
        list(qf1.mlp.parameters()) + list(qf2.mlp.parameters()) + list(qf1.encoder.parameters()),
        lr=q_lr,
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=policy_lr)

    if autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=q_lr)
    else:
        alpha = alpha

    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = num_envs * steps_per_env
    pbar = tqdm.tqdm(range(total_timesteps))
    cumulative_times = defaultdict(float)

    while global_step < total_timesteps:
        if eval_freq > 0 and (global_step - training_freq) // eval_freq < global_step // eval_freq:
            # evaluation
            actor.eval()
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(
                        actor.get_eval_action(eval_obs)
                    )
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)

            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)

            pbar.set_description(
                f"success_once: {eval_metrics_mean.get('success_once',0):.2f}, "
                f"return: {eval_metrics_mean.get('return',0):.2f}"
            )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if evaluate:
                break
            actor.train()

            if save_model:
                model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "qf1": qf1_target.state_dict(),
                        "qf2": qf2_target.state_dict(),
                        "log_alpha": log_alpha,
                    },
                    model_path,
                )
                print(f"model saved to {model_path}")

        # Collect samples
        rollout_time = time.perf_counter()
        for local_step in range(steps_per_env):
            global_step += 1 * num_envs
            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                actions, _, _, _ = actor.get_action(obs)
                actions = actions.detach()

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = {k: v.clone() for k, v in next_obs.items()}

            # handle final_info logic
            if bootstrap_at_done == "never":
                stop_bootstrap = truncations | terminations
            elif bootstrap_at_done == "always":
                stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool)
            else:
                stop_bootstrap = terminations

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                # If your environment does NOT provide final_observation, you might remove the next line
                # for k in real_next_obs.keys():
                #     real_next_obs[k][done_mask] = final_info["final_observation"][k][done_mask].clone()
                for k, v in final_info["episode"].items():
                    if logger is not None:
                        logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)
            obs = next_obs
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(num_envs * steps_per_env)

        # training
        if global_step < learning_starts:
            continue

        update_time = time.perf_counter()
        learning_has_started = True
        for local_update in range(grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(batch_size)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _, visual_feature = actor.get_action(data.next_obs)
                qf1_next_target = qf1_target(data.next_obs, next_state_actions, visual_feature)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions, visual_feature)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * gamma * min_qf_next_target.view(-1)

            # Critic update
            visual_feature = actor.encoder(data.obs)
            qf1_a_values = qf1(data.obs, data.actions, visual_feature).view(-1)
            qf2_a_values = qf2(data.obs, data.actions, visual_feature).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # Actor update
            if global_update % policy_frequency == 0:
                pi, log_pi, _, visual_feature = actor.get_action(data.obs)
                qf1_pi = qf1(data.obs, pi, visual_feature, detach_encoder=True)
                qf2_pi = qf2(data.obs, pi, visual_feature, detach_encoder=True)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                actor_loss = (alpha * log_pi - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Entropy tune
                if autotune:
                    with torch.no_grad():
                        _, log_pi2, _, _ = actor.get_action(data.obs)
                    alpha_loss = -(log_alpha.exp() * (log_pi2 + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # target net update
            if global_update % target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        if (global_step - training_freq) // log_freq < global_step // log_freq and logger is not None:
            logger.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            logger.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            logger.add_scalar("losses/alpha", alpha, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            logger.add_scalar(
                "time/total_rollout+update_time",
                cumulative_times["rollout_time"] + cumulative_times["update_time"],
                global_step,
            )
            if autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if not evaluate and save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(
            {
                "actor": actor.state_dict(),
                "qf1": qf1_target.state_dict(),
                "qf2": qf2_target.state_dict(),
                "log_alpha": log_alpha,
            },
            model_path,
        )
        print(f"model saved to {model_path}")
        if logger is not None:
            logger.close()

    envs.close()
