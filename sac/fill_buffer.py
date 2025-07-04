import os
import torch
import gymnasium as gym
import numpy as np
import argparse

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from sac_rgbd_base import Actor, ReplayBuffer, Args

def get_winner_envs(success_buffer, args):

    success_stack = torch.stack(success_buffer) #shape [50, num_envs]

    win_env_ids = []

    for cur_env_num in range(args.num_envs):
        env_success_traj = success_stack[:,cur_env_num]
        last_n_success = env_success_traj[-10:]
        num_success = last_n_success.sum()

        if num_success > 4:
            win_env_ids.append(cur_env_num)
            # print(f"Env num: {cur_env_num} | Success")

    return win_env_ids


def del_failed_envs(data_list, win_env_ids):
    assert len(data_list) == 50

    if isinstance(data_list[0], dict):
        print(f"This is an oberstavtin--- {data_list[0]['rgb'].shape, data_list[0]['state'].shape}")
        dict_keys = list(data_list[0].keys())
        # print(dict_keys)

        win_env_ids = torch.tensor(win_env_ids).to(device)

        for step_num in range(50):
            state_data = data_list[step_num][dict_keys[0]] #shape [num_envs, 29]
            rgb_data = data_list[step_num][dict_keys[1]] #shape [num_envs, 128, 128, 3]

            state_data = torch.index_select(state_data, dim=0, index=win_env_ids)
            rgb_data = torch.index_select(rgb_data, dim=0, index=win_env_ids)

            data_list[step_num][dict_keys[0]] = state_data
            data_list[step_num][dict_keys[1]] = rgb_data

        assert len(data_list[0]['rgb']) == len(data_list[0]['state']) == len(win_env_ids)

    else:

        for step_num in range(50):
            assert len(data_list[step_num]) == args.num_envs

            # for cur_env_num in range(len(data_list[step_num])):
            #     if cur_env_num not in win_env_ids:

            #         del data_list[cur_env_num]
            data_list[step_num] = [env for i, env in enumerate(data_list[step_num]) if i in win_env_ids]

        assert len(data_list[0]) == len(win_env_ids)


    return data_list


def collect_data(args, checkpoint_path, eval_env, replay_buffer, output_buffer_file):


    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Initialize actor and load weights
    obs, _ = eval_env.reset()
    actor = Actor(eval_env, sample_obs=obs).to(device)

    try:
        # Load model strictly, will raise an error if mismatched
        actor.load_state_dict(checkpoint['actor'])
    except RuntimeError as e:
        print(f"Strict load failed with error: {e}")
        print("Attempting to load with non-strict mode to ignore mismatched layers.")
        actor.load_state_dict(checkpoint['actor'], strict=False)

    actor.eval()

    # Track data
    
    obs_buf = []
    next_obs_buf = []
    action_buf = []
    reward_buf = []
    done_buf = []
    elapsed_steps_buf = []
    success_buf=[]
    success_count = 0 
    fail_count = 0 #if the agent fails to succeed in 50 steps, it's a fail
    collected_steps = 0

    obs, _ = eval_env.reset()

    # Evaluation loop
    step_count = 0
    while step_count < args.total_timesteps:
    # for step_count in range(args.total_timesteps):
        with torch.no_grad():
            action = actor.get_eval_action(obs)
            
        next_obs, reward, done, trunc, infos = eval_env.step(action)
        step_count += 1

        real_next_obs = {k: v.clone() for k, v in next_obs.items()}

        info_keys = list(infos.keys())
        elapsed_steps = infos['elapsed_steps']
        success = infos['success']       

        # print(f"shape debug---- { len(done)}") #, len(real_next_obs['state']), real_next_obs['state'].shape}")

        obs_buf.append(obs)
        next_obs_buf.append(real_next_obs)
        action_buf.append(action)
        reward_buf.append(reward)
        done_buf.append(done)
        elapsed_steps_buf.append(elapsed_steps)
        success_buf.append(success)

        if elapsed_steps[0] == 50:
            # print(f"shape of obs buf {obs_buf[0].shape}")

            win_env_ids = get_winner_envs(success_buf, args)

            print(f"winners--- {len(win_env_ids)} {win_env_ids}")

            # delete failed env entries
            obs_buf = del_failed_envs(obs_buf, win_env_ids)
            next_obs_buf = del_failed_envs(next_obs_buf, win_env_ids)
            action_buf = del_failed_envs(action_buf, win_env_ids)
            reward_buf = del_failed_envs(reward_buf, win_env_ids)
            done_buf = del_failed_envs(done_buf, win_env_ids)
            success_buf = del_failed_envs(success_buf, win_env_ids)

            reward_stack = torch.tensor(reward_buf)

            assert reward_stack.shape[-1] == len(win_env_ids)
            print(f"---- {reward_stack.shape}")

            print("\n")
            # print(len(obs_buf[0]))

            # Add data to replay buffer
            for ep_step in range(50):
                ep_state_buf = obs_buf[ep_step]['state']
                ep_rgb_buf = obs_buf[ep_step]['rgb']

                ep_next_state_buf = next_obs_buf[ep_step]['state']
                ep_next_rgb_buf = next_obs_buf[ep_step]['rgb']

                ep_action_buf = action_buf[ep_step]
                ep_reward_buf = reward_buf[ep_step]
                ep_done_buf = done_buf[ep_step]

                print(ep_state_buf.shape, ep_rgb_buf.shape, ep_next_state_buf.shape, len(ep_action_buf), len(ep_reward_buf), len(ep_done_buf))
            
                for win_env_count in range(len(win_env_ids)):
                    win_state_buf = ep_state_buf[win_env_count,:]
                    win_rgb_buf = ep_rgb_buf[win_env_count,:]

                    win_obs_buf = {
                                    'state': win_state_buf,
                                    'rgb': win_rgb_buf
                    }

                    win_next_state_buf = ep_next_state_buf[win_env_count,:]
                    win_next_rgb_buf = ep_next_rgb_buf[win_env_count,:]
                    
                    win_next_obs_buf = {
                                    'state': win_next_state_buf,
                                    'rgb': win_next_rgb_buf
                    }

                    win_action_buf = ep_action_buf[win_env_count]
                    win_reward_buf = ep_reward_buf[win_env_count]
                    win_done_buf = ep_done_buf[win_env_count]
                    print(f"---- {len(ep_reward_buf), ep_reward_buf, win_reward_buf}")
                    print(f"Adding to buffer: state {win_state_buf.shape}, rgb {win_rgb_buf.shape}, action {win_action_buf.shape}, reward {win_reward_buf.shape}")

                    replay_buffer.add(win_obs_buf, win_next_obs_buf, win_action_buf, win_reward_buf, win_done_buf)

            print(f"current rb.pos: {replay_buffer.pos}")
            print("------------------------")


            obs_buf.clear()
            next_obs_buf.clear()
            action_buf.clear()
            reward_buf.clear()
            done_buf.clear()
            elapsed_steps_buf.clear()
            success_buf.clear()


        obs = next_obs
    
    
    # #save buffer as pt
    # # output_buffer_file = "expert_buffer_2.pt"
    # torch.save(replay_buffer, output_buffer_file)
    # print("Replay buffer saved.")
    print(replay_buffer.rewards.shape)
    print(replay_buffer.rewards)
    
    eval_env.close()

# def inspect_buffer(output_buffer_file):

    

    # print(f"\n\nLoading expert buffer from {output_buffer_file}...")
    # expert_rb = torch.load(output_buffer_file, weights_only=False)

    # print(f"Length of buffer {len(expert_rb.rewards)}")
    # print(expert_rb.rewards[-10:])

if __name__ == "__main__":
    args = Args(exp_name="sac_rgbd", env_id="PickCube-v1", num_envs = 2, obs_mode="rgb", cuda=True, total_timesteps=200, buffer_size = 200,
                control_mode="pd_ee_delta_pos"  # Ensure this matches your training config
                )

    #init env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Environment setup
    env_kwargs = dict(
        obs_mode=args.obs_mode, 
        render_mode=args.render_mode, 
        sim_backend="gpu",
        control_mode=args.control_mode,  # Ensuring control mode matches training
        sensor_configs=dict(
            width=64,  # Set desired width
            height=64  # Set desired height
            )
    )
    eval_env = gym.make(args.env_id, num_envs=args.num_envs, **env_kwargs)
    eval_env = FlattenRGBDObservationWrapper(eval_env, rgb=True, depth=False, state=args.include_state)
    eval_env = ManiSkillVectorEnv(eval_env, args.num_envs, ignore_terminations=True, auto_reset=True ,record_metrics=True) #set 'auto_reset = False' if you don't want the env to reset after reaching goal

    # Print action space to verify dimensions
    print(f"Action space during evaluation: {eval_env.single_action_space.shape}")

    #init replay buffer
    rb = ReplayBuffer(env=eval_env, num_envs=args.num_envs, buffer_size=args.buffer_size, 
                        storage_device=device, sample_device=device    
                        )


    # Specify your checkpoint path here
    checkpoint_path = "runs/PickCube-v1__sac_rgbd__1__1738602197/ckpt_250048.pt"
    output_buffer_file = "expert_buffer_2.pt"

    collect_data(args=args, checkpoint_path=checkpoint_path, eval_env=eval_env, replay_buffer=rb, output_buffer_file=output_buffer_file)

    # inspect_buffer(output_buffer_file)











