import os
import time
import torch
import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from sac_rgbd_copy import Actor, Args  

def evaluate_model(args, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    env_kwargs = dict(
        obs_mode=args.obs_mode, 
        render_mode=args.render_mode, 
        sim_backend="gpu",
        control_mode=args.control_mode  
    )
    eval_env = gym.make(args.env_id, num_envs=1, **env_kwargs)
    eval_env = FlattenRGBDObservationWrapper(eval_env, rgb=True, depth=False, state=args.include_state)
    eval_env = ManiSkillVectorEnv(eval_env, 1, ignore_terminations=True, auto_reset=True ,record_metrics=True) #set 'auto_reset = False' if you don't want the env to reset after reaching goal

    # Set up video saving
    eval_output_dir = "./results" #f"runs/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}/videos"
    os.makedirs(eval_output_dir, exist_ok=True)
    eval_env = RecordEpisode(eval_env, output_dir=eval_output_dir, save_video=True, max_steps_per_video=args.num_eval_steps)

    # Print action space to verify dimensions
    print(f"Action space during evaluation: {eval_env.single_action_space.shape}")

    checkpoint = torch.load(checkpoint_path)

    obs, _ = eval_env.reset()
    actor = Actor(eval_env, sample_obs=obs).to(device)

    try:
        actor.load_state_dict(checkpoint['actor'])
    except RuntimeError as e:
        print(f"Strict load failed with error: {e}")
        print("Attempting to load with non-strict mode to ignore mismatched layers.")
        actor.load_state_dict(checkpoint['actor'], strict=False)

    actor.eval()

    # Evaluation loop
    obs, _ = eval_env.reset()
    for step in range(args.num_eval_steps):
        with torch.no_grad():
            action = actor.get_eval_action(obs)
            
        obs, reward, done, trunc, infos = eval_env.step(action)
        info_keys = list(infos.keys())
        # print(info_keys)
        # print(f"elapsed steps: {infos['elapsed_steps'].item()}   |   success: {infos['success'].item()} |  reward: {reward.item()}")
        if done.any():
            print("\n ---------------------------------------------------------------- \n")
            break

    print(f"Video saved to {eval_output_dir}")
    eval_env.close()

if __name__ == "__main__":
    args = Args(
        exp_name="sac_rgbd",
        env_id="PickCube-v1",
        obs_mode="rgb",
        cuda=True,
        evaluate=True,
        num_eval_steps=200,
        control_mode="pd_ee_delta_pos"  
    )

    checkpoint_path = "best_chkpts/dense_ckpt_440k_43.pt"
    evaluate_model(args, checkpoint_path)
