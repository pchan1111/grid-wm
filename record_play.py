import gymnasium
import argparse
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import copy
import colorama
import random
import json
import shutil
import pickle
import os

from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss


def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img = cv2.resize(img, (640, 640))
    return img


def build_single_env(env_name, image_size, use_native_resolution):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    # --- MODIFIED: Conditionally apply the ResizeObservation wrapper. ---
    if not use_native_resolution:
        env = gymnasium.wrappers.ResizeObservation(env, shape=(image_size, image_size))
    return env

def record_episode(env_name, image_size, num_episodes, output_path,
                   world_model: WorldModel, agent: agents.ActorCriticAgent, use_native_resolution: bool):
    
    print("Recording episode for env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)
    print(f"Collecting {num_episodes} episodes. Native resolution:" + colorama.Fore.YELLOW + f"{use_native_resolution}" + colorama.Style.RESET_ALL)
    print("Output video(s) will be based on: " + colorama.Fore.CYAN + f"{output_path}" + colorama.Style.RESET_ALL)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    world_model.eval()
    agent.eval()

    # Use a single environment
    env = build_single_env(env_name, image_size, use_native_resolution)
    
    all_rewards = []

    for episode_idx in tqdm(range(num_episodes)):
        
        # Reset environment and collectors for each episode
        current_obs, info = env.reset()
        sum_reward = 0
        
        obs_for_video = [process_visualize(current_obs)]
        
        context_obs = deque(maxlen=16)
        context_action = deque(maxlen=16)
        
        done = False
        truncated = False

        while not done and not truncated:
            with torch.no_grad():
                # If native resolution is used, the observation must be resized for the model.
                # Otherwise, the observation from the env is already the correct size.
                if use_native_resolution:
                    model_input_obs = cv2.resize(current_obs, (image_size, image_size), interpolation=cv2.INTER_AREA)
                else:
                    model_input_obs = current_obs

                if len(context_action) == 0:
                    action_int = env.action_space.sample()
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).cuda()
                    
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    
                    action_tensor = agent.sample(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=True
                    )
                    action_int = action_tensor.item()

                action_for_context = np.array([action_int])

            # The model's context always needs the resized observation.
            context_obs.append(rearrange(torch.Tensor(model_input_obs).unsqueeze(0).cuda(), "B H W C -> B 1 C H W") / 255)
            context_action.append(action_for_context)

            # The environment steps and returns the next observation (native or resized).
            obs, reward, done, truncated, info = env.step(action_int)
            
            # Save the observation from the environment directly for the video.
            obs_for_video.append(process_visualize(obs))
            sum_reward += reward
            current_obs = obs

        # Episode finished, save video
        print(f"Episode {episode_idx+1} finished with reward: {sum_reward}")
        all_rewards.append(sum_reward)
        
        # Use the extension from the provided output_path argument
        output_basename, output_ext = os.path.splitext(os.path.basename(output_path))
        if use_native_resolution:
            video_filename = f"{output_basename}_native_resolution_episode_{episode_idx}{output_ext}"
        else:
            video_filename = f"{output_basename}_episode_{episode_idx}{output_ext}"
        episode_video_path = os.path.join(output_dir, video_filename)
        
        # The video size is determined by the frame resolution (native or resized)
        frame_h, frame_w, _ = obs_for_video[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(episode_video_path, fourcc, 30.0, (frame_w, frame_h))
        for frame in obs_for_video:
            video_writer.write(frame)
        video_writer.release()
        print("Video for episode " + str(episode_idx+1) + " saved to: " + colorama.Fore.CYAN + f"{episode_video_path}" + colorama.Style.RESET_ALL)

    env.close()
    return np.mean(all_rewards) if all_rewards else 0


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-run_name", type=str, required=True)
    parser.add_argument("-output_path", type=str, required=True)
    parser.add_argument("-num_videos", type=int, required=True)
    # --- ADDED: New argument to choose resolution ---
    parser.add_argument("-native_resolution", action="store_true", help="If set, records in native resolution. Otherwise, records in model's image_size (e.g., 64x64).")
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)
    # print(colorama.Fore.RED + str(conf) + colorama.Style.RESET_ALL)

    # set seed
    seed_np_torch(seed=conf.BasicSettings.Seed)

    # build and load model/agent
    import train
    # This dummy env is now only used to get the action dimension, so it's fine.
    dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize, args.native_resolution)
    action_dim = dummy_env.action_space.n
    dummy_env.close()
    world_model = train.build_world_model(conf, action_dim)
    agent = train.build_agent(conf, action_dim)
    root_path = f"ckpt/{args.run_name}"

    import glob
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    steps = steps[-1:]
    print(f"Evaluating checkpoint for step: {steps}")
    results = []
    for step in tqdm(steps):
        world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
        agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
        # # eval
        episode_avg_return = record_episode(
            env_name=args.env_name,
            image_size=conf.BasicSettings.ImageSize, 
            num_episodes=args.num_videos, 
            output_path=args.output_path,
            world_model=world_model,
            agent=agent,
            use_native_resolution=args.native_resolution # Pass the flag
        )
        results.append([step, episode_avg_return])
