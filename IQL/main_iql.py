import numpy as np
import torch
import gym
import argparse
import os
import d4rl
from tqdm import trange
from coolname import generate_slug
import time
import json
from log import Logger

import utils
from utils import VideoRecorder, get_expert_traj, merge_trajectories, get_dataset_return
import IQL
from scipy.spatial import KDTree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

def squashing_func(distance, action_dim, beta=0.5, scale=1.0, no_action_dim=False):
    if no_action_dim:
        squashed_value = scale * np.exp(-beta * distance)
    else:
        squashed_value = scale * np.exp(-beta * distance/action_dim)
    
    return squashed_value
    
def rewarder(kd_tree, key, num_k, action_dim, beta, scale, no_action_dim=False):
    
    distance, _ = kd_tree.query(key, k=[num_k], workers=-1)
    reward = squashing_func(distance, action_dim, beta, scale, no_action_dim)
    return reward
        

def eval_policy(args, iter, video: VideoRecorder, logger: Logger, 
                policy, env_name, seed, mean, std, seed_offset=100, 
                eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    lengths = []
    returns = []
    avg_reward = 0.
    for _ in range(eval_episodes):
        # video.init(enabled=(args.save_video and _ == 0))
        state, done = eval_env.reset(), False
        # video.record(eval_env)
        steps = 0
        episode_return = 0
        while not done:
            state = (np.array(state).reshape(1, -1) - mean)/std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            # video.record(eval_env)
            avg_reward += reward
            episode_return += reward
            steps += 1
        lengths.append(steps)
        returns.append(episode_return)
        # video.save(f'eval_s{iter}_r{str(episode_return)}.mp4')

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    logger.log('eval/offline lengths_mean', np.mean(lengths), iter)
    logger.log('eval/offline returns_mean', np.mean(returns), iter)
    logger.log('eval/offline d4rl_score', d4rl_score, iter)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="IQL")               # Policy name
    parser.add_argument("--env", default="halfcheetah-medium-v2")        # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true", default=False)        # Save model and optimizer parameters
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument("--normalize", default=False, action='store_true')
    parser.add_argument("--k", default=1, type=int)                 # how many nearest neighbors are needed
    parser.add_argument("--beta", default=0.5, type=float)                      # coefficient in distance
    parser.add_argument("--scale", default=1.0, type=float)                    # scale of the reward function
    # IQL
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--temperature", default=3.0, type=float)
    parser.add_argument("--expectile", default=0.7, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    # Work dir
    parser.add_argument('--work_dir', default='tmp', type=str)
    parser.add_argument('--expl_noise', default=0.2, type=float)
    parser.add_argument('--mode', default='sas', type=str) # different modes of search, support sas, sa, ss
    parser.add_argument("--no_action_dim", action="store_true", default=False)     # whether to involve action dimension
    parser.add_argument("--dropout_rate", default=None)
    parser.add_argument("--bias", default=1.0, type=float)
    args = parser.parse_args()
    args.cooldir = generate_slug(2)

    # Build work dir
    base_dir = 'runs'
    utils.make_dir(base_dir)
    base_dir = os.path.join(base_dir, args.work_dir)
    utils.make_dir(base_dir)
    args.work_dir = os.path.join(base_dir, args.env)
    utils.make_dir(args.work_dir)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d-%H:%M", ts)
    exp_name = str(args.env) + '-' + ts + '-bs' + str(args.batch_size) + '-s' + str(args.seed)
    if args.policy == 'IQL':
        exp_name += '-t' + str(args.temperature) + '-e' + str(args.expectile)
    else:
        raise NotImplementedError
    exp_name += '-' + args.cooldir
    args.work_dir = args.work_dir + '/' + exp_name
    utils.make_dir(args.work_dir)

    args.model_dir = os.path.join(args.work_dir, 'model')
    utils.make_dir(args.model_dir)
    args.video_dir = os.path.join(args.work_dir, 'video')
    utils.make_dir(args.video_dir)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    utils.snapshot_src('.', os.path.join(args.work_dir, 'src'), '.gitignore')

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    dataset = d4rl.qlearning_dataset(env)
    replay_buffer.convert_D4RL(dataset)
    
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1
    
    data = get_expert_traj(args.env, env, dataset, num_top_episodes=1)
    data = merge_trajectories(data)
    
    if args.mode == 'sas':
        data = np.hstack([data[0], data[1], data[5]])  # stack state and action, and next state
        kd_tree = KDTree(data)
        # query every sample
        key = np.hstack([replay_buffer.state, replay_buffer.action, replay_buffer.next_state])
    elif args.mode == 'sa':
        data = np.hstack([data[0], data[1]])  # stack state and action
        kd_tree = KDTree(data)
        # query every sample
        key = np.hstack([replay_buffer.state, replay_buffer.action])
    elif args.mode == 'ss':
        data = np.hstack([data[0], data[5]])  # stack state and next state
        kd_tree = KDTree(data)
        # query every sample
        key = np.hstack([replay_buffer.state, replay_buffer.next_state])

    reward = rewarder(kd_tree, key, args.k, action_dim, args.beta, args.scale, args.no_action_dim)
    
    replay_buffer.reward = reward

    # adding reward bias
    if 'antmaze' in args.env:
        replay_buffer.reward -= args.bias * args.scale

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        # IQL
        "discount": args.discount,
        "tau": args.tau,
        "temperature": args.temperature,
        "expectile": args.expectile,
        "dropout_rate": float(args.dropout_rate) if args.dropout_rate is not None else None,
    }

    # Initialize policy
    if args.policy == 'IQL':
        policy = IQL.IQL(data, **kwargs)
    else:
        raise NotImplementedError


    logger = Logger(args.work_dir, use_tb=True)
    video = VideoRecorder(dir_name=args.video_dir)

    for t in trange(int(args.max_timesteps)):
        policy.train(replay_buffer, args.batch_size, logger=logger)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            eval_episodes = 100 if t+1 == int(args.max_timesteps) and 'antmaze' in args.env else args.eval_episodes
            d4rl_score = eval_policy(args, t+1, video, logger, policy, args.env,
                                     args.seed, mean, std, eval_episodes=eval_episodes)
            if args.save_model:
                policy.save(args.model_dir)