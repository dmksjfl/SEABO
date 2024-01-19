import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils
import TD3_BC
from tensorboardX import SummaryWriter

from utils import get_expert_traj, merge_trajectories, get_dataset_return

from scipy.spatial import KDTree

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


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1,-1) - mean)/std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="TD3_BC")               # Policy name
    parser.add_argument("--env", default="hopper-medium-v2")        # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    # TD3
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    # TD3 + BC
    parser.add_argument("--alpha", default=2.5)
    parser.add_argument("--normalize", default=True)
    # seabo
    parser.add_argument("--k", default=1, type=int)                 # how many nearest neighbors are needed
    parser.add_argument("--beta", default=0.5, type=float)          # coefficient in distance
    parser.add_argument("--scale", default=1.0, type=float)         # scale of the reward function
    parser.add_argument('--mode', default='sas', type=str)          # different modes of search, support sas, sa, ss
    parser.add_argument("--no_action_dim", action="store_true", default=False)     # whether to involve action dimension
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    outdir = args.dir
    writer = SummaryWriter('{}/tb'.format(outdir))
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        # TD3 + BC
        "alpha": args.alpha
    }

    # Initialize policy
    policy = TD3_BC.TD3_BC(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    dataset = d4rl.qlearning_dataset(env)
    replay_buffer.convert_D4RL(dataset)

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

    if args.normalize:
        mean,std = replay_buffer.normalize_states() 
    else:
        mean,std = 0,1
    
    evaluations = []
    for t in range(int(args.max_timesteps)):
        policy.train(replay_buffer, args.batch_size)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t+1}")
            eval_return = eval_policy(policy, args.env, args.seed, mean, std)
            writer.add_scalar('test return', eval_return, global_step = t+1)
            # np.save(f"./results/{file_name}", evaluations)
    if args.save_model: policy.save(f"./models/{file_name}")
