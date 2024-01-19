import numpy as np
import torch

import os
import random
import imageio
import gym
from tqdm import trange
import pickle
import tqdm
import tree


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in range(len(observations)):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs

def qlearning_dataset_with_timeouts(env, dataset=None,
                                    terminate_on_end=False,
                                    disable_goal=True,
                                    **kwargs):
    if dataset is None:
        dataset = env.get_dataset()

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    realdone_ = []
    if "infos/goal" in dataset:
        if not disable_goal:
            dataset["observations"] = np.concatenate(
                [dataset["observations"], dataset['infos/goal']], axis=1)
        else:
            pass

    episode_step = 0
    for i in range(N - 1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i + 1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        realdone_bool = bool(dataset['terminals'][i])
        if "infos/goal" in dataset:
            final_timestep = True if (dataset['infos/goal'][i] !=
                                    dataset['infos/goal'][i + 1]).any() else False
        else:
            final_timestep = dataset['timeouts'][i]

        if i < N - 1:
            done_bool += final_timestep

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        realdone_.append(realdone_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_)[:],
        'terminals': np.array(done_)[:],
        'realterminals': np.array(realdone_)[:],
    }

def load_trajectories(name: str, env, dataset, fix_antmaze_timeout=True):
    if "antmaze" in name and fix_antmaze_timeout:
        dataset = qlearning_dataset_with_timeouts(env)
    
    dones_float = np.zeros_like(dataset['rewards'])

    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset['observations'][i + 1] -
                        dataset['next_observations'][i]
                        ) > 1e-6 or dataset['terminals'][i] == 1.0:
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    dones_float[-1] = 1

    if 'realterminals' in dataset:
        # We updated terminals in the dataset, but continue using
        # the old terminals for consistency with original IQL.
        masks = 1.0 - dataset['realterminals'].astype(np.float32)
    else:
        masks = 1.0 - dataset['terminals'].astype(np.float32)
    traj = split_into_trajectories(
        observations=dataset['observations'].astype(np.float32),
        actions=dataset['actions'].astype(np.float32),
        rewards=dataset['rewards'].astype(np.float32),
        masks=masks,
        dones_float=dones_float.astype(np.float32),
        next_observations=dataset['next_observations'].astype(np.float32))
    return traj

def get_expert_traj(name: str, env, dataset, num_top_episodes=10):
    """Load expert demonstrations."""
    # Load trajectories from the given dataset
    trajs = load_trajectories(name, env, dataset)
    if num_top_episodes < 0:
        print("Loading the entire dataset as demonstrations")
        return trajs

    if 'antmaze' in name:
        returns = [sum([t[2] for t in traj]) / (1e-4 + np.linalg.norm(traj[0][0][:2])) for traj in trajs]
    else:
        returns = [sum([t[2] for t in traj]) for traj in trajs]
    idx = np.argpartition(returns, -num_top_episodes)[-num_top_episodes:]
    
    return [trajs[i] for i in idx]

def get_dataset_return(name, env, dataset):
    trajs = load_trajectories(name, env, dataset)
    episode_return = []
    for transition in trajs:
        N = len(transition)
        reward = 0
        for i in range(N):
            reward += transition[i][2]
        episode_return.append(reward)
    print(episode_return)
    print(np.max(np.array(episode_return)))
    print(np.min(np.array(episode_return)))

def merge_trajectories(trajs):
    flat = []
    for traj in trajs:
        for transition in traj:
            flat.append(transition)
    return tree.map_structure(lambda *xs: np.stack(xs), *flat)


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean)/std
        self.next_state = (self.next_state - mean)/std
        return mean, std


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def snapshot_src(src, target, exclude_from):
    make_dir(target)
    os.system(f"rsync -rv --exclude-from={exclude_from} {src} {target}")


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
