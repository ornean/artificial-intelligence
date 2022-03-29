import collections
import os
import random
from argparse import ArgumentParser

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d


# ---------------------------------------------------------------------------------------------------------------------
# Util methods
# ---------------------------------------------------------------------------------------------------------------------
def arguments():
    parser = ArgumentParser()
    parser.add_argument('--env', default='LunarLanderContinuous-v2')
    return parser.parse_args()


def save(agent, rewards, args):
    path = './runs/{}-{}'.format(args.env, "lastrun")
    try:
        os.makedirs(path)
    except:
        pass

    torch.save(agent.q.state_dict(), os.path.join(path, 'model_state_dict'))

    plt.cla()
    plt.plot(rewards, c='r', alpha=0.3)
    plt.plot(gaussian_filter1d(rewards, sigma=5), c='r', label='Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    plt.title('Branching DDQN: {}'.format(args.env))
    plt.savefig(os.path.join(path, 'reward.png'))

    pd.DataFrame(rewards, columns=['Reward']).to_csv(os.path.join(path, 'rewards.csv'), index=False)


def init_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------------------------------------------------
# StateDictHelper
# ---------------------------------------------------------------------------------------------------------------------

class StateDictHelper():
    """Modify an existing state_dict by applying an arbitrary lambda expression or one of the given convince methods"""

    @staticmethod
    def apply(state_dict, func=lambda x: x):
        state_dict_new = collections.OrderedDict()
        for k in state_dict.keys():
            state_dict_new[k] = func(state_dict[k])
        return state_dict_new

    @staticmethod
    def mul(state_dict, value):
        return StateDictHelper.apply(state_dict, lambda x: x * value)

    @staticmethod
    def add(state_dict, value):
        return StateDictHelper.apply(state_dict, lambda x: x+value)

    @staticmethod
    def add_sd(state_dict, other_state_dict):
        state_dict_new = collections.OrderedDict()
        for k in state_dict.keys():
            state_dict_new[k] = state_dict[k] + other_state_dict[k]
        return state_dict_new

    @staticmethod
    def print(state_dict, title, break_after=2):
        count = 0
        print("# ----------------------------------------------------------------------------------")
        print("#", title)
        print("# ----------------------------------------------------------------------------------")
        for k in state_dict.keys():
            count = count + 1
            print(k, ":")
            for l in state_dict[k]:
                print("list shape =", l.shape, ", content = ", end="")
                if len(l.shape) == 0:
                    print(l, end="")
                else:
                    for e in l:
                        print(e, " : ", end="")
                print()
            if count >= break_after:
                break
        print()

# ---------------------------------------------------------------------------------------------------------------------
# BranchingQNetwork
# ---------------------------------------------------------------------------------------------------------------------
class BranchingQNetwork(nn.Module):

    def __init__(self, observation_size, action_dim, n):
        super().__init__()

        self.ac_dim = action_dim
        self.n = n  # number of bins for discretised action (same in each dim/branch)

        self.model = nn.Sequential(nn.Linear(observation_size, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(128, n) for i in range(action_dim)])

    def forward(self, x):
        out = self.model(x)
        value = self.value_head(out)
        advs = torch.stack([l(out) for l in self.adv_heads], dim=1)
        # shape of [l(out) for l in self.adv_heads] is action_dim * torch.Size([observation_size, n])
        # print(advs.shape) # torch.Size([observation_size, action_dim, n])
        # print(advs.mean(2).shape) # torch.Size([observation_size, action_dim])
        # print(advs.mean(2, keepdim=True).shape) # torch.Size([observation_size, action_dim, 1])
        q_val = value.unsqueeze(2) + advs - advs.mean(2, keepdim=True)
        # print(q_val.shape) # torch.Size([observation_size, action_dim]) # torch.Size([observation_size, action_dim,n])

        return q_val

    def soft_update(self, other_model: "BranchingQNetwork", tau: float = None):
        """ Updates model parameters using a weighted sum with parameters of other_model controlled by tau.
            Expected usage: should be called on target model, other_model being online model.
            θ_target = tau * θ_online + (1 - tau) * θ_target """
        if tau is None:
            self.load_state_dict(other_model.state_dict())
            return

        # weighted update
        target_ratio = StateDictHelper.mul(self.state_dict(), 1 - tau)
        online_ratio = StateDictHelper.mul(other_model.state_dict(), tau)
        new_target_sd = StateDictHelper.add_sd(target_ratio, online_ratio)
        self.load_state_dict(new_target_sd)


# ---------------------------------------------------------------------------------------------------------------------
# BranchingDQN agent, network, config
# ---------------------------------------------------------------------------------------------------------------------

class BranchingDQN(nn.Module):

    def __init__(self, obs, ac, config, n):
        super().__init__()

        self.q = BranchingQNetwork(obs, ac, n)
        self.target = BranchingQNetwork(obs, ac, n)

        self.target.soft_update(self.q, tau=None)
        self.target_net_update_tau = config.target_net_update_tau
        self.target_net_update_freq = config.target_net_update_freq
        self.update_counter = 0
        self.gamma = config.gamma

    def get_action(self, x):
        with torch.no_grad():
            # a = self.q(x).max(1)[1]
            out = self.q(x).squeeze(0)
            action = torch.argmax(out, dim=1)
        return action.numpy()

    def update_policy(self, adam, memory, params):
        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.sample(params.batch_size)

        states = torch.tensor(b_states).float()
        actions = torch.tensor(b_actions).long().reshape(states.shape[0], -1, 1)
        rewards = torch.tensor(b_rewards).float().reshape(-1, 1)
        next_states = torch.tensor(b_next_states).float()
        masks = torch.tensor(b_masks).float().reshape(-1, 1)

        qvals = self.q(states)
        current_q_values = qvals.gather(2, actions).squeeze(-1)

        with torch.no_grad():
            argmax = torch.argmax(self.q(next_states), dim=2)

            max_next_q_vals = self.target(next_states).gather(2, argmax.unsqueeze(2)).squeeze(-1)
            max_next_q_vals = max_next_q_vals.mean(1, keepdim=True).expand(-1, max_next_q_vals.shape[1])

        expected_q_vals = rewards + max_next_q_vals * self.gamma * masks

        loss = F.mse_loss(expected_q_vals, current_q_values)

        adam.zero_grad()
        loss.backward()

        for p in self.q.parameters():
            p.grad.data.clamp_(-1., 1.)
        adam.step()

        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0:
            self.update_counter = 0
            # self.target.load_state_dict(self.q.state_dict())
            self.target.soft_update(self.q, self.target_net_update_tau)


# ---------------------------------------------------------------------------------------------------------------------
# ExperienceReplayMemory
# ---------------------------------------------------------------------------------------------------------------------
class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):

        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for b in batch:
            states.append(b[0])
            actions.append(b[1])
            rewards.append(b[2])
            next_states.append(b[3])
            dones.append(b[4])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

# ---------------------------------------------------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------------------------------------------------
class AgentConfig:

    def __init__(self,
                 epsilon_start=1.,
                 epsilon_final=0.01,
                 epsilon_decay=800,
                 gamma=0.99,  # discount factor
                 lr=0.0001,  # learning rate
                 target_net_update_freq=300,
                 target_net_update_tau=.4,  # parameter for weighted sum in soft update of target model
                 memory_size=50000,  # replay buffer size
                 batch_size=64,
                 learning_starts=1000,  # model updates only if memory has at least this number of transitions
                 max_frames=1000000):  # number of frames over the whole training
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        # Exponential epsilon decay
        self.epsilon_by_frame = lambda i: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(
            -1. * i / self.epsilon_decay)

        self.gamma = gamma
        self.lr = lr

        self.target_net_update_freq = target_net_update_freq
        self.target_net_update_tau = target_net_update_tau
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.learning_starts = learning_starts
        self.max_frames = max_frames



# ---------------------------------------------------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------------------------------------------------

class TensorEnv(gym.Wrapper):

    def __init__(self, env_name):
        super().__init__(gym.make(env_name))

    def process(self, x):
        return torch.tensor(x).reshape(1, -1).float()

    def reset(self):
        return self.process(super().reset())

    def step(self, a):
        ns, r, done, infos = super().step(a)
        return self.process(ns), r, done, infos


class BranchingTensorEnv(TensorEnv):

    def __init__(self, env_name, n):
        super().__init__(env_name)
        self.n = n
        # In used environment, action values min and max are the same for all actions
        discretized_min = self.action_space.low[0]
        discretized_max = self.action_space.high[0]

        self.discretized = np.linspace(discretized_min, discretized_max, self.n)

    def step(self, a):
        #action = self.discretized[a]
        action = np.array([self.discretized[aa] for aa in a])
        return super().step(action)


if __name__ == "__main__":
    args = arguments()
    init_seeds(3600)

    bins = 6
    env = BranchingTensorEnv(args.env, bins)

    config = AgentConfig()
    memory = ExperienceReplayMemory(config.memory_size)

    action_size = env.action_space.shape[0] if type(env.action_space) == gym.spaces.box.Box else env.action_space.n
    agent = BranchingDQN(env.observation_space.shape[0], action_size, config, bins)
    adam = optim.Adam(agent.q.parameters(), lr=config.lr)

    s = env.reset()
    ep_reward = 0.
    recap = []  # stores cumulative reward per episode (episode rewards)

    p_bar = tqdm(total=config.max_frames)
    for frame in range(config.max_frames):

        epsilon = config.epsilon_by_frame(frame)

        if np.random.random() > epsilon:
            action = agent.get_action(s)
        else:
            action = np.random.randint(0, bins, size=env.action_space.shape[0])

        ns, r, done, infos = env.step(action)
        ep_reward += r

        if done:
            ns = env.reset()
            recap.append(ep_reward)
            p_bar.set_description('Rew: {:.3f}'.format(ep_reward))
            ep_reward = 0.

        memory.push((s.reshape(-1).numpy().tolist(), action, r, ns.reshape(-1).numpy().tolist(), 0. if done else 1.))
        s = ns

        p_bar.update(1)

        # Update online and target model only from specified number of frames
        if frame > config.learning_starts:
            agent.update_policy(adam, memory, config)

        if frame % 1000 == 0:
            save(agent, recap, args)

    p_bar.close()
