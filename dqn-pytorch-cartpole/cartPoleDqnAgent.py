import collections
import os
import random
from typing import Deque
import gym
import numpy as np
from cartPoleDqn import DQN
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

PROJECT_PATH = Path.cwd()
MODELS_PATH = PROJECT_PATH.joinpath("models")
ONLINE_MODEL_PATH = MODELS_PATH.joinpath("dqn_cartpole.h5")
TARGET_MODEL_PATH = MODELS_PATH.joinpath("target_dqn_cartpole.h5")
LOG_PATH = PROJECT_PATH.joinpath("logs")

MEAN_REWARD_STOP_THRESHOLD = 400  # Game won if mean of last rewards greater than this
EPISODE_REWARD_THRESHOLD_FOR_PENALTY = 499
MODEL_REWARD_PENALTY = 100.0


class Agent:
    def __init__(self, env: gym.Env, summary_writer: SummaryWriter):
        # Environment
        self.env = env
        # Experience replay buffer
        self.replay_buffer_size = 100_000  # 50_000
        self.train_start = 2_000  # 2_500                      # start train after n observations
        self.memory: Deque = collections.deque(  # memory double-ended-queue
            maxlen=self.replay_buffer_size
        )
        # Agent Parameters
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999  # 0.995
        self.batch_size = 32
        # DQN Parameters
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.hidden_size = 24
        self.learning_rate = 0.001
        # DQN Model
        self.dqn = DQN(
            input_size=self.observation_size,
            hidden_size=self.hidden_size,
            output_size=self.action_size,
            learning_rate=self.learning_rate
        )
        self.target_dqn = DQN(
            input_size=self.observation_size,
            hidden_size=self.hidden_size,
            output_size=self.action_size,
            learning_rate=self.learning_rate
        )
        self.summary_writer = summary_writer

    def get_action(self, state: np.ndarray):
        """ Select action for state with epsilon-greedy strategy """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.dqn(state).detach().numpy())

    def train(self, num_episodes: int):
        last_rewards: Deque = collections.deque(maxlen=4)  # rewards for last n observations
        best_reward_mean = 0.0  # best mean reward from n last observations

        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                # apply penalty on model when episode fails before total reward for episode threshold met
                if done and total_reward < EPISODE_REWARD_THRESHOLD_FOR_PENALTY:
                    reward = -MODEL_REWARD_PENALTY
                self.remember(state, action, reward, next_state, done)
                # Train from experiences if enough collected in experience replay buffer
                if len(self.memory) > self.train_start:
                    self.replay()
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay
                total_reward += reward
                state = next_state

                if done:
                    # compensate applied penalty to compute accurate total reward
                    if total_reward < (EPISODE_REWARD_THRESHOLD_FOR_PENALTY + 1):
                        total_reward += MODEL_REWARD_PENALTY

                    print(f"Episode: {episode} Reward: {total_reward} Epsilon: {self.epsilon}")

                    self.summary_writer.add_scalar("episode_reward", total_reward, episode)

                    last_rewards.append(total_reward)
                    current_reward_mean = np.mean(last_rewards)

                    # Update target model only if mean of last rewards greater than best reward mean so far
                    if current_reward_mean > best_reward_mean:
                        self.target_dqn.update_model(self.dqn)
                        best_reward_mean = current_reward_mean
                        self.dqn.save_model(ONLINE_MODEL_PATH)
                        self.target_dqn.save_model(TARGET_MODEL_PATH)
                        print(f"New best mean: {best_reward_mean}")

                        if best_reward_mean > MEAN_REWARD_STOP_THRESHOLD:
                            return
        print(f"Best reward mean: {best_reward_mean}")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_next, dones = zip(*minibatch)

        states = np.concatenate(states).astype(np.float32)
        states_next = np.concatenate(states_next).astype(np.float32)

        q_values = self.dqn(states)
        with torch.no_grad():
            q_values_next = self.target_dqn(states_next)

        for i in range(self.batch_size):
            a = actions[i]
            done = dones[i]
            if done:
                q_values[i][a] = rewards[i]
            else:
                q_values[i][a] = rewards[i] + self.gamma * max(q_values_next[i])

        self.dqn.fit(states, q_values)

    def play(self, num_episodes: int, render: bool = True):
        self.dqn.load_model(ONLINE_MODEL_PATH)
        self.target_dqn.load_model(TARGET_MODEL_PATH)

        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)

            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                total_reward += reward
                state = next_state
                if done:
                    print(f"Episode: {episode} Reward: {total_reward}")
                    break


def init_seeds():
    seed = 600
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    init_seeds()
    env = gym.make("CartPole-v1")

    summary_writer = SummaryWriter(log_dir=str(LOG_PATH.joinpath("pytorch-cartpole")))

    agent = Agent(env, summary_writer)
    agent.train(num_episodes=400)

    input("Play?")
    agent.play(num_episodes=10, render=True)
