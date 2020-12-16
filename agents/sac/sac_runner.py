"""
Decentralized SAC agent
"""
from torch.optim import Adam
import gym

from .sac_model import TwinnedQNetwork, CateoricalPolicy
from torch.nn import functional as ff

from abc import ABC, abstractmethod
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from benchmark.agents.sac.sac_agent import SacdAgent

from .sac_memory import LazyMultiStepMemory
from .sac_utils import RunningMeanStats


class Runner(ABC):

    def __init__(self, obs_space, act_space, log_dir, scenario_path, n_agent, config, num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, dueling_net=False, num_eval_steps=125000, max_episode_steps=27000,
                 log_interval=10, eval_interval=1000, cuda=True):
        super().__init__()
        self.obs_space = self.get_obs_shape(obs_space.spaces)
        self.act_space = act_space
        self.config = config
        self.cuda = cuda
        self.n_agent = n_agent
        self.agent_ids = [f"AGENT-{i}" for i in range(n_agent)]
        self.scenario_path = scenario_path
        self.lr = lr
        self.dueling_net = dueling_net

        # Set seed.
        torch.manual_seed(0)
        np.random.seed(0)

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.env = self.make_env()
        self.agents = self.init_agents()
        # LazyMemory efficiently stores FrameStacked states.
        self.memory = LazyMultiStepMemory(
            n_agent=n_agent,
            capacity=memory_size,
            state_shape=self.obs_space,
            device=self.device, gamma=gamma, multi_step=multi_step)
        self.train_return = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.use_per = use_per
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps = max_episode_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval

    def run(self):
        while True:
            print(self.steps, self.num_steps)
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0 \
               and self.steps >= self.start_steps

    def train_episode(self):
        self.episodes += 1
        episode_return = [0., 0.]
        episode_steps = 0

        all_done = False
        flag = True
        while flag:
            try:
                state = self.env.reset()
                flag = False
            except:
                print("env reset err")
                self.env = self.make_env()
                print("re-init env")

        while (not all_done) and episode_steps <= self.max_episode_steps:

            action_list = {}
            processed_state = {}
            for i, agent_id in enumerate(state):
                processed_a_state = self.process_obs(state[agent_id], 0)
                processed_state[agent_id] = processed_a_state
                if self.start_steps > self.steps:
                    action = self.act_space.sample()
                else:
                    action = self.agents[i].explore(processed_a_state)
                action_list[agent_id] = action

            next_state, reward, done, _ = self.env.step(action_list)

            for i, agent_id in enumerate(reward):
                episode_return[i] += reward[agent_id]

            for i, agent_id in enumerate(done):
                all_done = (all_done or done[agent_id])

            processed_next_state = {}
            for i, agent_id in enumerate(next_state):
                processed_next_state[agent_id] = self.process_obs(next_state[agent_id], 0)

            self.memory.append(processed_state, action_list, reward, processed_next_state, done)

            self.steps += 1
            episode_steps += 1
            state = next_state

            if self.is_update():
                self.learn()

            if self.steps % self.target_update_interval == 0:
                for i in range(self.n_agent):
                    self.agents[i].update_target()

            if self.steps % self.eval_interval == 0:
                for i in range(self.n_agent):
                    self.agents[i].save_models(os.path.join(self.model_dir, str(i)))

        # We log running mean of training rewards.
        self.train_return.append(episode_return)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train_mean', self.train_return.get(2), self.steps)
            self.writer.add_scalar(
                'reward/train_agent1', self.train_return.get(0), self.steps)
            self.writer.add_scalar(
                'reward/train_agent2', self.train_return.get(1), self.steps)

        print(f'Episode: {self.episodes:<4}  '
              f'Episode steps: {episode_steps:<4}  '
              f'Return: {np.mean(episode_return):<5.1f}')

    def learn(self):
        self.learning_steps += 1

        if self.use_per:
            batch, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            weights = 1.

        q1_loss = q2_loss = policy_loss = entropy_loss = alpha = mean_q1 = mean_q2 = entropies = np.zeros(2)
        for i in range(self.n_agent):
            q1_loss[i], q2_loss[i], policy_loss[i], entropy_loss[i], alpha[i], mean_q1[i], mean_q2[i], entropies[i] = \
                self.agents[i].learn(batch, weights)

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', np.mean(q1_loss), self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', np.mean(q2_loss), self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', np.mean(policy_loss), self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', np.mean(entropy_loss), self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', np.mean(alpha), self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q1', np.mean(mean_q1), self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', np.mean(mean_q2), self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', np.mean(entropies), self.learning_steps)

    def evaluate(self):
        num_episodes = 0
        total_return = [0., 0.]
        num_steps = 0

        while True:
            state = self.env.reset()
            episode_steps = 0
            episode_return = [0., 0.]
            all_done = False
            while (not all_done) and episode_steps <= self.max_episode_steps:
                action_list = {}
                processed_state = {}
                for i, agent_id in enumerate(state):
                    processed_a_state = self.process_obs(state[agent_id], 0)
                    processed_state[agent_id] = processed_a_state
                    action = self.agents[i].exploit(processed_a_state)
                    action_list[agent_id] = action

                next_state, reward, done, _ = self.env.step(action_list)

                for i, agent_id in enumerate(reward):
                    episode_return[i] += reward[agent_id]

                for i, agent_id in enumerate(done):
                    all_done = (all_done or done[agent_id])

                num_steps += 1
                episode_steps += 1
                state = next_state

            num_episodes += 1
            total_return[0] += episode_return[0]
            total_return[1] += episode_return[1]

            if num_steps > self.num_eval_steps:
                break

        mean_return = np.mean(total_return / num_episodes)

        self.writer.add_scalar(
            'reward/test_mean', mean_return, self.steps)
        self.writer.add_scalar(
            'reward/test_agent1', total_return[0], self.steps)
        self.writer.add_scalar(
            'reward/test_agent2', total_return[1], self.steps)

        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)

    def make_env(self):
        agent_spec = {
            agent_id: AgentSpec(
                **self.config["agent"], interface=AgentInterface(**self.config["interface"])
            )
            for agent_id in self.agent_ids
        }
        env = gym.make(
            "smarts.env:hiway-v0",
            seed=42,
            scenarios=[str(self.scenario_path)],
            headless=False,
            agent_specs=agent_spec, )
        return env

    def init_agents(self):
        agents = []
        for i in range(self.n_agent):
            agent = SacdAgent(ids=i, obs_space=self.obs_space, act_space=self.act_space, lr=self.lr,
                              cuda=self.cuda)
            agents.append(agent)
        return agents

    @staticmethod
    def get_obs_shape(obs_space):
        dims = 0
        for o in obs_space.values():
            dims = dims + o.shape[0]
        return (dims,)

    def process_obs(self, obs, offset):
        if not isinstance(obs, OrderedDict):
            obs = OrderedDict(sorted(obs.items()))
        array_a_obs = np.zeros(self.obs_space[0])
        for o in obs.values():
            for lo in range(len(o)):
                array_a_obs[lo + offset] = o[lo]
            offset += len(o)
        return array_a_obs

    def __del__(self):
        self.writer.close()
