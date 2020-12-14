from collections import deque
import numpy as np
import torch
import math


class MultiStepBuff:

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.reset()

    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def get(self, gamma=0.99):
        assert len(self.rewards) > 0
        state = self.states.popleft()
        action = self.actions.popleft()
        reward = self._nstep_return(gamma)
        return state, action, reward

    def _nstep_return(self, gamma):
        r = np.sum([r * (gamma ** i) for i, r in enumerate(self.rewards)])
        self.rewards.popleft()
        return r

    def reset(self):
        # Buffer to store n-step transitions.
        self.states = deque(maxlen=self.maxlen)
        self.actions = deque(maxlen=self.maxlen)
        self.rewards = deque(maxlen=self.maxlen)

    def is_empty(self):
        return len(self.rewards) == 0

    def is_full(self):
        return len(self.rewards) == self.maxlen

    def __len__(self):
        return len(self.rewards)


class LazyMemory(dict):

    def __init__(self, capacity, state_shape, device):
        super(LazyMemory, self).__init__()
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.device = device
        self.reset()

    def reset(self):
        self['state'] = []
        self['next_state'] = []
        self['action'] = np.array([{} for i in range(self.capacity)])
        self['reward'] = np.array([{} for i in range(self.capacity)])
        self['done'] = np.array([{} for i in range(self.capacity)])

        self._n = 0
        self._p = 0

    def append(self, state, action, reward, next_state, done,
               episode_done=None):
        self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        self['state'].append(state)
        self['next_state'].append(next_state)
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

        self.truncate()

    def truncate(self):
        while len(self['state']) > self.capacity:
            del self['state'][0]
            del self['next_state'][0]

    def sample(self, batch_size):
        indices0 = np.random.randint(low=0, high=len(self), size=math.floor(batch_size/2))
        indices1 = np.random.randint(low=0, high=len(self), size=math.ceil(batch_size/2))
        return self._sample(indices0, indices1, batch_size)

    def _sample(self, indices0, indices1, batch_size):
        bias = -self._p if self._n == self.capacity else 0

        states = np.empty(
            (batch_size, *self.state_shape), dtype=np.uint8)
        next_states = np.empty(
            (batch_size, *self.state_shape), dtype=np.uint8)
        actions = np.empty((batch_size, 1))
        rewards = np.empty((batch_size, 1))
        dones = np.empty((batch_size, 1))

        for i, index in enumerate(indices0):
            _index = np.mod(index+bias, self.capacity)
            states[i, ...] = self['state'][_index]['AGENT-0']
            next_states[i, ...] = self['next_state'][_index]['AGENT-0']
            actions[i][0] = self['action'][index]['AGENT-0']
            rewards[i][0] = self['reward'][index]['AGENT-0']
            dones[i][0] = self['done'][index]['AGENT-0']
        for i, index in enumerate(indices1):
            _index = np.mod(index + bias, self.capacity)
            states[i+len(indices0), ...] = self['state'][_index]['AGENT-1']
            next_states[i+len(indices0), ...] = self['next_state'][_index]['AGENT-1']
            actions[i+len(indices0)][0] = self['action'][index]['AGENT-1']
            rewards[i+len(indices0)][0] = self['reward'][index]['AGENT-1']
            dones[i+len(indices0)][0] = self['done'][index]['AGENT-1']

        states = torch.Tensor(states).to(self.device).float()
        next_states = torch.Tensor(next_states).to(self.device).float()
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._n


class LazyMultiStepMemory(LazyMemory):

    def __init__(self, capacity, state_shape, device, gamma=0.99,
                 multi_step=3):
        super(LazyMultiStepMemory, self).__init__(
            capacity, state_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    def append(self, state, action, reward, next_state, done):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if self.buff.is_full():
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done)

            if done:
                while not self.buff.is_empty():
                    state, action, reward = self.buff.get(self.gamma)
                    self._append(state, action, reward, next_state, done)
        else:
            self._append(state, action, reward, next_state, done)
