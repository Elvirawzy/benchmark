"""
Decentralized SAC agent
"""
from torch.optim import Adam

from .sac_model import TwinnedQNetwork, CateoricalPolicy
import os
import numpy as np
import torch
from .sac_utils import update_params


class SacdAgent:

    def __init__(self, ids, obs_space, act_space, lr=0.0003, target_entropy_ratio=0.98, dueling_net=False, cuda=True):
        self.agent_id = ids
        self.obs_space = obs_space
        self.act_space = act_space
        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        # Define networks.
        self.policy = CateoricalPolicy(self.obs_space[0], act_space.n).to(self.device)
        self.online_critic = TwinnedQNetwork(self.obs_space[0], act_space.n,
                                             dueling_net=dueling_net).to(device=self.device)
        self.target_critic = TwinnedQNetwork(self.obs_space[0], act_space.n,
                                             dueling_net=dueling_net).to(device=self.device).eval()

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # Disable gradient calculations of the target network.
        for param in self.target_critic.parameters():
            param.requires_grad = False

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = \
            -np.log(1.0 / act_space.n) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

    def explore(self, state):
        # Act with randomness.
        state = torch.Tensor(state[None, ...]).to(self.device).float()
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.item()

    def exploit(self, state):
        # Act without randomness.
        state = torch.Tensor(state[None, ...]).to(self.device).float()
        with torch.no_grad():
            action = self.policy.act(state)
        return action.item()

    def learn(self, batch, weights):
        assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and \
               hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        s = torch.tensor(batch['s_%d' % self.agent_id], dtype=torch.float32)
        a = torch.tensor(batch['a_%d' % self.agent_id], dtype=torch.float32)
        r = torch.tensor(batch['r_%d' % self.agent_id], dtype=torch.float32)
        s_next = torch.tensor(batch['s_next_%d' % self.agent_id], dtype=torch.float32)
        d = torch.tensor(batch['d_%d' % self.agent_id], dtype=torch.float32)

        # calc_critic_loss
        curr_q1, curr_q2 = self.calc_current_q(s, a)
        target_q = self.calc_target_q(r, s_next, d)
        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()
        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        # calc_policy_loss
        policy_loss, entropies = self.calc_policy_loss(s, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)
        update_params(self.policy_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)
        self.alpha = self.log_alpha.exp()

        return q1_loss.detach().item(), q2_loss.detach().item(), policy_loss.detach().item(), \
               entropy_loss.detach().item(), self.alpha.detach().item(), mean_q1, mean_q2, \
               entropies.detach().mean().item()

    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())

    def calc_current_q(self, states, actions):
        curr_q1, curr_q2 = self.online_critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                    torch.min(next_q1, next_q2) - self.alpha * log_action_probs
            )).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_policy_loss(self, states, weights):
        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))
