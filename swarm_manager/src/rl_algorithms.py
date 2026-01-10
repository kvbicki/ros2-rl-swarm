#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPO(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        return self.shared(x)

    def act(self, x):
        features = self.forward(x)
        mean = self.actor(features)
        std = self.log_std.exp().clamp(min=0.01, max=1.0)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std = torch.nan_to_num(std, nan=0.1, posinf=1.0, neginf=0.01)
        return mean, std

    def evaluate(self, x):
        features = self.forward(x)
        return self.critic(features)

class PPOAgent:
    def __init__(self, input_dim, output_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=10, value_coef=0.5, entropy_coef=0.05, max_grad_norm=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.policy = PPO(input_dim, output_dim, hidden_dim=256).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []

    def act(self, state):
        state_t = torch.FloatTensor(state).to(self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        
        with torch.no_grad():
            mean, std = self.policy.act(state_t)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            logprob = dist.log_prob(action).sum(dim=1)
        
        return action.cpu().numpy(), logprob.cpu().numpy()

    def store_transition(self, state, action, logprob, reward, done):
        self.memory.append((state, action, logprob, reward, done))

    def train(self):
        if len(self.memory) == 0:
            return
        
        states, actions, old_logprobs, rewards, dones = zip(*self.memory)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(old_logprobs)).to(self.device)
        
        returns = self._compute_returns(rewards, dones)
        returns = torch.FloatTensor(returns).to(self.device)
        
        for _ in range(self.k_epochs):
            mean, std = self.policy.act(states)
            values = self.policy.evaluate(states).squeeze(-1)
            
            dist = torch.distributions.Normal(mean, std)
            new_logprobs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1)
            
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values, returns)
            entropy_loss = -entropy.mean()
            
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        self.memory.clear()

    def _compute_returns(self, rewards, dones):
        returns = []
        discounted_reward = 0.0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0.0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        return returns