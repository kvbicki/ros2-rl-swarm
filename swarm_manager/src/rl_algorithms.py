#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPO(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def act(self, x):
        x = self.forward(x)
        mean = self.actor(x)
        std = self.log_std.exp()
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        return mean, std

    def evaluate(self, x):
        x = self.forward(x)
        return self.critic(x)

class PPOAgent:
    def __init__(self, input_dim, output_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPO(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.memory = []

    def act(self, state):
        state_t = torch.FloatTensor(state).to(self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        with torch.no_grad():
            mean, std = self.policy.act(state_t)
            dist = torch.distributions.Normal(mean, std.unsqueeze(0) if std.dim() == 1 else std)
            action = dist.sample()
            logprob = dist.log_prob(action).sum(dim=1)
        return action.squeeze(0).cpu().numpy(), logprob.squeeze(0).cpu().item()

    def store_transition(self, state, action, logprob, reward, done):
        self.memory.append((state, action, logprob, reward, done))

    def train(self):
        if len(self.memory) == 0:
            return

        states, actions, old_logprobs, rewards, dones = zip(*self.memory)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(old_logprobs)).to(self.device)

        discounted_rewards = []
        dr = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                dr = 0.0
            dr = r + self.gamma * dr
            discounted_rewards.insert(0, dr)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        if discounted_rewards.numel() > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        for _ in range(self.k_epochs):
            mean, std = self.policy.act(states)
            std_batch = std.unsqueeze(0).expand_as(mean) if std.dim() == 1 else std
            dist = torch.distributions.Normal(mean, std_batch)
            new_logprobs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1)
            values = self.policy.evaluate(states).squeeze(-1)
            advantages = discounted_rewards - values
            ratio = torch.exp(new_logprobs - old_logprobs)
            s1 = ratio * advantages
            s2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = (-torch.min(s1, s2)).mean() + 0.5 * F.mse_loss(values, discounted_rewards) - 0.01 * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory.clear()
