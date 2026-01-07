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
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        lr=3e-4, 
        gamma=0.99, 
        eps_clip=0.2, 
        k_epochs=10,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.policy = PPO(input_dim, output_dim, hidden_dim=256).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.memory = []
        
        self.training_iterations = 0

    def act(self, state):
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        state_t = torch.FloatTensor(state).to(self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        
        with torch.no_grad():
            mean, std = self.policy.act(state_t)
            
            if std.dim() == 1:
                std = std.unsqueeze(0).expand_as(mean)
            
            std = std.clamp(min=0.01) 
            dist = torch.distributions.Normal(mean, std)
            
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            
            logprob = dist.log_prob(action).sum(dim=1)
        
        return action.squeeze(0).cpu().numpy(), logprob.squeeze(0).cpu().item()

    def store_transition(self, state, action, logprob, reward, done):
        self.memory.append((state, action, logprob, reward, done))

    def train(self):
        if len(self.memory) == 0:
            return {
            'loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0
        }
        
        states, actions, old_logprobs, rewards, dones = zip(*self.memory)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(old_logprobs)).to(self.device)
        
        returns = self._compute_returns(rewards, dones)
        returns = torch.FloatTensor(returns).to(self.device)
        
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.k_epochs):
            mean, std = self.policy.act(states)
            values = self.policy.evaluate(states).squeeze(-1)
            
            if std.dim() == 1:
                std = std.unsqueeze(0).expand_as(mean)
            
            dist = torch.distributions.Normal(mean, std)
            new_logprobs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1)
            
            advantages = returns - values.detach()
            
            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values, returns)
            
            entropy_loss = -entropy.mean()
            
            loss = (
                policy_loss + 
                self.value_coef * value_loss + 
                self.entropy_coef * entropy_loss
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), 
                self.max_grad_norm
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        avg_loss = total_loss / self.k_epochs
        avg_policy_loss = total_policy_loss / self.k_epochs
        avg_value_loss = total_value_loss / self.k_epochs
        avg_entropy = total_entropy / self.k_epochs
        
        self.training_iterations += 1
        
        self.memory.clear()
        
        return {
            'loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy
        }

    def _compute_returns(self, rewards, dones):
        returns = []
        discounted_reward = 0.0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0.0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        return returns

    def get_statistics(self):
        return {
            'training_iterations': self.training_iterations,
            'memory_size': len(self.memory),
            'device': str(self.device)
        }
