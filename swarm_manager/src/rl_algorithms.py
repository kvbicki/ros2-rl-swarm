#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Tuple, List

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(input_dim, output_dim).to(self.device)
        self.target_network = DQN(input_dim, output_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.output_dim = output_dim
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.output_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

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
    
    def act(self, state):
        x = self.forward(state)
        mean = self.actor(x)
        std = self.log_std.exp()
        return mean, std
    
    def evaluate(self, state):
        x = self.forward(state)
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
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mean, std = self.policy.act(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        return action.cpu().numpy(), action_logprob.cpu().numpy()
    
    def store_transition(self, state, action, logprob, reward, done):
        self.memory.append((state, action, logprob, reward, done))
    
    def train(self):
        states = []
        actions = []
        logprobs = []
        rewards = []
        dones = []
        
        for transition in self.memory:
            state, action, logprob, reward, done = transition
            states.append(state)
            actions.append(action)
            logprobs.append(logprob)
            rewards.append(reward)
            dones.append(done)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        logprobs = torch.FloatTensor(np.array(logprobs)).to(self.device)
        
        discounted_rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        for _ in range(self.k_epochs):
            mean, std = self.policy.act(states)
            dist = torch.distributions.Normal(mean, std)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            ratio = torch.exp(new_logprobs - logprobs)
            
            advantages = discounted_rewards - self.policy.evaluate(states).squeeze()
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) - 0.01 * entropy + 0.5 * F.mse_loss(self.policy.evaluate(states).squeeze(), discounted_rewards)
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.memory.clear()

class SAC(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

class SACCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SACAgent:
    def __init__(self, input_dim, output_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = SAC(input_dim, output_dim).to(self.device)
        self.critic1 = SACCritic(input_dim, output_dim).to(self.device)
        self.critic2 = SACCritic(input_dim, output_dim).to(self.device)
        self.target_critic1 = SACCritic(input_dim, output_dim).to(self.device)
        self.target_critic2 = SACCritic(input_dim, output_dim).to(self.device)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.memory = deque(maxlen=100000)
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
    def act(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.actor(state)
        std = log_std.exp()
        
        if evaluate:
            return mean.cpu().numpy()[0]
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        return torch.tanh(action).cpu().numpy()[0]
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=256):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_states)
            next_std = next_log_std.exp()
            next_dist = torch.distributions.Normal(next_mean, next_std)
            next_actions = next_dist.rsample()
            next_actions = torch.tanh(next_actions)
            
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            
            next_log_prob = next_dist.log_prob(next_actions).sum(dim=1, keepdim=True)
            next_log_prob -= (2 * (np.log(2) - next_actions - F.softplus(-2 * next_actions))).sum(dim=1, keepdim=True)
            
            target_q = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_log_prob)
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        mean, log_std = self.actor(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        actions_reparametrized = dist.rsample()
        actions_reparametrized = torch.tanh(actions_reparametrized)
        
        log_prob = dist.log_prob(actions_reparametrized).sum(dim=1, keepdim=True)
        log_prob -= (2 * (np.log(2) - actions_reparametrized - F.softplus(-2 * actions_reparametrized))).sum(dim=1, keepdim=True)
        
        q1 = self.critic1(states, actions_reparametrized)
        q2 = self.critic2(states, actions_reparametrized)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)