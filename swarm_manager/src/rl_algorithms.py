#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPO(nn.Module):
    """
    Proximal Policy Optimization sieć neuronowa.
    Zawiera actor (polityka) i critic (funkcja wartości).
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # Ogranicza wyjście do [-1, 1]
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Log standard deviation jako parametr uczący się
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        """Propagacja przez współdzieloną warstwę"""
        return self.shared(x)

    def act(self, x):
        """
        Zwraca średnią i odchylenie standardowe dla akcji.
        
        Returns:
            mean: Średnia akcji
            std: Odchylenie standardowe
        """
        features = self.forward(x)
        mean = self.actor(features)
        std = self.log_std.exp().clamp(min=0.01, max=1.0)  # Ograniczenie std
        
        # Zabezpieczenie przed NaN
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std = torch.nan_to_num(std, nan=0.1, posinf=1.0, neginf=0.01)
        
        return mean, std

    def evaluate(self, x):
        """Zwraca wartość stanu (value function)"""
        features = self.forward(x)
        return self.critic(features)


class PPOAgent:
    """
    Agent uczący się za pomocą PPO.
    """
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
        
        # Hiperparametry
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Model
        self.policy = PPO(input_dim, output_dim, hidden_dim=256).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Pamięć dla transitions
        self.memory = []
        
        # Statystyki treningowe
        self.training_iterations = 0

    def act(self, state):
        """
        Wybiera akcję na podstawie stanu.
        
        Args:
            state: Wektor stanu
            
        Returns:
            action: Wybrana akcja
            logprob: Log prawdopodobieństwa akcji
        """
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        state_t = torch.FloatTensor(state).to(self.device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        
        with torch.no_grad():
            mean, std = self.policy.act(state_t)
            
            # Rozszerz std jeśli potrzeba
            if std.dim() == 1:
                std = std.unsqueeze(0).expand_as(mean)
            
            # Stwórz rozkład normalny
            std = std.clamp(min=0.01) 
            dist = torch.distributions.Normal(mean, std)
            
            # Próbkuj akcję
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)  # Ogranicz akcje
            
            # Oblicz log prawdopodobieństwo
            logprob = dist.log_prob(action).sum(dim=1)
        
        return action.squeeze(0).cpu().numpy(), logprob.squeeze(0).cpu().item()

    def store_transition(self, state, action, logprob, reward, done):
        """Zapisz transition do pamięci"""
        self.memory.append((state, action, logprob, reward, done))

    def train(self):
        """
        Trenuj agenta na zebranych danych z pamięci.
        """
        if len(self.memory) == 0:
            return {
            'loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0
        }
        
        # Rozpakuj dane z pamięci
        states, actions, old_logprobs, rewards, dones = zip(*self.memory)
        
        # Konwertuj do tensorów
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(old_logprobs)).to(self.device)
        
        # Oblicz zdyskontowane nagrody (returns)
        returns = self._compute_returns(rewards, dones)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalizacja returns (pomaga w stabilności)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Trenuj przez k epok
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.k_epochs):
            # Forward pass przez sieć
            mean, std = self.policy.act(states)
            values = self.policy.evaluate(states).squeeze(-1)
            
            # Rozszerz std jeśli potrzeba
            if std.dim() == 1:
                std = std.unsqueeze(0).expand_as(mean)
            
            # Stwórz rozkład i oblicz nowe log prawdopodobieństwa
            dist = torch.distributions.Normal(mean, std)
            new_logprobs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1)
            
            # Oblicz advantages
            advantages = returns - values.detach()
            
            # Policy loss (PPO clip objective)
            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus (zachęca do eksploracji)
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = (
                policy_loss + 
                self.value_coef * value_loss + 
                self.entropy_coef * entropy_loss
            )
            
            # Backward pass i optymalizacja
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping dla stabilności
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), 
                self.max_grad_norm
            )
            
            self.optimizer.step()
            
            # Statystyki
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        # Statystyki średnie
        avg_loss = total_loss / self.k_epochs
        avg_policy_loss = total_policy_loss / self.k_epochs
        avg_value_loss = total_value_loss / self.k_epochs
        avg_entropy = total_entropy / self.k_epochs
        
        self.training_iterations += 1
        
        # Wyczyść pamięć
        self.memory.clear()
        
        return {
            'loss': avg_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy
        }

    def _compute_returns(self, rewards, dones):
        """
        Oblicza zdyskontowane nagrody (returns).
        
        Args:
            rewards: Lista nagród
            dones: Lista flag zakończenia
            
        Returns:
            Lista zdyskontowanych nagród
        """
        returns = []
        discounted_reward = 0.0
        
        # Oblicz od tyłu
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0.0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        return returns

    def get_statistics(self):
        """Zwraca statystyki agenta"""
        return {
            'training_iterations': self.training_iterations,
            'memory_size': len(self.memory),
            'device': str(self.device)
        }