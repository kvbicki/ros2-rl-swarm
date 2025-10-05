#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import os

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def save_policy(policy, path='models/policy_net.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(policy.state_dict(), path)

def load_policy(policy, path='models/policy_net.pth'):
    if os.path.exists(path):
        policy.load_state_dict(torch.load(path))
