import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)  # 输出为资产权重（总和为1）
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

class PPOAgentContinuous:
    def __init__(self, state_dim, action_dim, action_asset_list, lr=3e-4, gamma=0.99):
        print("✅ PPOAgentContinuous: 初始化开始")
        self.policy = PPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_weight = 0.01
        self.action_asset_list = action_asset_list
        print("✅ PPOAgentContinuous: 初始化结束")

    def get_action(self, state):
        print("🚀 PPOAgent: 开始获取动作")
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.policy(state_tensor)
        return probs.squeeze().numpy()  # 返回资产权重向量

    def get_decision(self, market_state):
        feature_vector = market_state['feature_vector']
        rp_decision = market_state['rp_decision']
        cvar_decision = market_state['cvar_decision']
        ppo_input = np.concatenate([feature_vector, rp_decision, cvar_decision])
        probs = self.get_action(ppo_input)
        return pd.Series(probs, index=self.action_asset_list)

    def compute_advantage(self, rewards, values, masks):
        returns = []
        R = 0
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        advantages = returns - values
        return returns, advantages

    def train(self, trajectories, epochs=10):
        print("🎯 PPOAgentContinuous: 开始训练")
        states = torch.FloatTensor(np.array([t[0] for t in trajectories]))
        actions = torch.FloatTensor(np.array([t[1] for t in trajectories]))  # 连续动作向量
        rewards = [t[2] for t in trajectories]
        masks = torch.FloatTensor([t[3] for t in trajectories])

        probs, values = self.policy(states)
        values = values.squeeze()
        returns, advantages = self.compute_advantage(rewards, values.detach(), masks)
        advantages = advantages.detach()

        for epoch in range(epochs):
            new_probs, new_values = self.policy(states)
            policy_loss = ((new_probs - actions)**2).mean()  # MSE 损失
            value_loss = (returns - new_values.squeeze()).pow(2).mean()
            entropy = -torch.sum(new_probs * torch.log(new_probs + 1e-8), dim=1).mean()
            loss = policy_loss + 0.5 * value_loss - self.entropy_weight * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch+1}: PolicyLoss={policy_loss.item():.4f}, ValueLoss={value_loss.item():.4f}, Entropy={entropy.item():.4f}")
        print("🎯 PPOAgentContinuous: 训练结束")

    def get_confidence(self, market_state):
        feature_vector = market_state['feature_vector']
        rp_decision = market_state['rp_decision']
        cvar_decision = market_state['cvar_decision']
        ppo_input = np.concatenate([feature_vector, rp_decision, cvar_decision])
        state_tensor = torch.FloatTensor(ppo_input).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.policy(state_tensor)
        confidence_score = torch.max(probs).item()  # 最大资产权重，表示最强偏好
        return confidence_score
