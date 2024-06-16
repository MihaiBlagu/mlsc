import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_unbalanced_disk

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.999, epsilon_min=0.01, batch_size=64, buffer_size=5000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def memorize(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()
        
    def replay(self):
        if len(self.buffer) < self.batch_size:
            return
        minibatch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Environment setup
env = gym_unbalanced_disk.UnbalancedDisk(dt=0.025, umax=3.) # umax is the highest voltage
state_size = env.observation_space.shape[0]

# Discretize the action space
n_discrete_actions = 11
action_space = np.linspace(env.action_space.low, env.action_space.high, n_discrete_actions)

# Initialize DQN agent
agent = DQNAgent(state_size, n_discrete_actions)

# Training loop
episodes = 100
max_steps = 500
rewards = []

for episode in range(1, episodes + 1):
    state, info = env.reset()
    total_reward = 0
    for step in range(max_steps):
        action_index = agent.select_action(state)
        action = action_space[action_index]
        next_state, reward, done, truncated, _ = env.step(action)
        agent.memorize(state, action_index, reward, next_state, done or truncated)
        agent.replay()
        state = next_state
        total_reward += reward
        if done or truncated:
            break
    agent.update_target_network()
    rewards.append(total_reward)
    print(f"Episode: {episode}, Total Reward: {total_reward}")

# Save the trained model
torch.save(agent.q_network.state_dict(), 'dqn_best_params_100.pth')

# Plotting the rewards
plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Reward over Episodes')
plt.show()
