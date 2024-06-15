import gymnasium as gym
import gym_unbalanced_disk
import time
import torch
import numpy as np
from torch import nn

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

# Load the environment
env = gym_unbalanced_disk.UnbalancedDisk(dt=0.025, umax=3.)

# Set up the model
state_size = env.observation_space.shape[0]
n_discrete_actions = 11
action_space = np.linspace(env.action_space.low, env.action_space.high, n_discrete_actions)

q_network = QNetwork(state_size, n_discrete_actions)
q_network.load_state_dict(torch.load("multitarget_agent.pth" )) # 'dqn_90gamma_100epochs.pth' 'dqn_best_params_100.pth'
q_network.eval()

# Function to select action using the trained model
def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(state)
    action_index = torch.argmax(q_values).item()
    return action_space[action_index]

# Run the environment with the trained model
env.render_mode = 'human'
obs, info = env.reset()
try:
    for i in range(200):
        action = select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs, reward)
        env.render()
        time.sleep(1/24)
        if terminated or truncated:
            obs, info = env.reset()
finally:
    env.close()
