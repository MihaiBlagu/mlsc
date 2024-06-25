import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_unbalanced_disk

seed = 46
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


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
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.99,
                 epsilon_min=0.01, batch_size=64, buffer_size=5000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
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
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions)  
        
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())



# Environment 
env = gym_unbalanced_disk.UnbalancedDisk(dt=0.025, umax=3.) # umax is the highest voltage
state_size = env.observation_space.shape[0]

# Discrete action space
n_discrete_actions = 11
action_space = np.linspace(-3, 3, n_discrete_actions)

# DQN agent
agent = DQNAgent(state_size, n_discrete_actions)

# class CustomReward:
#     def __init__(self, max_steps):
#         self.max_steps = max_steps

#     def angle_weight(self, step):
#         # Linear decay: starts from 1 and goes to 0
#         return 1 #- (step / self.max_steps)

#     def velocity_weight(self, step):
#         # Linear growth: starts from 0 and goes to 1
#         return step / self.max_steps

#     def reward(self, th, th_dot, step):
#         angle_reward = 0

#         if step <200:
#             angle_reward = np.exp(-((th % (2 * np.pi)) - np.pi) ** 2 / (2 * (np.pi / 14) ** 2))
#         if step >= 200 and step <400:
#             angle_reward = np.exp(-((th % (2 * np.pi)) - np.pi - np.pi/18) ** 2 / (2 * (np.pi / 14) ** 2))
#         if step >= 400 and step <600:
#             angle_reward = np.exp(-((th % (2 * np.pi)) - np.pi + np.pi/18) ** 2 / (2 * (np.pi / 14) ** 2))

#         velocity_penalty = -0.001 * th_dot ** 2
        
#         weight_angle = self.angle_weight(step)
#         weight_velocity = self.velocity_weight(step)
        
#         return  angle_reward + weight_velocity * velocity_penalty
        


# class CustomReward:
#     def __init__(self, max_steps):
#         self.max_steps = max_steps
#         self.target_angles = [np.pi, np.pi - np.pi/18, np.pi + np.pi/18]
#         self.target_weights = [0.5, 1, 1.5]  
        
#     def reward(self, th, th_dot, step):
#         angle_rewards = []
#         for target_angle, weight in zip(self.target_angles, self.target_weights):
#             angle_rewards.append(np.exp(-((th % (2 * np.pi)) - target_angle) ** 2 / (2 * (np.pi / 14) ** 2)) * weight)
        
#         return sum(angle_rewards)

# class CustomReward:
#     def __init__(self, max_steps):
#         self.max_steps = max_steps

#     def reward(self, th, th_dot, step):
#         # Define target angles
#         target_angles = [np.pi, np.pi + np.pi/18, np.pi - np.pi/18]
        
#         def target_weight(step, start, mid, end):
#             if step < start:
#                 return 0.0
#             elif step < mid:
#                 return (step - start) / (mid - start)
#             elif step < end:
#                 return 1.0 - (step - mid) / (end - mid)
#             else:
#                 return 0.0
        
#         rewards = [
#             np.exp(-((th % (2 * np.pi)) - target_angle) ** 2 / (2 * (np.pi / 7) ** 2))
#             for target_angle in target_angles
#         ]
        
#         weights = [
#             target_weight(step, 0, 100, 200),  # First target
#             target_weight(step, 100, 300, 400),  # Second target
#             target_weight(step, 300, 500, 600)  # Third target
#         ]
        
#         weights = np.array(weights, dtype=np.float64)
        
#         weights_sum = weights.sum()
#         if weights_sum == 0:
#             weights_sum = 1.0
        
#         weights /= weights_sum
        
#         total_reward = sum(w * r for w, r in zip(weights, rewards))
        
#         # Ensure the reward is a finite value
#         if not np.isfinite(total_reward):
#             total_reward = 0.0
        
#         return total_reward 

class CustomReward:
    def __init__(self, max_steps):
        self.max_steps = max_steps

    def reward(self, th, th_dot, step):
        # Target pos
        target_angles = [np.pi, np.pi + np.pi/18, np.pi - np.pi/18]
        
        # Weighting function for each target
        def target_weight(step, start, mid, end):
            if step < start:
                return 0.0
            elif step < mid:
                return (step - start) / (mid - start)
            elif step < end:
                return 1.0 - (step - mid) / (end - mid)
            else:
                return 0.0
        
        # Reward for each target pos
        rewards = [
            np.exp(-((th % (2 * np.pi)) - target_angle) ** 2 / (2 * (np.pi / 7) ** 2))
            for target_angle in target_angles
        ]
        
       
        weights = [
            target_weight(step, 0, 100, 200),    
            target_weight(step, 100, 300, 400),  
            target_weight(step, 300, 500, 600)   
        ]
        
        # weights normalization
        weights = np.array(weights, dtype=np.float64)
        weights_sum = weights.sum()
        if weights_sum == 0:
            weights_sum = 1.0
        weights /= weights_sum
        
        # Velocity penalty
        velocity_penalty = -0.001 * th_dot ** 2
        
        # Velocity weight
        if step < 200:
            weight_velocity = step / 200
        else:
            weight_velocity = 1.0 - (step - 200) / 200
        
        # Total reward
        total_reward = sum(w * r for w, r in zip(weights, rewards)) + weight_velocity * velocity_penalty

        if not np.isfinite(total_reward):
            total_reward = 0.0
        
        return total_reward
    
episodes = 300
max_steps = 600
custom_reward = CustomReward(max_steps)
rewards = []

for episode in range(1, episodes + 1):
    state, info = env.reset()
    total_reward = 0
    for step in range(max_steps):
        action_index = agent.select_action(state)
        action = action_space[action_index]
        next_state, reward, done, truncated, _ = env.step(action)

        th = state[0]  
        th_dot = state[1]  
        custom_reward_value = custom_reward.reward(th, th_dot, step)
        
        agent.memorize(state, action_index, custom_reward_value, next_state, done or truncated)
        agent.replay()
        state = next_state
        total_reward += custom_reward_value

        if (step + 1) % 200 == 0:
            agent.epsilon = agent.epsilon_start

        if done or truncated:
            break
    agent.update_target_network()
    rewards.append(total_reward)
    print(f"Episode: {episode}, Total Reward: {total_reward}")

model_name='models/multitarget_original_reward_3targets_epsilonstart.pth'
print(model_name)
torch.save(agent.q_network.state_dict(), model_name)

# Plotting the rewards
plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Reward over Episodes')
plt.show()