import gymnasium as gym
import gym_unbalanced_disk
import time
import torch
import numpy as np
from actor_critic.model import ActorCritic

from stable_baselines3 import A2C

def select_action(actor_crit, obs):
    # Convert observation to a tensor with batch dimension
    obs = torch.tensor(obs[None, :], dtype=torch.float32)
    
    # Get action probabilities from the actor network
    with torch.no_grad():
        action_probs = actor_crit.actor(obs)[0].numpy()
    
    # Select the action with the highest probability
    action = np.argmax(action_probs)
    
    return action

def load_model(path, env, num_actions, hidden_size):
    model = ActorCritic(env, num_actions, hidden_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def sim_custom():
     # Load the environment
    env = gym_unbalanced_disk.UnbalancedDisk(dt=0.025, umax=3.)

    # Parameters (set according to the best found in grid search or manually specified)
    n_discrete_actions = 7
    hidden_size = 40 
    model_path = './models/actor-crit-checkpoint'

    # Load the trained model
    model = load_model(model_path, env, n_discrete_actions, hidden_size)

    # Run the environment with the trained model
    env.render_mode = 'human'
    obs, info = env.reset()
    try:
        for i in range(400):
            action = select_action(model, obs)
            # action = np.linspace(env.action_space.low, env.action_space.high, n_discrete_actions)[action] # continuous only
            obs, reward, terminated, truncated, info = env.step(action)
            print(obs, reward)
            env.render()
            time.sleep(1/24)
            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()

def sim_stable_baselines():
     # Load the environment
    env = gym_unbalanced_disk.UnbalancedDisk(dt=0.025, umax=3.)

    # Parameters (set according to the best found in grid search or manually specified)
    n_discrete_actions = 5
    model_path = './models/a2c_sb3_best_model.zip'

    # Load the trained model
    model = A2C.load(model_path)

    # Run the environment with the trained model
    env.render_mode = 'human'
    obs, info = env.reset()
    try:
        for i in range(400):
            action, states = model.predict(obs)
            action = np.linspace(env.action_space.low, env.action_space.high, n_discrete_actions)[int(action)] # continuous only
            obs, reward, terminated, truncated, info = env.step(action)
            print(obs, reward)
            env.render()
            time.sleep(1/24)
            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()


if __name__ == '__main__':
    sim_custom()
    # sim_stable_baselines()
