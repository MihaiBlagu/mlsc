import torch
import numpy as np
import gymnasium as gym
import gym_unbalanced_disk
import itertools
import os
import matplotlib.pyplot as plt

from actor_critic.model import ActorCritic
from actor_critic.utils import A2C_rollout
from actor_critic.discretizer import Discretizer

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

# set seeds for everything
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def train(gamma = 0.97, # 0.05 
    N_iterations=7, 
    N_rollout=20000, 
    N_epochs=7, 
    N_evals=10, 
    alpha_actor=0.3,
    alpha_entropy=0.6, #changing this will change how much entropy is weighted (decrease if Entropy=0)
    lr=0.005,
    num_actions=7,# 7
    hidden_size=40,
    curr_best=-float('inf')):

    # Environment setup
    env = gym_unbalanced_disk.UnbalancedDisk(dt=0.025, umax=3.) # umax is the highest voltage
    env = gym.wrappers.time_limit.TimeLimit(env,max_episode_steps=1000)

    actor_crit = ActorCritic(env, num_actions, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(actor_crit.parameters(), lr=lr) #low learning rate
    new_best, model, scores = A2C_rollout(actor_crit, optimizer, env, alpha_actor=alpha_actor, alpha_entropy=alpha_entropy,
                gamma=gamma, N_iterations=N_iterations, N_rollout=N_rollout, N_epochs=N_epochs, 
                N_evals=N_evals, best_score=curr_best)

    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward per Episode')
    plt.title('Training Progress')
    plt.grid(True)
    plt.savefig(f'./plots/ac_scores_{new_best}.png')
    plt.show()

    return new_best, model


def grid_search():
    gamma_values = [0.97, 0.98, 0.99]
    N_iterations_values = [7]
    N_rollout_values = [20000]
    N_epochs_values = [7]
    N_evals_values = [10]
    alpha_actor_values = [0.2, 0.3, 0.4]
    alpha_entropy_values = [0.4, 0.5, 0.6]
    lr_values = [0.005]
    num_actions_values = [7]
    hidden_size_values = [40]
    
    best_score = -float('inf')
    best_params = None
    
    # Ensure the models directory exists
    os.makedirs('./models', exist_ok=True)
    
    with open('./models/ac_best_params.txt', 'w') as f:
        f.write("Starting grid search...\n")
    
    # Generate all possible combinations of the hyperparameters
    param_combinations = list(itertools.product(
        gamma_values, N_iterations_values, N_rollout_values, N_epochs_values, N_evals_values, alpha_actor_values,
        alpha_entropy_values, lr_values, num_actions_values, hidden_size_values
    ))

    for params in param_combinations:
        gamma, N_iterations, N_rollout, N_epochs, N_evals, alpha_actor, alpha_entropy, lr, num_actions, hidden_size = params
        
        print(f"Testing parameters: gamma={gamma}, N_iterations={N_iterations}, N_rollout={N_rollout}, "
              f"N_epochs={N_epochs}, N_evals={N_evals}, alpha_actor={alpha_actor}, alpha_entropy={alpha_entropy}, "
              f"lr={lr}, num_actions={num_actions}, hidden_size={hidden_size}")

        score, model = train(
            gamma=gamma,
            N_iterations=N_iterations,
            N_rollout=N_rollout,
            N_epochs=N_epochs,
            N_evals=N_evals,
            alpha_actor=alpha_actor,
            alpha_entropy=alpha_entropy,
            lr=lr,
            num_actions=num_actions,
            hidden_size=hidden_size
        )

        if score > best_score:
            best_score = score
            best_params = params

            with open('./models/best_params.txt', 'w') as f:
                f.write(f"Best score: {best_score}\n")
                f.write(f"Parameters: gamma={gamma}, N_iterations={N_iterations}, N_rollout={N_rollout}, "
                        f"N_epochs={N_epochs}, N_evals={N_evals}, alpha_actor={alpha_actor}, alpha_entropy={alpha_entropy}, "
                        f"lr={lr}, num_actions={num_actions}, hidden_size={hidden_size}\n")
            
            # save model
            model_path = f'./models/ac_best_{best_score:.2f}'
            torch.save(model.state_dict(), model_path)
    
    return best_score, best_params


def eval_model(model='./models/actor'):
    pass


if __name__ == '__main__':
    train()
    # grid_search()
    # train_stable_baselines()