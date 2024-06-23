import gym
import gym.wrappers
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gym_unbalanced_disk
import itertools
import logging
import time


def train_baselines(env, lr, gamma, gae_lambda, ent_coef, vf_coef, n_steps, total_timesteps=25000):
    model = A2C('MlpPolicy', 
                env, 
                verbose=1,
                n_steps=n_steps,
                learning_rate=lr, 
                gamma=gamma, 
                # gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                seed=42)
    
    model.learn(total_timesteps=total_timesteps)
    
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=2)

    return model, mean_reward


def grid_search_baselines(env, param_grid, total_timesteps=25000):
    best_score = float('-inf')
    best_model = None
    best_params = None

    param_names = param_grid.keys()
    param_values = param_grid.values()

    for param_combination in itertools.product(*param_values):
        params = dict(zip(param_names, param_combination))
        lr = params['lr']
        gamma = params['gamma']
        # gae_lambda = params['gae_lambda']
        ent_coef = params['ent_coef']
        vf_coef = params['vf_coef']
        n_steps = params['n_steps']

        # model, mean_reward = train_baselines(env, lr, gamma, gae_lambda, ent_coef, vf_coef, n_steps, total_timesteps)
        # log / 800, pi / 7
        model, mean_reward = train_baselines(env, lr, gamma, None, ent_coef, vf_coef, n_steps, total_timesteps)

        if mean_reward is not None and mean_reward > best_score:
            best_score = mean_reward
            best_model = model
            best_params = params

            best_model.save('./models/a2c_sb3_best_model')
            with open('./models/sb3_best_params.txt', 'w') as f:
                f.write(f"Best Params: {best_params}\n")
                f.write(f"Best Score: {best_score}\n")

    return best_model, best_score

if __name__ == "__main__":
    param_grid = {
        'lr': [0.005, 0.001],
        'gamma': [0.95, 0.93, 0.9],
        # 'gae_lambda': [0.95], #[0.9, 0.95, 1.0],
        'ent_coef': [0.4, 0.5, 0.6], #[0.1, 0.5, 0.9],
        'vf_coef': [0.3, 0.4, 0.5], # [0.25, 0.5, 0.75],
        'n_steps': [15, 20, 25] # , 20]
    }

    # Example environment (replace with your actual environment creation)
    env = gym_unbalanced_disk.UnbalancedDisk(dt=0.025, umax=3.) # umax is the highest voltage
    env = gym.wrappers.time_limit.TimeLimit(env,max_episode_steps=1000)
    env = Monitor(env, allow_early_resets=True)

    best_model, best_score = grid_search_baselines(env, param_grid)
