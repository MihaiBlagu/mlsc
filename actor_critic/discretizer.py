import gymnasium as gym
import numpy as np


# class Discretizer(gym.Wrapper):
#     def __init__(self, env, bins=10):
#         super(Discretizer, self).__init__(env) #sets self.env
#         self.bins = bins

#         self.action_space = gym.spaces.Discrete(self.bins)
#         self.alow, self.ahigh = env.action_space.low, env.action_space.high


#     def step(self, action):
#         action = self.opt_action(action)
#         observation, _, done, info = self.env.step(action)
#         # normalize
#         observation[0] = (observation[0] + np.pi) % (2 * np.pi) - np.pi
#         reward = self.get_reward(observation, action=action)

#         return np.array(observation), reward, done, info    # Use minus reward here


#     def reset(self):
#         return self.env.reset()


#     def random_action(self):
#         action = self.env.action_space.sample()
#         return ((action - self.alow)/(self.ahigh - self.alow)*self.bins).astype(int)


#     def opt_action(self, action):
#         action = action / self.bins *(self.ahigh - self.alow) + self.alow
#         return action


#     def get_reward(self, observation, action):
#         theta = (observation[0] + np.pi) % (2 * np.pi) - np.pi
#         omega = observation[1]

#         alpha, beta, gamma = 100, 0.05, 0.5
#         reward = alpha*theta**2 - beta*omega**2 - gamma*action**2
#         return reward

class Discretizer(gym.ActionWrapper):
    def __init__(self, env, bins=11):
        super(Discretizer, self).__init__(env)
        self.n_discrete_actions = bins
        self.action_space = gym.spaces.Discrete(bins)
        self.action_values = np.linspace(env.action_space.low, env.action_space.high, bins)

    def action(self, action):
        # Convert discrete action to continuous action
        return np.array([self.action_values[action]])

    def reset(self, **kwargs):
        # Remove 'options' argument if present for compatibility
        if 'options' in kwargs:
            kwargs.pop('options')
        return self.env.reset(**kwargs)

    def step(self, action):
        # Convert discrete action to continuous action
        continuous_action = self.action(action)
        return self.env.step(continuous_action)
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()