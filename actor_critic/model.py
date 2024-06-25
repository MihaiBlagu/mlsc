import torch.nn as nn
import torch
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, env, num_actions=6, hidden_size=40):
        super(ActorCritic, self).__init__()
        num_inputs = env.observation_space.shape[0]
        self.num_actions = num_actions

        #define your layers here:
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)  #a)
        self.critic_linear2 = nn.Linear(hidden_size, 1) #a)
        self.actor_linear1 = nn.Linear(num_inputs, hidden_size) #a)
        self.actor_linear2 = nn.Linear(hidden_size, self.num_actions) #a)
    
    def actor(self, state, return_logp=False):
        #state has shape (Nbatch, Nobs)
        hidden = torch.tanh(self.actor_linear1(state)) #a)
        h = self.actor_linear2(hidden) #a=)
        h = h - torch.max(h,dim=1,keepdim=True)[0] #for additional numerical stability
        logp = h - torch.log(torch.sum(torch.exp(h),dim=1,keepdim=True)) #log of the softmax
        if return_logp:
            return logp
        else:
            return torch.exp(logp) #by default it will return the probability

        # # State has shape (Nbatch, Nobs)
        # hidden = torch.tanh(self.actor_linear1(state))
        # output = torch.tanh(self.actor_linear2(hidden))  # Output in range [-1, 1]
        # continuous_action = output * 3  # Scale output to range [-3, 3]
        # return continuous_action[0]
    
    def critic(self, state):
        #state has shape (Nbatch, Nobs)
        hidden = torch.tanh(self.critic_linear1(state)) #a)
        return self.critic_linear2(hidden)[:,0] #a) #no activation function
    
    def forward(self, state):
        #state has shape (Nbatch, Nobs)
        return self.critic(state), self.actor(state)