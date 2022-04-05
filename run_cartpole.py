import gym
import numpy as np
import matplotlib.pyplot as plt

from env_runner import CartPoleRunner

from agents import DQNAgent, DDQNAgent
import torch as T


if __name__ == '__main__':

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    
    agent = DDQNAgent(lr=0.0001, input_dims=(env.observation_space.shape), 
                      n_actions=env.action_space.n, dir="models/", 
                      env_name=env_name, algo='DQNAgent', batch_size=32, 
                      replace_target_cnt=1000, mem_size=50000, gamma=0.95, 
                      epsilon=1.0, eps_min=0.01, eps_dec=.9995)
 
    env_runner = CartPoleRunner(agent, env)

    env_runner.run(n_episodes=2000, save=True)