import gym
import numpy as np
import matplotlib.pyplot as plt

from env_runner import CartPoleRunner

from agents import DQNAgent, DDQNAgent
import torch as T


if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    
    agent = DDQNAgent(lr=0.0001, input_dims=(env.observation_space.shape), 
                  n_actions=env.action_space.n, dir="models/", env_name='CartPole-v1', algo='DQNAgent', batch_size=32, replace_target_cnt=1000,
                  mem_size=50000, gamma=0.95, epsilon=1.0, eps_min=0.01, eps_dec=.9995)

       
    env_runner = CartPoleRunner(agent, env)

    env_runner.run(n_episodes=2000, save=True)



    #     if i % 10 == 0:
    #         avg_score = np.mean(scores[-10:])
    #         win_pct_list.append(avg_score)
    #         if i % 100 == 0:
    #             print('episode ', i, 'score, %.2f avg score %.2f' % (score, avg_score) ,
    #                   'epsilon %.2f' % agent.epsilon)
    # plt.plot(win_pct_list)
    # plt.show()