import gym
import numpy as np
import matplotlib.pyplot as plt

from Q_model import QNetwork
import torch as T

from typing import Tuple

class Memory():
    def __init__(self, size:int, state_shape: Tuple[int, ...]) -> None:
        self.size = size
        self.current_states = T.zeros(size, *state_shape)
        self.actions = T.zeros(size)
        self.rewards = T.zeros(size)
        self.next_states = T.zeros(size, *state_shape)
        self.dones = T.zeros(size)
        self.idx_last = 0

    # def _remove_past_memories(self, n_samples:int) -> None:
    #     self.current_states[:-n_samples] = self.current_states[n_samples:]
    #     self.actions[:-n_samples] = self.actions[n_samples:]
    #     self.rewards[:-n_samples] = self.rewards[n_samples:]
    #     self.next_states[:-n_samples] = self.next_states[n_samples:]
    #     self.dones[:-n_samples] = self.dones[n_samples:]

    #     self.current_states[n_samples:] = 0
    #     self.actions[n_samples:] = 0
    #     self.rewards[n_samples:] = 0
    #     self.next_states[n_samples:] = 0
    #     self.dones[n_samples:] = 0

    #     self.idx_last -= n_samples

    def collect_experience(self, experience) -> None:
        index = self.idx_last % self.size
        current_state, action, reward, next_state, done = experience

        self.current_states[index] = current_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done
        self.idx_last = index

    def get_sample(self, sample_size: int, max_prev: int) -> Tuple[T.Tensor, ...]:
        if sample_size > self.idx_last:
            raise ValueError()
        
        past_frontier = min(max_prev, self.idx_last)

        sample_idxs = np.rand.choice(past_frontier, size=sample_size, replace=False)
        return (self.current_states[sample_idxs], self.actions[sample_idxs], self.rewards[sample_idxs],
                self.next_states[sample_idxs], self.dones[sample_idxs]) 
        

class Agent():
    def __init__(self, lr, input_dims, n_actions, dir, name, env_name, algo,
                 mem_size, gamma=0.99, epsilon=1, eps_min=0.01, eps_dec=.9999995) -> None:
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = list(range(n_actions))

        self.dir = dir
        self.env_name = env_name
        self.algo = algo

        self.gamma = gamma
        self.eps_min = eps_min
        self.epsilon = epsilon
        self.eps_dec = eps_dec


        self.memory = Memory(mem_size, input_dims)
        self.q_net = QNetwork(lr=0.001, n_actions=self.n_actions, 
                              input_dim=self.input_dims, 
                              dir=self.dir, 
                              name=f"{self.env_name}_{self.algo}_q_eval")
        
        self.target_net = QNetwork(lr=0.001, n_actions=self.n_actions, 
                              input_dim=self.input_dims, 
                              dir=self.dir, 
                              name=f"{self.env_name}_{self.algo}_q_target")
        self.update_target_network()


    def update_target_network(self):
        self.q_net.save_checkpoint()
        self.target_net.load_checkpoint()


    def choose_action(self, state):
        if np.random.random() < self.eps:
            return  np.random.choice(self.action_space)
        else: 
            state = T.tensor(state, dtype=T.float).to(self.q_net.device)
            actions = self.q_net.forward(state)
            return T.argmax(actions).item()


    def decrease_eps(self):
        self.eps = self.eps*self.eps_dec**2 if self.eps > self.eps_min else self.eps_min


    def learn(self, state, action, reward, next_state, done):
        self.q_net.optimizer.zero_grad()
        
        states = T.tensor(state, dtype=T.float).to(self.q_net.device)
        actions = T.tensor(action)
        rewards = T.tensor(reward)
        next_states = T.tensor(next_state, dtype=T.float).to(self.q_net.device)
        dones = T.tensor(done)

        q_pred = self.q_net.forward(states)[actions]
        q_next = self.target_net.forward(next_states).max()

        # Filter non-terminal states
        q_next = T.logical_and(q_next, ~dones)

        q_target = rewards + self.gamma * q_next

        loss = self.q_net.loss(q_target, q_pred).to(self.q_net.device)
        loss.backward()
        self.q_net.optimizer.step()

        self.decrease_eps()




if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    print(env.observation_space.shape)

    agent = Agent(lr=0.0001, input_dims=env.observation_space.shape, n_actions=env.action_space.n)
    memory = Memory(..., ...) # FIXME


    scores = []
    win_pct_list = []
    n_games = 10000
    sample_size = 32
    max_prev = ... # FIXME
    update_steps = ... # FIXME

    for i in range(n_games):
        done = False
        observation = env.reset()
        observation_q = observation
        score = 0
        i = 0
        while not done:
            action = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            memory.collect_experience(observation, action, reward, observation_, done)
            if memory.idx_last > 32:
                minibatch = memory.get_sample(sample_size=sample_size, max_prev=max_prev)
                agent.learn(*minibatch)

            if i % update_steps == 0:
                agent.update_target_network()
            
            # agent.learn(observation, action, reward, observation_)

            score += reward
            i += 1
            observation = observation_
        scores.append(score)
        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            win_pct_list.append(avg_score)
            if i % 1000 == 0:
                print('episode ', i, 'score, %.2f avg score %.2f' % (score, avg_score) ,
                      'epsilon %.2f' % agent.eps)
    plt.plot(win_pct_list)
    plt.show()
