import numpy as np
from deep_q_net import QNetwork
import torch as T

from memory import Memory      

class DQNAgent():
    def __init__(self, lr, input_dims, n_actions, dir, env_name, algo, batch_size, replace_target_cnt,
                 mem_size, gamma=0.99, epsilon=1, eps_min=0.01, eps_dec=1e-5) -> None:
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
        self.batch_size = batch_size

        self.q_net = QNetwork(lr=lr, n_actions=self.n_actions, 
                              input_dims=self.input_dims, 
                              dir=self.dir, 
                              name=f"{self.env_name}_{self.algo}_q_eval")
        
        self.target_net = QNetwork(lr=lr, n_actions=self.n_actions, 
                              input_dims=self.input_dims, 
                              dir=self.dir, 
                              name=f"{self.env_name}_{self.algo}_q_target")
        self.replace_target_cnt = replace_target_cnt
        self.learning_step_cnt = 0
        self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return  np.random.choice(self.action_space)
        else: 
            state = T.tensor(np.array([state]), dtype=T.float).to(self.q_net.device)
            actions = self.q_net.forward(state)
            return T.argmax(actions).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.collect_experience(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, next_state, done = self.memory.get_sample(sample_size=self.batch_size)
        state = state.to(self.q_net.device)
        action = action.to(self.q_net.device)
        reward = reward.to(self.q_net.device)
        next_state= next_state.to(self.q_net.device)
        done = done.to(self.q_net.device)
        return state, action, reward, next_state, done

    def decrease_eps(self):
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_net.save_checkpoint()
        self.target_net.save_checkpoint()

    def load_models(self):
        self.q_net.load_checkpoint()
        self.target_net.load_checkpoint()

    def _get_qnext(self, next_states):
        return self.target_net.forward(next_states).max(dim=1)[0]

    def learn(self):
        if self.memory.idx_last < self.batch_size:
            return

        self.q_net.optimizer.zero_grad()

        if self.learning_step_cnt % self.replace_target_cnt == 0:
            self.update_target_network()
        
        states, actions, rewards, next_states, dones = self.sample_memory()

        q_pred = self.q_net.forward(states)[np.arange(self.batch_size), actions]
        q_next = self._get_qnext(next_states)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        loss = self.q_net.loss(q_target, q_pred).to(self.q_net.device)
        loss.backward()
        self.q_net.optimizer.step()

        self.decrease_eps()
        self.learning_step_cnt += 1


class DDQNAgent(DQNAgent):
    def __init__(self, lr, input_dims, n_actions, dir, env_name, algo, batch_size, replace_target_cnt,
                 mem_size, gamma=0.99, epsilon=1, eps_min=0.01, eps_dec=1e-5) -> None:
        super(DDQNAgent, self).__init__(lr, input_dims, n_actions, dir, env_name, algo, batch_size, replace_target_cnt,
                 mem_size, gamma=0.99, epsilon=1, eps_min=0.01, eps_dec=1e-5)
        
    def _get_qnext(self, next_states):
        actions_idx = self.q_net.forward(next_states).argmax(dim=1)
        q_next = self.target_net.forward(next_states)[np.arange(self.batch_size), actions_idx]
        return q_next


# class DDQNAgent():
#     def __init__(self, lr, input_dims, n_actions, dir, env_name, algo, batch_size, replace_target_cnt,
#                  mem_size, gamma=0.99, epsilon=1, eps_min=0.01, eps_dec=1e-5) -> None:
        
#         self.lr = lr
#         self.input_dims = input_dims
#         self.n_actions = n_actions
#         self.action_space = list(range(n_actions))

#         self.dir = dir
#         self.env_name = env_name
#         self.algo = algo

#         self.gamma = gamma
#         self.eps_min = eps_min
#         self.epsilon = epsilon
#         self.eps_dec = eps_dec

#         self.memory = Memory(mem_size, input_dims)
#         self.batch_size = batch_size

#         self.q_net = QNetwork(lr=lr, n_actions=self.n_actions, 
#                               input_dims=self.input_dims, 
#                               dir=self.dir, 
#                               name=f"{self.env_name}_{self.algo}_q_eval")
        
#         self.target_net = QNetwork(lr=lr, n_actions=self.n_actions, 
#                               input_dims=self.input_dims, 
#                               dir=self.dir, 
#                               name=f"{self.env_name}_{self.algo}_q_target")
#         self.replace_target_cnt = replace_target_cnt
#         self.learning_step_cnt = 0
#         self.update_target_network()

#     def update_target_network(self):
#         self.target_net.load_state_dict(self.q_net.state_dict())

#     def choose_action(self, state):
#         if np.random.random() < self.epsilon:
#             return  np.random.choice(self.action_space)
#         else: 
#             state = T.tensor(np.array([state]), dtype=T.float).to(self.q_net.device)
#             actions = self.q_net.forward(state)
#             return T.argmax(actions).item()

#     def store_transition(self, state, action, reward, next_state, done):
#         self.memory.collect_experience(state, action, reward, next_state, done)

#     def sample_memory(self):
#         state, action, reward, next_state, done = self.memory.get_sample(sample_size=self.batch_size)
#         state = state.to(self.q_net.device)
#         action = action.to(self.q_net.device)
#         reward = reward.to(self.q_net.device)
#         next_state= next_state.to(self.q_net.device)
#         done = done.to(self.q_net.device)
#         return state, action, reward, next_state, done

#     def decrease_eps(self):
#         self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min

#     def save_models(self):
#         self.q_net.save_checkpoint()
#         self.target_net.save_checkpoint()

#     def load_models(self):
#         self.q_net.load_checkpoint()
#         self.target_net.load_checkpoint()

#     def learn(self):
#         if self.memory.idx_last < self.batch_size:
#             return

#         self.q_net.optimizer.zero_grad()

#         if self.learning_step_cnt % self.replace_target_cnt == 0:
#             self.update_target_network()
        
#         states, actions, rewards, next_states, dones = self.sample_memory()

#         q_pred = self.q_net.forward(states)[np.arange(self.batch_size), actions] # FIXME
       
#         actions_idx = self.q_net.forward(next_states).argmax(dim=1)
#         q_next = self.target_net.forward(next_states)[np.arange(self.batch_size), actions_idx]


#         # Filter non-terminal states
#         q_next[dones] = 0.0

#         q_target = rewards + self.gamma * q_next

#         loss = self.q_net.loss(q_target, q_pred).to(self.q_net.device)
#         loss.backward()
#         self.q_net.optimizer.step()

#         self.decrease_eps()
#         self.learning_step_cnt += 1