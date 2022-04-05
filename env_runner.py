import numpy as np

class EnvironmentRunner():
    def __init__(self, agent, env) -> None:
        self.env = env
        self.agent = agent
        
    def _reward(self, reward:float, done:bool):
        return reward

    def _run_episode(self):
        done = False
        observation = self.env.reset()
        score = 0
        while not done:
            action = self.agent.choose_action(observation)
            observation_, reward, done, info = self.env.step(action)

            reward = self._reward(reward, done)

            self.agent.store_transition(observation, action, reward, 
                                        observation_, int(done))
            self.agent.learn()
            
            score += reward
            observation = observation_
        return score

    def run(self, n_episodes:int, save:bool, 
            verbose: bool = True, verbosity: int = 100):
        scores = []
        for i in range(n_episodes):
            score = self._run_episode()
            scores.append(score)

            if verbose:
                if i % verbosity == 0:
                    print('episode ', i, 'score, %.2f avg score %.2f' % (score, np.mean(scores[:-100])) ,
                      'epsilon %.2f' % self.agent.epsilon)

        if save:
            self.agent.save_models()

        return scores


class CartPoleRunner(EnvironmentRunner):
    def __init__(self, agent, env) -> None:
        super(CartPoleRunner, self).__init__(agent, env)

    def _reward(self, reward:float, done:bool):
        reward = reward if not done else -reward
        return reward