import gym

class EnvironmentRunner():
    def __init__(self, agent) -> None:
        self.env = None
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

            self.agent.store_transition(observation, action, reward, observation_, int(done))
            self.agent.learn()
            score += reward

        return score

    def run(self, n_episodes:int, save:bool):
        scores = []
        for _ in range(n_episodes):
            score = self._run_episode()
            scores.append(score)
        
        if save:
            self.agent.save_models()

        return scores


class CartPoleRunner(EnvironmentRunner):
    def __init__(self, agent) -> None:
        super(CartPoleRunner, self).__init__(agent)
        self.env = gym.make('CartPole-v1')

    def _reward(self, reward:float, done:bool):
        return reward if not done else -reward