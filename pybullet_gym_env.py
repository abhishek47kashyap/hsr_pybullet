"""
Ref:
- https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
- https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952
- https://medium.com/cloudcraftz/build-a-custom-environment-using-openai-gym-for-reinforcement-learning-56d7a5aa827b
- https://github.com/openai/gym/blob/master/gym/envs/box2d/bipedal_walker.py
"""

import gym
from gym import spaces

from stable_baselines3.common.env_checker import check_env

class HsrPybulletEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        self.action_space = spaces.Box()
        self.observation_space = None
    
    def step(self, action):
        ...
    
    def reset(self):
        ...
    
    def render(self, mode="human", close=False):
        ...

if __name__ == "__main__":
    env = HsrPybulletEnv()
    check_env(env)