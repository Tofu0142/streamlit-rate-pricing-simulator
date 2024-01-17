from typing import Optional, Union, List
# import gymnasium as gym
#from gymnasium import spaces
import gym 
from gym import spaces
import pandas as pd
import numpy as np
import scipy.stats as stats
from room_pricing.bandits.bayesian_model import spaces as custom_spaces


class BaseEnv(gym.Env):
    def __init__(
        self, **kwargs
    ):
        self.current_index = 0
        self.action = None
        self._current_state = None

    def _get_obs(self):
        raise RuntimeError("Not Implemented")

    def _get_info(self):
        raise RuntimeError("Not Implemented")

    def step(self, action):
        self.current_index += 1
        return None, None, False, False, {}

    def reset(self, seed=0, options=None):
        observation = self._get_obs()
        info = self._get_info()
        self._current_state = observation
        self.current_index = 0

        return observation, info


class LogisticDemandEnvironment(BaseEnv):
    def __init__(
        self,
        intercept: float = 10,
        coefficients: Union[List[float], Optional[float]] = np.array(
            [-1 / 25, -1 / 10]
        ),
        mean_base_price: float = 8.0,
        mean_apw: float = 4,
        n_reps: int = 10000,
        **kwargs
    ):
        self.observation_space = spaces.Dict(
            {
                # This will probably need some more customisation,
                # e.g. features can be passed as class arguments, along with their moments
                "base_price": custom_spaces.SkewNormSpace(mean=mean_base_price),
                "apw": custom_spaces.NegBinomialSpace(mean=mean_apw),
            }
        )

        self.action_space = spaces.Box(low=0, high=1, dtype=np.float32)

        self.intercept = intercept
        self.features = ["base_price", "apw"]
        self.coefficients = coefficients
        self.n_reps = n_reps

        super().__init__(**kwargs)

    def _get_info(self):
        return {}

    def _get_obs(self, size=1):
        return {
            k: self.observation_space[k].sample(size)
            for k in self.observation_space.keys()
        }

    def get_obs_dataset(self, size=10):
        return pd.DataFrame.from_dict(self._get_obs(size), orient="columns")

    def step(self, action):
        terminated = self.current_index > self.n_reps
        reward = self.get_conversion(action)
        observation = self._get_obs()
        self._current_state = observation
        info = self._get_info()
        self.current_index += 1

        return observation, reward, terminated, False, info

    def get_conversion(self, action):
        y = self.demand_function(
            price=self._current_state["base_price"] * (1 + action),
            apw=self._current_state["apw"],
        )
        return stats.bernoulli(y).rvs()

    def demand_function(
        self,
        price,
        apw,
    ):
        y = (
            self.intercept
            + self.coefficients[0] * price
            + self.coefficients[1] * apw
            + 1 / 100 * np.random.randn()
        )
        y = 1 / (1 + np.exp(-y))

        return y
