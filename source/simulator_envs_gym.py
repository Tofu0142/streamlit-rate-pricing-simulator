from typing import Optional, Union, List, Dict
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import scipy.stats as stats

from spaces_gym import custom_spaces

from room_pricing.bandits.bayesian_model import environment_gym as custom_env
"""
class SimulatorEnv1(custom_env.BaseEnv):
    def __init__(
            self,
            discrete_action_space: bool = True,
            n_actions: int = 100,
            intercept: float = 0.0,
            coefficients: Union[List[float], float] = None,
            mean_base_price: float = 10.0,
            mean_apw: float = 5.0,
            n_reps: int = 1000,
            **kwargs):
        self.intercept = intercept
        self.coefficients = coefficients
        self.mean_apw = mean_apw
        self.mean_base_price = mean_base_price
        self.n_reps = n_reps
        # self.action_space = spaces.Box(low=0, high=1, dtype=np.float32)
        # change action space to be a discrete space with 100 values
        #self.action_space = spaces.Discrete(100, seed=42)

        # =================
        self.discrete_action_space = discrete_action_space

        if discrete_action_space:
            self.action_space = custom_spaces.NormalizedDiscreteSpace(
                boundary=1, n=n_actions
            )
        else:
            self.action_space = spaces.Box(low=0, high=1, dtype=np.float32)
        # =================
        self.observation_space = spaces.Dict({
            "apw": custom_spaces.NegBinomialSpace(mean=mean_apw),
            "base_price": custom_spaces.Simulator1_Price_Space(mean_base_price=mean_base_price),
            
            }
        )
        self.features = ["base_price", "apw"]
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
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        observation = self._get_obs()
        
        #observation = {
            "apw": self.mean_apw,
            "base_price": self.mean_base_price,
        }
        info = self._get_info()
        self._current_state = observation
        self.current_index = 0

        return observation, info
"""


class LogisticModelFeature:
    def __init__(self, 
                 name: str,
                 model_coefficient: float,
                 space: str,
                 **kwargs,):
        self.name = name
        self.model_coefficient = model_coefficient
        self.space = custom_spaces[space](**kwargs)

# if you want to change the space of the base price, you can do it here
BASE_PRICE = LogisticModelFeature(
    name="base_price",
    model_coefficient=-1 / 25,
    space="simulator1_price",
    mean_base_price=100,
)
APW = LogisticModelFeature(
    name="apw",
    model_coefficient=-1 / 10,
    space="negative_binomial",
    mean=14,
)

FEATURES = {"base_price": BASE_PRICE, "apw": APW}


class SimulatorEnv1(custom_env.BaseEnv):
    def __init__(self,
                 discrete_action_space: bool = True,
                 n_actions: int = 100,
                 intercept: float = 10.0,
                 features: Dict[str, LogisticModelFeature] = None,
                 n_reps: int = 10000,
                 **kwargs):
        self.discrete_action_space = discrete_action_space

        if discrete_action_space:
            # self.action_space = custom_spaces['discrete'](boundary=1, n=n_actions)
            self.action_space = custom_spaces["discrete"](
                start = kwargs.get("first_arm", 0),
                end = kwargs.get("last_arm", 1),
                n = n_actions,
            )
        else:
            self.action_space = spaces.Box(low=0, high=1, dtype=np.float32)

        if features is None:
            self.features = FEATURES 
        else:
            self.features = features
        if "base_price" not in self.features:
            raise ValueError("base_price must be included in feature space")

        self.extra_features = [k for k in self.features.keys() if k != "base_price"]

        self.intercept = intercept

        self.observation_space = spaces.Dict(
            {k: v.space for k, v in self.features.items()}
        )

        self.n_reps = n_reps
        self._current_state = None

        super().__init__(**kwargs)

    def _get_info(self):
        #return {"action": self.action}
        return {}

    def _get_obs(self, size=1):
        return {
            k: self.observation_space[k].sample(size)
            for k in self.observation_space.keys()
        }

    def get_obs_dataset(self, size=10):
        return pd.DataFrame.from_dict(self._get_obs(size), orient="columns")

    @staticmethod
    def get_closest_available_action(action, available_actions):
        action = min(available_actions, key=lambda item: abs(item - action))
        return action

    def step(self, action):
        terminated = self.current_index > self.n_reps
        if self.discrete_action_space:
            action = self.get_closest_available_action(
                action, self.action_space.to_list()
            )

        reward = self.get_conversion(action)
        observation = self._get_obs()
        self._current_state = observation
        # Save action taken to be passed into info — useful when this is constrained by the environment
        self.action = action
        info = self._get_info()
        self.current_index += 1

        truncated = False

        return observation, reward, terminated, truncated, info

    def get_conversion(self, action):
        y = self.demand_function2(action, **self._current_state)

        return stats.bernoulli(y).rvs()

    def demand_function(self, action: float, **kwargs):
        """
        Calculates the demand function based on the given action and keyword arguments.

        Parameters:
            action (float): The action to be used in the calculation.
            **kwargs: Additional keyword arguments representing the features.

        Returns:
            float: The calculated demand value.

        Raises:
            ValueError: If the number of arguments does not match the number of features.
        """

        # Check if the number of arguments matches the number of features
        if len(kwargs) != len(self.features):
            raise ValueError(
                f"Invalid number of arguments. Available arguments: {self.features.keys()}"
            )

        y = self.intercept + 1 / 100 * np.random.randn()
        for arg in kwargs:
            if arg == "base_price":
                y += self.features[arg].model_coefficient * kwargs[arg] * (1 + action)
            else:
                y += self.features[arg].model_coefficient * kwargs[arg]

        y = 1 / (1 + np.exp(-y))

        return y
    
    def demand_function2(self, action: float, **kwargs):
        """
        This demand function is based on above, but it's more sensitive to price changes.
        
        """

        # check if the number of arguments matches the number of features
        if len(kwargs) != len(self.features):
            raise ValueError(
                f"Invalid number of arguments. Available arguments: {self.features.keys()}"
            )
        
        y = self.intercept + 1 / 100 * np.random.randn()
        if "hotel_level" in kwargs:
            if kwargs["hotel_level"] < 5:
                for arg in kwargs:
                    if arg == "base_price":
                        y += self.features[arg].model_coefficient * 1.5 * kwargs[arg] * (1 + action)
                    else:
                        y += self.features[arg].model_coefficient * kwargs[arg]

            else:
                for arg in kwargs:
                    if arg == "base_price":
                        y += self.features[arg].model_coefficient * 0.5 * kwargs[arg] * (1 + action)
                    else:
                        y += self.features[arg].model_coefficient * kwargs[arg]
        else:
            for arg in kwargs:
                if arg == "base_price":
                    y += self.features[arg].model_coefficient * kwargs[arg] * (1 + action)
                else:
                    y += self.features[arg].model_coefficient * kwargs[arg]

        y = 1 / (1 + np.exp(-y))

        return y 

