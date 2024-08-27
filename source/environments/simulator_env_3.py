from typing import Optional, Union, List, Dict
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import scipy.stats as stats
import math
from environments.spaces_gym import custom_spaces

from room_pricing.bandits.bayesian_model import environment_gym as custom_env


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

"""
This simulator is for scenario 5
"""

class SimulatorEnv3(custom_env.BaseEnv):
    def __init__(self,
               discrete_action_space: bool = True,
               n_actions: int = 100,
               intercept: float = 10.0,
               features: Dict[str, LogisticModelFeature] = None,
               n_reps: int = 2000,
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
        self.data = self._get_obs(self.n_reps)
        super().__init__(**kwargs)

    def _get_info(self):
        return {}
    
    def _get_obs(self, size=1):
        data = {
            k: self.observation_space[k].sample(size)
            for k in self.observation_space.keys()
        }
        return pd.DataFrame.from_dict(data, orient="columns")

    def get_obs_dataset(self):
        return self.data

    
    def _fill_obs_dataset(self, size=10):
        """
        when apw becomes 0, we need to delete these rows from the dataset
        and fill the dataset with new rows
        """
        new_data = self._get_obs(size)
        self.data = pd.concat([self.data, new_data], ignore_index=True)
       

    def _delete_zero_apw(self):
        """
        delete data with apw < 0
        """
        self.data = self.data[self.data['apw'] >= 0]

    @staticmethod
    def get_closest_available_action(action, available_actions):
        action = min(available_actions, key=lambda item: abs(item - action))
        return action
    
    def step(self, action, current_day):
        terminated = ((self.current_index+1) >= self.n_reps)
        if self.discrete_action_space:
            action = self.get_closest_available_action(
                action, self.action_space.to_list()
            )
        reward = self.get_conversion(action, current_day)
        try:
            observation = self.data.iloc[self.current_index].to_dict()
        except IndexError:
            # Handle cases where self.current_index is out of bounds
            print(f"IndexError: Current index {self.current_index} is out of bounds.")
            raise
        self._current_state = observation
        # Save action taken to be passed into info — useful when this is constrained by the environment
        self.action = action
        info = self._get_info()
        self.current_index += 1

        truncated = False

        return observation, reward, terminated, truncated, info
    
    def reset(self):
        self.current_index = 0
        self.data['apw'] = self.data['apw'] - 1
        self._delete_zero_apw()
        self._fill_obs_dataset(self.n_reps - len(self.data))
        self._current_state = self.data.iloc[self.current_index].to_dict()

    def get_conversion(self, action, current_day):
        y = self.demand_function(action, current_day, **self._current_state)
        y = max(0, min(y, 1))  # Ensure y is within [0, 1]
        return stats.bernoulli(y).rvs()
    
    def demand_function(self, action: float, current_day: int, **kwargs):
        # Check if the required arguments are provided
        if len(kwargs) != len(self.features):
            raise ValueError(f"Invalid number of arguments. Available arguments: {self.features.keys()}")
        
        # Determine which demand function to use based on the current day
        if current_day <= 3:
            return self.demand_func1(action, **kwargs)
        elif (current_day > 3) and (current_day < 7):
            return self.demand_func2(action, **kwargs)
        elif (current_day >= 7) and (current_day < 10):
            return self.demand_func1(action, **kwargs)
        else:
            return self.demand_func2(action, **kwargs)

    def demand_func1(self, action: float, **kwargs):
        if len(kwargs) != len(self.features):
            raise ValueError(
                f"Invalid number of arguments. Available arguments: {self.features.keys()}"
            )

        y = self.intercept + 1 / 100 * np.random.randn()   
        if kwargs["apw"] // 3 ==0:
            apw = kwargs["apw"]

        else:
            apw = math.ceil(kwargs["apw"] / 3) * 3

        multiplier, _promotion = self.calculate_elasticity_increase(apw, kwargs["promotion"])
        multiplier = multiplier if multiplier > 0.5 else 0.5
        kwargs['promotion'] = _promotion
        for arg in kwargs:
            if arg == "base_price":
                y += self.features[arg].model_coefficient * 1 * kwargs[arg] * (1 + action)
            else:
                y += self.features[arg].model_coefficient * kwargs[arg]
        y = 1 / (1 + np.exp(-y))

        return y
        
    def demand_func2(self, action: float, **kwargs):
        if len(kwargs) != len(self.features):
            raise ValueError(
                f"Invalid number of arguments. Available arguments: {self.features.keys()}"
            )

        y = self.intercept + 1 / 100 * np.random.randn()   
        if kwargs["apw"] // 3 ==0:
            apw = kwargs["apw"]

        else:
            apw = math.ceil(kwargs["apw"] / 3) * 3

        multiplier, _promotion = self.calculate_elasticity_increase(apw, kwargs["promotion"])
        multiplier = multiplier if multiplier > 0.5 else 0.5
        kwargs['promotion'] = _promotion
        for arg in kwargs:
            if arg == "base_price":
                y += self.features[arg].model_coefficient  * 1.05 * kwargs[arg] * (1 + action)
            else:
                y += self.features[arg].model_coefficient * kwargs[arg]
        y = 1 / (1 + np.exp(-y))

        return y

    def calculate_elasticity_increase(self, apw, initial_promotion):
        """
        There is elasticity factor in demand_function, which is calculated by apw and promotion.

        the promotion decreases as the apw decreases, but it follows binomial distribution.
        if apw is over 30, the promotion will not change
        """

        p = apw/30 if apw < 30 else 1
        promotion = np.random.binomial(n=int(initial_promotion ), p=p) / 100

        # Y increase as apw and promotion decrease
        Y = 1 / (apw + 1) + 1 / (promotion + 1 )  # Add 1 to avoid division by zero
        return Y, promotion
        
