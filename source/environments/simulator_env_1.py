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
This simulator is for scenario 1-4

"""
class SimulatorEnv1(custom_env.BaseEnv):
    def __init__(self,
                 discrete_action_space: bool = True,
                 n_actions: int = 100,
                 intercept: float = 10.0,
                 features: Dict[str, LogisticModelFeature] = None,
                 n_reps: int = 10000,
                 data = None,
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
        self.scenario = kwargs.get("scenario", 1)
        if not data:
            self.data = self.get_obs_dataset(self.n_reps)
        else:
            self.data = data
        self.current_index = 0
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
        terminated =( (self.current_index+1) >= self.n_reps)
        if self.discrete_action_space:
            action = self.get_closest_available_action(
                action, self.action_space.to_list()
            )

        reward = self.get_conversion(action)
        # observation = self._get_obs()
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

    def get_conversion(self, action):
        if self.scenario == 1:
            y = self.demand_function1(action, **self._current_state)
        elif self.scenario == 2:
            y = self.demand_function2(action, **self._current_state)
        elif self.scenario == 3:
            y = self.demand_function3(action, **self._current_state)
        elif self.scenario == 4:
            y = self.demand_function4(action, **self._current_state)
        else:
            raise ValueError("Invalid scenario number")

        return stats.bernoulli(y).rvs()


#==================different demand functions==================
    def demand_function1(self, action: float, **kwargs):
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
                y += self.features[arg].model_coefficient * 0.8 * kwargs[arg] * (1 + action)
            else:
                y += self.features[arg].model_coefficient * kwargs[arg]

        y = 1 / (1 + np.exp(-y))

        return y
    
    def demand_function2(self, action: float, **kwargs):
        """
        All customers are price elastic
        """
        # Check if the number of arguments matches the number of features
        if len(kwargs) != len(self.features):
            raise ValueError(
                f"Invalid number of arguments. Available arguments: {self.features.keys()}"
            )

        y = self.intercept + 1 / 100 * np.random.randn()
        for arg in kwargs:
            if arg == "base_price":
                y += self.features[arg].model_coefficient * 1.5 * kwargs[arg] * (1 + action)
            else:
                y += self.features[arg].model_coefficient * kwargs[arg]

        y = 1 / (1 + np.exp(-y))

        return y


    def demand_function3(self, action: float, **kwargs):
        """
         customers who are looking at 5-star hotels are price inelastic, customers who are < 5star are price elastic.
        
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

    def demand_function4(self, action: float, **kwargs):

        if len(kwargs) != len(self.features):
            raise ValueError(
                f"Invalid number of arguments. Available arguments: {self.features.keys()}"
            )
        
        y = self.intercept + 1 / 100 * np.random.randn()

        for arg in kwargs:
            if arg == "base_price":
                # Adjusting the multiplier for 'base_price' based on 'customer_type'
                multiplier = 1.5 if kwargs["customer_type"] == 0 else 1.1  # Increased multiplier for customer_type 1
                y += self.features[arg].model_coefficient * multiplier * kwargs[arg] * (1 + action)
            else:
                y += self.features[arg].model_coefficient * kwargs[arg]

        y = 1 / (1 + np.exp(-y))

        return y 

