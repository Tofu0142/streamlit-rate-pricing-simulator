from typing import Dict, Any 

import pandas as pd 
from mabwiser.mab import MAB, LearningPolicyType,NeighborhoodPolicyType, LearningPolicy, NeighborhoodPolicy

import mlflow 

from environments.spaces_gym import custom_spaces

class SimulatorBanditModel(MAB, mlflow.pyfunc.PythonModel):
    
    def __init__(self, 
                 first_arm: float, 
                 last_arm: float, 
                 n_arms: int, 
                 learning_policy: LearningPolicyType ,
                 ):
        arms = custom_spaces["discrete"](start=first_arm, end=last_arm, n=n_arms)
        self.action_space = arms
        learning_policy = learning_policy
        
        #neighborhood_policy = neighborhood_policy

        super().__init__(arms.to_list(), learning_policy)
        self.data = pd.DataFrame()
        self._data = pd.DataFrame()

    def train(self, reward, *args, **kwargs):
        if reward == 'profit':
            super().partial_fit(self._data["action"], self._data["reward"], *args, **kwargs)
            self._data = pd.DataFrame()
        elif reward == 'converted':
            super().partial_fit(self._data["action"], self._data["converted"], *args, **kwargs)
            self._data = pd.DataFrame()

    def choose_action(self, *args, **kwargs):
        if self._is_initial_fit:
            return super().predict(*args, **kwargs)
        else:
            return self.action_space.sample()[0]
        
    def collect(
        self,
        observation: Dict[str, Any],
        action: float,
        reward: float | int,
        converted: float | int,
    ):
        temp_df = pd.DataFrame(
            {**observation, **{"action": [action], "reward": [reward], "converted": [converted]}}
        )
        
        self.data = pd.concat(
            [self.data, temp_df],
            ignore_index=True,
        )

        self._data = pd.concat(
            [self._data, temp_df],
            ignore_index=True,
        )

    def predict(self, *args, **kwargs):
        return super().predict(None)    
    