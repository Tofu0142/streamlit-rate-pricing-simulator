import bambi as bmb
import pandas as pd
import numpy as np
import gymnasium as gym
from scipy.special import lambertw


class ThompsonAgent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.data = pd.DataFrame()
        self.training_features = ["price", "apw"]
        self.target = "converted"
        self.results = None

    @property
    def formula(self):
        formula = f"{self.target} ~ "
        formula += " + ".join(self.training_features)

        return formula

    def get_posterior(self):
        return self.results.posterior

    def get_sample_curve(self, mode="random"):
        posterior = self.get_posterior()

        self.params = {}

        if mode == "random":
            f = np.random.choice
        elif mode == "predict":
            f = np.median
        else:
            raise ValueError("Only random and predict allowed")

        self.params["intercept"] = f(list(posterior["Intercept"].values[0]))

        for col in self.training_features:
            self.params[col] = f(list(posterior[col].values[0]))

        return self.params

    def get_closest_available_action(self, action, available_actions):
        action = min(available_actions, key=lambda item: abs(item - action))
        return action

    def max_action(self, base_price, apw, params):
        B = params["price"] * base_price
        A = params["intercept"] + B + params["apw"] * apw

        z = A - B - 1
        z = np.real(lambertw(np.exp(z)))
        action = -(z + B + 1) / B

        if self.env.discrete_action_space:
            action = self.get_closest_available_action(
                action, available_actions=self.env.action_space.to_list()
            )
        return action

    def _choose_action(self, observation, params=None):
        if params is None:
            params = self.params

        action = self.max_action(
            observation["base_price"], observation["apw"], params=params
        )
        return action

    def choose_action(self, observation, mode="random"):
        _ = self.get_sample_curve(mode)
        action = self._choose_action(observation)
        return action

    def collect(self, observation, action, reward):
        _data = {
            "price": [observation["base_price"] * (1 + action)],
            "apw": [observation["apw"]],
            "converted": [reward],
        }

        self.data = pd.concat((self.data, pd.DataFrame(_data)), ignore_index=True)

    def train(
        self,
        draws: int = 2000,
        chains: int = 1,
        tune: int = 1500,
    ):
        self.model = bmb.Model(self.formula, self.data, family="bernoulli")
        self.results = self.model.fit(
            draws=draws, chains=chains, tune=tune, idata_kwargs={"log_likelihood": True}
        )

    def predict_batch(self, data):
        data = data.copy()
        current_params = self.get_sample_curve(mode="predict")
        data["action"] = data.apply(
            lambda row: self._choose_action(
                {"base_price": row["base_price"], "apw": row["apw"]}, current_params
            ),
            axis=1,
        )
        data["action"] = data["action"].clip(lower=0, upper=1)

        return data
