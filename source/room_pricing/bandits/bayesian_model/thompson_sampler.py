import bambi as bmb
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import lambertw
from room_pricing.bandits.bayesian_model.environment import BaseEnv


class ThompsonSampler:
    def __init__(
        self,
        env: BaseEnv,
        context_features=["base_price", "apw"],
        training_features=["price", "apw"],
    ):
        self.env = env
        self.context_features = context_features
        self.training_features = training_features
        self.action_feature = "fee"
        self.target = "converted"

    @property
    def formula(self):
        formula = f"{self.target} ~ "
        formula += " + ".join(self.training_features)

        return formula

    def fit_model(
        self,
        data,
        training_size,
        draws,
        chains,
        tune,
    ):
        if training_size is not None:
            data = data[
                self.env.current_index - training_size : self.env.current_index, :
            ].copy()
        else:
            data = data.copy()

        self.model = bmb.Model(self.formula, data, family="bernoulli")
        self.results = self.model.fit(
            draws=draws, chains=chains, tune=tune, idata_kwargs={"log_likelihood": True}
        )

    def get_sample_curve(self, mode="random"):
        posterior = self.results.posterior

        self.params = {}

        if mode == "random":
            f = np.random.choice
        elif mode == "predict":
            f = np.median
        else:
            raise (ValueError, "Only random and predict allowed")

        self.params["intercept"] = f(list(posterior["Intercept"].values)[0])

        for col in self.training_features:
            self.params[col] = f(list(posterior[col].values)[0])

        return

    def get_revenue_maximiser_action(self, data, params=None):
        if params is None:
            params = self.params

        def max_action(base_price, apw):
            B = params["price"] * base_price
            A = params["intercept"] + B + params["apw"] * apw

            z = A - B - 1
            z = np.real(lambertw(np.exp(z)))

            return -(z + B + 1) / B

        data["fee"] = data[["base_price", "apw"]].apply(
            lambda row: max_action(row["base_price"], row["apw"]), axis=1
        )

        return data

    def choose_action(self, data, mode="random"):
        self.get_sample_curve(mode)
        self.get_revenue_maximiser_action(data)
        return data

    def get_conversions(self, data):
        if "fee" not in data.columns:
            # Used in the initialisation of training set
            data["fee"] = np.random.rand(len(data))

        if "converted" not in data.columns:
            data["price"] = data["base_price"] * (1 + data["fee"])
            conv_prob = self.env.demand_function(data["price"], data["apw"])
            data["converted"] = conv_prob.apply(stats.bernoulli.rvs)
        else:
            data["converted"] = np.where(
                data["converted"],
                data["price"] >= data["base_price"] * (1 + data["fee"]),
                data["price"] * (1 - 0.2) >= data["base_price"] * (1 + data["fee"]),
            ).astype(int)

        return data

    def run(
        self,
        training_size=None,
        draws=2000,
        chains=1,
        tune=1500,
        full=False,
    ):
        self.env.reset()
        training_data = self.env.get_batch_of_data(full=full)
        training_data = self.get_conversions(training_data)

        while self.env.current_index < len(self.env.training_set):
            print("Training Bayesian model")
            self.fit_model(
                training_data,
                training_size=training_size,
                draws=draws,
                chains=chains,
                tune=tune,
            )
            if not full:
                data = self.env.get_batch_of_data()
                data = self.choose_action(data)
                data = self.get_conversions(data)
                training_data = pd.concat((training_data, data))
            else:
                self.env.current_index += 1

    def predict(self, data):
        data = data.copy()
        data = self.choose_action(data, mode="predict")
        data[self.action_feature] = data[self.action_feature].clip(lower=0, upper=1)
        return data
