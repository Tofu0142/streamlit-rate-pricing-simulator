import pandas as pd
import scipy
from typing import List, Optional
from .utils import to_vw_example_format


class BaseEnvironment:
    def __init__(
        self,
        data: pd.DataFrame,
        context_columns: List[str],
        extra_columns: Optional[List[str]] = None,
        format_function: callable = to_vw_example_format,
    ):
        self.df = data.copy()

        self.n_examples = len(self.df)

        self.context_columns = context_columns
        self.extra_columns = extra_columns
        self.format_function = format_function
        self.DF = self.df.copy().to_dict(orient="records")
        self.X = self.df[context_columns].copy().to_dict(orient="records")
        if extra_columns is not None:
            self.E = self.df[extra_columns].copy().to_dict(orient="records")

        self.current_index = 0

    def reset(self):
        self.current_index = 0

    @property
    def current_row(self):
        return self.DF[self.current_index]

    def get_context(self):
        return self.X[self.current_index]

    def step(self):
        self.current_index += 1

    def give_reward(self, *args, **kwargs):
        raise RuntimeError("Not Implemented")


class LinearRegressionEnvironment(BaseEnvironment):
    def __init__(self, label_column, **kwargs):
        super().__init__(**kwargs)
        self.label_column = label_column
        self.Y = self.df[label_column].copy()

    def give_reward(self, action, *args, **kwargs):
        return -((action - self.Y[self.current_index]) ** 2)


class LogisticDemandEnvironment(BaseEnvironment):
    def __init__(self, label_column, demand_function, **kwargs):
        super().__init__(**kwargs)
        self.label_column = label_column
        self.Y = self.df[label_column].copy()
        self.demand_function = demand_function
        self.MAX_REWARD = 2

    def give_reward(self, action, *args, **kwargs):
        price = self.current_row["x"] * (1 + action)
        prob = self.demand_function(price, *args, **kwargs)
        prob = scipy.stats.bernoulli.rvs(prob)

        return prob * price
