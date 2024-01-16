from typing import Optional, Union, List
import pandas as pd
import numpy as np
import scipy.stats as stats


class BaseEnv:
    def __init__(
        self,
        data: pd.DataFrame = None,
        batch_size: int = 50,
        test_size: Optional[float] = None,
    ):
        self.batch_size = batch_size
        self.current_index = 0
        if data is None:
            self.data = self.generate_data()
        else:
            self.data = data.copy()

        if test_size is not None:
            self.training_set = self.data.sample(frac=1 - test_size).copy()
            self.test_set = self.data.iloc[
                ~self.data.index.isin(self.training_set.index)
            ].copy()
        else:
            self.training_set = self.data.copy()
            self.test_set = self.data.copy()

    def demand_function(self):
        raise RuntimeError("Not Implemented")

    def generate_data(self):
        raise RuntimeError("Not Implemented")

    def reset(self):
        self.current_index = 0

    def get_batch_of_data(self, full=False):
        if full:
            self.current_index = len(self.training_set) - 1
            return self.training_set.copy()
        else:
            start = self.current_index
            end = start + self.batch_size
            df = self.training_set.iloc[start:end, :].copy()
            self.current_index += self.batch_size
            return df


class LogisticDemandEnvironment(BaseEnv):
    def __init__(
        self,
        intercept: float = 0,
        coefficients: Union[List[float], float] = None,
        mean_base_price: float = 8.0,
        mean_apw: float = 4,
        data: Optional[pd.DataFrame] = None,
        n_data: Optional[int] = None,
        **kwargs
    ):
        self.intercept = intercept
        self.features = ["base_price", "apw"]
        self.coefficients = coefficients
        self.n_data = 1000 if n_data is None else n_data
        self.mean_base_price = mean_base_price
        self.mean_apw = mean_apw

        super().__init__(data=data, **kwargs)

    def generate_data(self):
        prices = stats.skewnorm.rvs(
            100, loc=self.mean_base_price, scale=75, size=self.n_data
        )

        sigma = self.mean_apw * 1.2
        loc = self.mean_apw
        apws = stats.nbinom.rvs(
            n=loc**2 / (sigma**2 - loc),
            p=loc / sigma**2,
            size=self.n_data,
        )

        df = pd.DataFrame({"base_price": prices, "apw": apws})

        return df

    def demand_function(
        self,
        price,
        awp,
    ):
        y = (
            self.intercept
            + self.coefficients[0] * price
            + self.coefficients[1] * awp
            + 1 / 100 * np.random.randn()
        )
        y = 1 / (1 + np.exp(-y))

        return y
