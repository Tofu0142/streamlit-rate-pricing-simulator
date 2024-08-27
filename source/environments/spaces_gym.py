import random 
import scipy.stats as stats
from typing import Optional
from gymnasium import spaces
import numpy as np
np.random.seed(seed=233423)

from room_pricing.bandits.bayesian_model import spaces as custom_spaces


class Simulator1_Price_Space(custom_spaces.BaseSpace):
    def __init__(self, mean_base_price: float):
        self._shape = (1,)
        self.dtype = np.float32
        self.mean_base_price = mean_base_price
        self.location_quality_dist = stats.norm(loc=7, scale=3)
        self.property_rating_dist = stats.norm(loc=6, scale=1)
        self.seasonal_demand_dist = stats.uniform(0.8, 1.2)
        self.dist = stats.norm(loc=mean_base_price, scale=1)

    def sample(self, size: int = 1):
        location_quality_samples = self.location_quality_dist.rvs(size)
        property_rating_samples = self.property_rating_dist.rvs(size)
        seasonal_demand_samples = self.seasonal_demand_dist.rvs(size)

        price_samples = self.mean_base_price + location_quality_samples * 10 * 0.4 + property_rating_samples * 20 * 0.4 + seasonal_demand_samples * 0.2

        if size == 1:
            return price_samples[0]
        else:
            return price_samples
    
    def contains(self, x):
        return np.isscalar(x) and x >= 0
    

class Simulator2_Price_Space(custom_spaces.BaseSpace):
    def __init__(self, mean_base_price: float):
        self._shape = (1,)
        self.dtype = np.float32
        self.mean_base_price = mean_base_price
    
    def _get_season_factor(self, is_peak_season: bool):
        if is_peak_season:
            return np.random.normal(loc=1.2, scale=0.1)
        else:
            return np.random.normal(loc=1.0, scale=0.05)
        
    def _get_day_factor(self, is_weekend: bool):
        if is_weekend:
            return np.random.normal(loc=1.1, scale=0.05)
        else:
            return np.random.normal(loc=1.0, scale=0.03)
        
    def _calculate_price(self, is_peak_season: bool, is_weekend: bool, room_type: str):
        seasonal_factor = self._get_season_factor(is_peak_season)
        day_factor = self._get_day_factor(is_weekend)

        room_type_factor = {'standard': 1.0, 'deluxe': 1.2, 'suite': 1.5}
        room_factor = room_type_factor.get(room_type, 1.0)
        return self.mean_base_price * seasonal_factor * day_factor * room_factor
    
    def sample(self, size: int = 1):
        price_samples = []
        for _ in range(size):
            is_peak_season = random.choice([True, False])
            is_weekend = random.choice([True, False])
            room_type = random.choice(['standard', 'deluxe', 'suite'])
            price = self._calculate_price(is_peak_season, is_weekend, room_type)
            price_samples.append(price)
        if size == 1:
            return price_samples[0]
        else:
            return price_samples
        
    def contains(self, x):
        return np.isscalar(x) and x >= 0


class NegBinomialSpace(custom_spaces.BaseSpace):
    def __init__(self, mean: float, std: Optional[float] = None):
        self._shape = (1,)
        self.dtype = np.float32
        self.loc = mean
        if std is None:
            self.scale = self.loc * 1.2
        else:
            self.scale = std

        self.dist = stats.nbinom(
            n=self.loc**2 / (self.scale**2 - self.loc), p=self.loc / self.scale**2
        )

    def contains(self, x):
        return np.isscalar(x) and x >= 0
    

class BetaDistributionSpace(custom_spaces.BaseSpace):
    def __init__(self, alpha: float, beta: float):
        self._shape = (1,)
        self.dtype = np.float32
        self.alpha = alpha
        self.beta = beta
    
    def sample(self, size: int = 1):
        sample = stats.beta.rvs(self.alpha, self.beta, size=size) * 10
        sample = np.ceil(sample)
        if size == 1:
            return sample[0].astype(int)
        else:
            return [int(x) for x in sample]

    def contains(self, x):
        return np.isscalar(x) and x >= 0


class UniformSpace(custom_spaces.BaseSpace):
    def __init__(self, low: float, scale: float):
        self._shape = (1,)
        self.dtype = np.float32
        self.loc = low
        self.scale = scale
    
    def sample(self, size: int = 1):
        sample = stats.uniform.rvs(self.loc, self.scale, size=size)
        if size == 1:
            return sample[0]
        else:
            return sample

    def contains(self, x):
        return np.isscalar(x) and x >= 0
    

class NormalizedDiscreteSpace(spaces.Discrete):
    def __init__(
        self,
        end: int = 1,
        *args,
        **kwargs,
    ):
        self.end = end
        self._start = 1

        start = kwargs.get("start", None)
        if start is None:
            start = 0
        kwargs["start"] = 1
        self._start = start

        super().__init__(*args, **kwargs)
        self.alpha = (self.n * self._start - self.end) / (self.n - 1)
        self.beta = (self.end - self._start) / (self.n - 1)

    def to_list(self):
        return list(np.linspace(self._start, self.end, self.n))

    def sample(self, **kwargs):
        return np.array([self.alpha + self.beta * super().sample()], dtype=float)

class RandomSpace(custom_spaces.BaseSpace):
    def __init__(self, low: float, high: float):
        self._shape = (1,)
        self.dtype = np.float32
        self.low = low
        self.high = high

    def sample(self, size: int = 1):
        sample = [random.randint(self.low, self.high) for _ in range(size)]
        if size == 1:
            return sample[0]
        else:
            return sample
        

custom_spaces = {
    "base" : custom_spaces.BaseSpace,
    "simulator1_price" : Simulator1_Price_Space,
    "simulator2_price" : Simulator2_Price_Space,
    "negative_binomial" : NegBinomialSpace,
    "beta_distribution" : BetaDistributionSpace,
    "uniform" : UniformSpace,
    "discrete" : NormalizedDiscreteSpace,
    "random" : RandomSpace
}