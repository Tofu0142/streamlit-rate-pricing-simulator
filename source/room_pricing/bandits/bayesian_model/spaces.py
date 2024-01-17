import scipy.stats as stats
from typing import Optional
from gymnasium import spaces


class BaseSpace(spaces.Space):
    dist = None

    def sample(self, size: int = 1):
        result = self.dist.rvs(size)
        if size == 1:
            return result[0]
        else:
            return result


class SkewNormSpace(BaseSpace):
    def __init__(self, mean: float, std: float = 75, a: float = 100):
        self.loc = mean
        self.scale = std
        self.a = a

        self.dist = stats.skewnorm(a=a, loc=self.loc, scale=self.scale)


class NegBinomialSpace(BaseSpace):
    def __init__(self, mean: float, std: Optional[float] = None):
        self.loc = mean
        if std is None:
            self.scale = self.loc * 1.2
        else:
            self.scale = std

        self.dist = stats.nbinom(
            n=self.loc**2 / (self.scale**2 - self.loc), p=self.loc / self.scale**2
        )


class NormalizedDiscreteSpace(spaces.Discrete):
    def __init__(
        self,
        boundary: int = 1,
        *args,
        **kwargs,
    ):
        self.boundary = boundary
        super().__init__(*args, **kwargs)

    def to_list(self):
        return np.linspace(self.start, self.boundary, self.n)

    def sample(self, **kwargs):
        return np.array([super().sample() / self.n], dtype=float)

