from pymoo.model.sampling import Sampling
from pymoo.rand import random


class RandomSampling(Sampling):
    """
    Randomly sample points in the real space by considering the lower and upper bounds of the problem.
    """

    def sample(self, problem, pop, n_samples, **kwargs):

        m = problem.n_var
        val = random.random(size=(n_samples, m))
        for i in range(m):
            val[:, i] = val[:, i] * (problem.xu[i] - problem.xl[i]) + problem.xl[i]

        return pop.new("X", val)
