import os
import numpy as np

from pymoo.optimize import minimize
from pymoo.util import plotting
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymop.factory import get_problem
from pymop.problem import Problem


class Myproblem(Problem):
    def __init__(self, n_var=2, n_obj=2, n_constr=2, xl=-5, xu=5,
                 type_var=np.double):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl,
                         xu=xu, type_var=type_var)

    def _evaluate(self, x, f, *args, **kwargs):
        x1 = x[:, 0]
        x2 = x[:, 1]

        # Define custom goal function
        f[:, 0] = (np.power(x1, 4) - 10 * np.power(x1, 2) + x1 * x2 +
                   np.power(x2, 4) - np.power(x1, 2) * np.power(x2, 2))
        f[:, 1] = (np.power(x2, 4) - np.power(x1, 2) * np.power(x2, 2)
                   + np.power(x1, 4) + x1 * x2)

        # If use margin condition, use g to define. Notice the condition need
        # to normalize and < 0

class Viennet(Problem):
    def __init__(self, n_var=2, n_obj=3, n_constr=0, xl=-3, xu=3,
                 type_var=np.double):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl,
                         xu=xu, type_var=type_var)

    def _evaluate(self, x, f, *args, **kwargs):
        x1 = x[:, 0]
        x2 = x[:, 1]

        # Define custom goal function
        f[:, 0] = (0.5 * (np.power(x1, 2) + np.power(x2, 2)) +
                   np.sin(np.power(x1, 2) + np.power(x2, 2)))
        f[:, 1] = (1 / 8 * np.power(3 * x1 - 2 * x2 + 4, 2) +
                   1 / 27 * np.power(x1 - x2 + 1, 2) + 15)
        f[:, 2] = (1 / (np.power(x1, 2) + np.power(x2, 2) + 1) -1.1 *
                   np.exp(-(np.power(x1, 2) + np.power(x2, 2))))


def use_package_function(func_name):

    # Parameter setting
    n_var = 7
    n_obj = 3
    n_points = 100
    iter_epoch = 500
    random_seed = 5

    problem = get_problem(func_name, n_var=n_var, n_obj=n_obj)
    # problem = get_problem(func_name)

    # create the reference directions to be used for the optimization
    ref_dirs = UniformReferenceDirectionFactory(n_obj, n_points=n_points).do()

    res = minimize(problem,
                   method='nsga3',
                   method_args={
                       'pop_size': n_points+1,
                       'ref_dirs': ref_dirs},
                   termination=('n_gen', iter_epoch),
                   # pf=pf,
                   seed=random_seed,
                   disp=True)
    plotting.plot(res.F)


def use_custom_function():
    # Create a custom problem.

    # Parameter setting
    n_var = 2
    n_obj = 2
    n_constr = 0
    x_min = -5
    x_max = 5
    n_points = 100
    iter_epoch = 500
    random_seed = 5

    problem = Myproblem()
    # problem = Viennet()

    # create the reference directions to be used for the optimization
    ref_dirs = UniformReferenceDirectionFactory(
        n_dim=n_obj, n_points=n_points).do()

    res = minimize(
        problem, method='nsga3', method_args={
            'pop_size': n_points + 1, 'ref_dirs': ref_dirs},
        termination=('n_gen', iter_epoch), seed=random_seed, disp=True)
    plotting.plot(res.F)


def main():
    savepath1 = './result/package/'
    savepath2 = './result/custom/'
    for path in [savepath1, savepath2]:
        if not os.path.exists(path):
            os.makedirs(path)

    # use_custom_function()
    use_package_function('dtlz1')
    # use_package_function('osy')


if __name__ == '__main__':
    main()
