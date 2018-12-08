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


def use_package_function(func_name, savepath):

    # Parameter setting
    n_var = 7
    n_obj = 3
    n_points = 91
    iter_epoch = 100
    random_seed = 5

    problem = get_problem(func_name, n_var, n_obj)

    # create the reference directions to be used for the optimization
    ref_dirs = UniformReferenceDirectionFactory(n_obj, n_points).do()

    # create the pareto front for the given reference lines
    pf = problem.pareto_front(ref_dirs)

    res = minimize(problem,
                   method='nsga3',
                   method_args={
                       'pop_size': n_points + 1,
                       'ref_dirs': ref_dirs},
                   termination=('n_gen', iter_epoch),
                   pf=pf,
                   seed=random_seed,
                   disp=True)
    plotting.plot(res.F, savepath)


def use_custom_function(savepath):
    # Create a custom problem.

    # Parameter setting
    n_var = 2
    n_obj = 2
    n_constr = 0
    x_min = -5
    x_max = 5
    n_points = 91
    iter_epoch = 100
    random_seed = 5

    problem = Myproblem(n_var, n_obj, n_constr, x_min, x_max)

    # create the reference directions to be used for the optimization
    ref_dirs = UniformReferenceDirectionFactory(
        n_dim=n_obj, n_points=n_points).do()

    res = minimize(
        problem, method='nsga3', method_args={
            'pop_size': n_points + 1, 'ref_dirs': ref_dirs},
        termination=('n_gen', iter_epoch), seed=random_seed, disp=True)
    plotting.plot(savepath, res.F)


def main():
    savepath = './result/custom/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    use_custom_function(savepath=savepath)
    # use_package_function('dtlz1', './result/package/')


if __name__ == '__main__':
    main()
