import numpy as np

from pymoo.algorithms.nsga3 import calc_niche_count, niching, get_extreme_points_c, associate_to_niches
from pymoo.model.survival import Survival
from pymoo.util.non_dominated_sorting import NonDominatedSorting


class ReferenceDirectionSurvivalNonDominated(Survival):
    def __init__(self, ref_dirs):
        super().__init__(True)
        self.ref_dirs = ref_dirs
        self.extreme_points = None
        self.intercepts = None
        self.nadir_point = None
        self.ideal_point = np.full(ref_dirs.shape[1], np.inf)
        self.worst_point = np.full(ref_dirs.shape[1], -np.inf)

    def _do(self, pop, n_survive, D=None, **kwargs):

        # attributes to be set after the survival
        F = pop.get("F")

        # find or usually update the new ideal point - from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        # calculate the fronts of the population
        fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
        self.nadir_point = get_nadir_point_from_fronts(F, fronts, self.ideal_point)

        #  consider only the population until we come to the splitting front
        I = np.concatenate(fronts)
        pop, rank, F = pop[I], rank[I], F[I]

        # update the front indices for the current population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # associate individuals to niches
        niche_of_individuals, dist_to_niche = associate_to_niches(F, self.ref_dirs, self.ideal_point, self.nadir_point)
        pop.set('rank', rank, 'niche', niche_of_individuals, 'dist_to_niche', dist_to_niche)

        # if we need to select individuals to survive
        if len(pop) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=np.int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=np.int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            S = niching(F[last_front, :], n_remaining, niche_count, niche_of_individuals[last_front],
                        dist_to_niche[last_front])

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            pop = pop[survivors]

        return pop


def get_nadir_point_from_fronts(F, fronts, ideal_point, epsilon=10e-3):
    n_obj = F.shape[1]
    nadir_point = np.full(n_obj, np.inf)

    for m in range(n_obj):
        for k in range(len(fronts)):
            nadir_point[m] = np.max(F[fronts[k], m])
            # at least epsilon different OR we are in the last front already
            if nadir_point[m] - ideal_point[m] > epsilon:
                break

    return nadir_point


