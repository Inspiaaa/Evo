
from evo2 import Individual, Evolution, Selection, SocialDisasters
import random

try:
    from tqdm import trange
except ImportError:
    trange = range


def _clamp(val, minval, maxval):
    return min(max(val, minval), maxval)


class Optimisation (Individual):
    __slots__ = "params"

    def __init__(self):
        super().__init__()
        self.params = []

    def create(self, init_params):
        for lower, upper in zip(init_params["lower"], init_params["upper"]):
            self.params.append(random.uniform(lower, upper))

    def mutate(self, mutate_params):
        for i in range(len(self.params)):
            val = self.params[i]
            val += random.uniform(mutate_params["lower"][i], mutate_params["upper"][i]) * mutate_params["intensity"]
            val = _clamp(val, mutate_params["lower"][i], mutate_params["upper"][i])
            self.params[i] = val

    def pair(self, other, pair_params):
        offspring = Optimisation()
        split_pos = int(random.random() * len(self.params))
        offspring.params = self.params[:split_pos] + other.params[split_pos:]
        return offspring


def maximise_multi_param(
        func,
        lower_bounds,
        upper_bounds,
        search_gens=200,
        maximisation_gens=200,
        population_size=100,
        max_stall_gens=200,
        min_search_diversity=50):

    evo = Evolution(
        Optimisation,
        population_size,
        n_offsprings=population_size//2,
        init_params={"lower": lower_bounds, "upper": upper_bounds},
        mutate_params={"lower": lower_bounds, "upper": upper_bounds, "intensity": 0.1},
        pair_params={},
        selection_method=Selection.roulette_wheel,
        fitness_func=lambda i: func(*i.params)
    )

    # Search pass: Try to expand and overcome local minima
    for _ in trange(search_gens, leave=False):
        evo.evolve()
        diversity = evo.population.compute_diversity()
        if diversity < min_search_diversity:
            # Randomly re-initialise solutions that are too similar
            SocialDisasters.packing(evo.population, min_search_diversity * 0.1)

        if evo.stall_gens > max_stall_gens:
            break

    # Maximisation pass: Converge and find the best possible solution
    evo.selection_method = Selection.fittest
    for _ in trange(maximisation_gens, leave=False):
        evo.evolve()

        if evo.stall_gens > max_stall_gens:
            break

    return evo.get_best_n(1)[0].params


if __name__ == '__main__':
    from math import *

    def cost(x, y):
        # https://en.wikipedia.org/wiki/Test_functions_for_optimization
        # Eggholder function
        # Correct solution: x=512, y=404.2319

        # `-` before calculation to reformulate the minimisation problem into a maximisation problem
        return - (
                - (y + 47) * sin(sqrt(abs(x/2 + (y + 47))))
                - x * sin(sqrt(abs(abs(x - (y + 47)))))
        )


    print(maximise_multi_param(cost, lower_bounds=[-512, -512], upper_bounds=[512, 512]))
