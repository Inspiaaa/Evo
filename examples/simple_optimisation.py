
from evo2 import Individual, Evolution

import random
import matplotlib.pyplot as plt
import time


class Optimisation (Individual):
    __slots__ = "x"

    def __init__(self):
        self.x = 0
        super().__init__()

    def __repr__(self):
        return str(self.x)

    def create(self, init_params):
        self.x = random.uniform(init_params["lower"], init_params["upper"])

    def pair(self, other, pair_params):
        offspring = Optimisation()
        offspring.x = self.x + (other.x - self.x) / 2
        return offspring

    def mutate(self, mutate_params):
        self.x += random.uniform(-mutate_params["intensity"], +mutate_params["intensity"])
        self.x = max(mutate_params["lower"], min(self.x, mutate_params["upper"]))


def curve(x):
    return -x*(x-1)*(x-2)*(x-3)*(x-4)


init_params = {"lower": 0, "upper": 4}
mutate_params = {"intensity": 0.25, **init_params}

evo = Evolution(
    Optimisation,
    20,
    n_offsprings=10,
    init_params=init_params,
    mutate_params=mutate_params,
    fitness_func=lambda obj: curve(obj.x))


curve_y = [curve(x/100) for x in range(init_params["lower"]*100, init_params["upper"]*100)]
curve_x = [x/100 for x in range(init_params["lower"]*100, init_params["upper"]*100)]

plt.plot(curve_x, curve_y)


prev_individuals = []
for it in range(100):
    for i in prev_individuals:
        i.remove()
    prev_individuals.clear()

    fittest = evo.next_gen()

    for i in fittest:
        point = plt.scatter(i.x, curve(i.x), color="r")
        prev_individuals.append(point)

    plt.draw()
    plt.pause(0.001)
    time.sleep(0.1)

plt.show()