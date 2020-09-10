
from evo import Individual, Evolution

import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


random.seed(100)


city_names = [str(i) for i in range(20)]
city_positions = {name: (random.randint(0, 100), random.randint(0, 100)) for name in city_names}


distance_matrix = {}
for start_city in city_names:
    local_distances = {}

    for end_city in city_names:
        start = city_positions[start_city]
        end = city_positions[end_city]
        dist = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)

        local_distances[end_city] = dist

    distance_matrix[start_city] = local_distances


def fitness(solution):
    if len(solution.city_names) < 2:
        return 0

    distance = 0
    prev_city = solution.city_names[0]

    # Use a local variable to speed it up
    cached_distance_matrix = distance_matrix
    for next_city in solution.city_names[1:]:
        distance += cached_distance_matrix[prev_city][next_city]
        prev_city = next_city

    return -distance


class TSP (Individual):
    __slots__ = "city_names"

    def __init__(self):
        self.city_names = []

    def pair(self, other, pair_params):
        split_pos = int(pair_params["split_ratio"] * len(self.city_names))

        own_head = self.city_names[:split_pos]
        own_tail = self.city_names[split_pos:]
        other_tail = other.city_names[split_pos:]

        duplicates = set(own_head) & set(other_tail)
        replacements = list(set(own_tail) - set(other_tail))

        offspring_cities = own_head + [city if city not in duplicates else replacements.pop() for city in other_tail]

        offspring = TSP()
        offspring.city_names = offspring_cities
        return offspring

    def mutate(self, mutate_params):
        indices = range(len(self.city_names))

        for _ in range(random.randint(0, mutate_params["rate"])):
            a, b = random.sample(indices, 2)
            self.city_names[a], self.city_names[b] = self.city_names[b], self.city_names[a]

    def create(self, init_params):
        self.city_names = city_names.copy()
        random.shuffle(self.city_names)


def visualise_route(solution, blocking=True):
    plt.clf()
    cities_x = [city_positions[name][0] for name in city_names]
    cities_y = [city_positions[name][1] for name in city_names]
    plt.scatter(x=cities_x, y=cities_y, s=500, zorder=1)

    if len(solution.city_names) < 2:
        return

    prev = solution.city_names[0]
    for name in solution.city_names[1:]:
        cities_x = city_positions[prev][0], city_positions[name][0]
        cities_y = city_positions[prev][1], city_positions[name][1]
        plt.plot(cities_x, cities_y, c="grey", zorder=0)
        prev = name

    if blocking:
        plt.show()
    else:
        plt.draw()
        plt.pause(0.001)


evo = Evolution(
    TSP,
    100,
    n_offsprings=25,
    pair_params={"split_ratio": 0.1},
    mutate_params={"rate": 20},
    fitness_func=fitness
)

#visualise_route(TSP())
#plt.show()

pre_optimisation = -fitness(evo.pool.individuals[0])

for i in tqdm(range(250)):
    best = evo.evolve()
    visualise_route(best[0], blocking=False)

visualise_route(best[0], blocking=False)
post_optimisation = -fitness(best[0])

print(round(pre_optimisation), round(post_optimisation))
print("Improvement:", (1-post_optimisation/pre_optimisation)*100, "%")

plt.show()