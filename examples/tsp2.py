
from evo2 import Individual, Evolution, Selection

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
        super(TSP, self).__init__()
        self.city_names = []

    def pair(self, other, pair_params):
        # Based on https://www.researchgate.net/publication/236026740_AN_EFFICIENT_CROSSOVER_OPERATOR_FOR_TRAVELING_SALESMAN_PROBLEM

        offspring = TSP()

        if len(self.city_names) < 1:
            return offspring

        # The algorithm requires at least 1 starting city
        start_index = random.randint(1, len(self.city_names))
        offspring.city_names.extend(self.city_names[:start_index])

        visited = set(self.city_names[:start_index])

        for own_next, other_next in zip(self.city_names[start_index:], other.city_names[start_index:]):

            last_city = offspring.city_names[-1]
            dist_own = distance_matrix[last_city][own_next]
            dist_other = distance_matrix[last_city][other_next]

            min_dist = min(dist_own, dist_other)

            if min_dist == dist_own and own_next not in visited:
                offspring.city_names.append(own_next)
                visited.add(own_next)
            elif min_dist == dist_other and other_next not in visited:
                offspring.city_names.append(other_next)
                visited.add(other_next)
            else:
                for city in self.city_names:
                    if city not in visited:
                        offspring.city_names.append(city)
                        visited.add(city)
                        break

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
    40,
    n_offsprings=10,
    pair_params={"split_ratio": 0.1},
    mutate_params={"rate": 20},
    selection_method=Selection.random,
    fitness_func=fitness
)

#visualise_route(TSP())
#plt.show()

pre_optimisation = -fitness(evo.pool.individuals[0])

for i in tqdm(range(500)):
    best = evo.evolve()

    diversity = evo.pool.compute_diversity()
    if diversity != 0:
        evo.mutate_params["rate"] = int(500 / diversity)

    visualise_route(best[0], blocking=False)


visualise_route(best[0], blocking=False)
post_optimisation = -fitness(best[0])

print(round(pre_optimisation), round(post_optimisation))
print("Improvement:", (1-post_optimisation/pre_optimisation)*100, "%")

plt.show()