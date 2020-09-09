
from abc import ABC, abstractmethod


# TODO: Add choice by roulette for crossovers https://hackernoon.com/genetic-algorithms-explained-a-python-implementation-sd4w374i


class Individual (ABC):
    @abstractmethod
    def pair(self, other, pair_params):
        pass

    @abstractmethod
    def mutate(self, mutate_params):
        pass

    @abstractmethod
    def create(self, init_params):
        pass

    def fitness(self):
        return None


class Population:
    def __init__(self, individual_class, default_fitness_func, size: int, init_params):
        self.default_fitness_func = default_fitness_func

        self.individuals = []
        for _ in range(size):
            individual = individual_class()
            individual.create(init_params)
            self.individuals.append(individual)

        self.individuals.sort(key=self.get_fitness, reverse=True)

    def get_fitness(self, individual):
        fitness = individual.fitness()
        if fitness is None:
            fitness = self.default_fitness_func(individual)
        return fitness

    def kill_weakest(self, n=1):
        del self.individuals[len(self.individuals)-n:]

    def add(self, new_individuals):
        self.individuals.extend(new_individuals)
        self.individuals.sort(key=self.get_fitness, reverse=True)

    def get_parents(self, n):
        mothers = self.individuals[0: n*2: 2]
        fathers = self.individuals[1: n*2+1: 2]

        return mothers, fathers


class Evolution:
    def __init__(self, individual_class, size, n_offsprings, init_params=None, mutate_params=None, pair_params=None, fitness_func=None):
        assert n_offsprings <= size / 2
        assert size >= 2
        assert n_offsprings >= 1

        self.mutate_params = mutate_params
        self.pair_params = pair_params
        self.n_offsprings = n_offsprings

        self.pool = Population(individual_class, fitness_func, size, init_params)

    def evolve(self, iterations=10):
        for _ in range(iterations):
            self.next_gen()

        return self.pool.individuals

    def next_gen(self):
        mothers, fathers = self.pool.get_parents(self.n_offsprings)

        offsprings = []
        for mother, father in zip(mothers, fathers):
            offspring = mother.pair(father, self.pair_params)
            offspring.mutate(self.mutate_params)
            offsprings.append(offspring)

        self.pool.kill_weakest(self.n_offsprings)
        self.pool.add(offsprings)

        return self.pool.individuals
