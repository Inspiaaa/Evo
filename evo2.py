
from abc import ABC, abstractmethod
import operator
import math
import random
import itertools


# TODO: Add choice by roulette for crossovers https://hackernoon.com/genetic-algorithms-explained-a-python-implementation-sd4w374i


class Individual (ABC):
    __slots__ = "fitness",

    def __init__(self):
        self.fitness = None

    @abstractmethod
    def pair(self, other, pair_params):
        pass

    @abstractmethod
    def mutate(self, mutate_params):
        pass

    @abstractmethod
    def create(self, init_params):
        pass

    def compute_fitness(self):
        pass


class Population:
    def __init__(self, individual_class, default_fitness_func, size: int, init_params):
        self.default_fitness_func = default_fitness_func
        self._is_sorted = False

        self.individuals = []
        for _ in range(size):
            individual = individual_class()
            individual.create(init_params)
            self.setup(individual)
            self.individuals.append(individual)

        self.sort_by_fitness()

    def setup(self, individual: Individual):
        """Sets up and computes the fitness of an individual for future usage.

        individual.compute_fitness() is used, but if it has no effect, the default fitness function that was passed
        in will be used

        Args:
            individual: The individual instance to be set up
        """

        if individual.fitness is not None:
            return

        individual.compute_fitness()
        if individual.fitness is None:
            individual.fitness = self.default_fitness_func(individual)

    def kill_weakest(self, n=1):
        self.sort_by_fitness()
        del self.individuals[len(self.individuals)-n:]

    def add(self, new_individuals):
        for individual in new_individuals:
            self.setup(individual)

        self.individuals.extend(new_individuals)
        self._is_sorted = False

    def sort_by_fitness(self):
        if self._is_sorted:
            return

        self.individuals.sort(key=operator.attrgetter("fitness"), reverse=True)
        self._is_sorted = True

    def compute_diversity(self):
        # Calculate the diversity based on the standard deviation of the fitness values
        mean = sum(i.fitness for i in self.individuals) / len(self.individuals)
        variance = sum((i.fitness - mean) ** 2 for i in self.individuals) / len(self.individuals)
        standard_deviation = math.sqrt(variance)
        return standard_deviation

    def get_parents(self, n):
        mothers = self.individuals[0: n*2: 2]
        fathers = self.individuals[1: n*2+1: 2]

        return mothers, fathers


def _grouper(n, iterable, fillvalue=None):
    """grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


def _get_parents_from(selected, n_offsprings=None):
    if n_offsprings is None:
        n_offsprings = len(selected) // 2

    mothers = selected[0: n_offsprings * 2: 2]
    fathers = selected[1: n_offsprings * 2 + 1: 2]

    return mothers, fathers


class Selection:
    @staticmethod
    def tournament(population: Population, n_offsprings: int, contenders_per_round=2):
        contenders = random.sample(population.individuals, n_offsprings*2*contenders_per_round)

        champions = [
            max(round_contenders, key=operator.attrgetter("fitness"))
            for round_contenders
            in _grouper(contenders_per_round, contenders)
        ]

        return _get_parents_from(champions)

    @staticmethod
    def roulette_wheel(population: Population, n_offsprings: int):
        pass

    @staticmethod
    def random(population: Population, n_offsprings: int):
        return _get_parents_from(random.sample(population.individuals, n_offsprings*2))

    @staticmethod
    def fittest(population: Population, n_offsprings: int):
        population.sort_by_fitness()
        return _get_parents_from(population.individuals, n_offsprings)


class Evolution:
    def __init__(self,
                 individual_class,
                 size,
                 n_offsprings,
                 init_params=None,
                 mutate_params=None,
                 pair_params=None,
                 selection_method=Selection.fittest,
                 fitness_func=None):

        print(selection_method)

        assert n_offsprings <= size / 2
        assert size >= 2
        assert n_offsprings >= 1
        assert selection_method is not None

        self.mutate_params = mutate_params
        self.pair_params = pair_params
        self.n_offsprings = n_offsprings

        self.selection_method = selection_method

        self.pool = Population(individual_class, fitness_func, size, init_params)

    def get_best(self, n):
        self.pool.sort_by_fitness()
        return self.pool.individuals[:n]

    def evolve(self, iterations=1):
        for _ in range(iterations):
            self.next_gen()

        return self.pool.individuals

    def next_gen(self):
        mothers, fathers = self.selection_method(self.pool, self.n_offsprings)

        offsprings = []
        for mother, father in zip(mothers, fathers):
            offspring = mother.pair(father, self.pair_params)
            offspring.mutate(self.mutate_params)
            offsprings.append(offspring)

        self.pool.kill_weakest(self.n_offsprings)
        self.pool.add(offsprings)

        return self.pool.individuals
