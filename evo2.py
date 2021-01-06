
from abc import ABC, abstractmethod
import operator
import math
import random
import itertools


# TODO: Tidy examples


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
        self.init_params = init_params
        self.individual_class = individual_class

        self.individuals = []
        for _ in range(size):
            individual = individual_class()
            individual.create(init_params)
            self.setup(individual)
            self.individuals.append(individual)

        self.sort_by_fitness()

    def create_random_individual(self):
        individual = self.individual_class()
        individual.create(self.init_params)
        self.add(individual)
        return individual

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
        # You can add either 1 or multiple individuals
        if not _is_iterable(new_individuals):
            new_individuals = [new_individuals]
        for individual in new_individuals:
            self.setup(individual)

        self.individuals.extend(new_individuals)
        self._is_sorted = False

    def sort_by(self, scoring_method):
        self.individuals.sort(key=scoring_method)
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


def _is_iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


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


# Other methods to preserve diversity:
# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.106.8662&rep=rep1&type=pdf
class SocialDisasters:
    @staticmethod
    def packing(population: Population, margin):
        population.sort_by_fitness()
        if len(population.individuals) < 1:
            return

        to_replace = []

        prev = population.individuals[0]
        for idx, individual in enumerate(population.individuals[1:]):
            diff = individual.fitness - prev.fitness
            if diff < margin:
                to_replace.append(idx+1)
            else:
                prev = individual

        for idx in reversed(to_replace):
            del population.individuals[idx]
            population.create_random_individual()

    @staticmethod
    def judgement_day(population: Population):
        if len(population.individuals) < 1:
            return

        population.sort_by_fitness()
        to_replace = len(population.individuals)-1
        del population.individuals[1:]

        for _ in range(to_replace):
            population.create_random_individual()


class Selection:
    @staticmethod
    def tournament(population: Population, n_offsprings: int, contenders_per_round=2):
        contenders = set(population.individuals)
        champions = []

        round_num = 0
        for round_num in range(min(n_offsprings*2, len(population.individuals)-1)):
            round_contenders = set(contenders.pop() for _ in range(contenders_per_round))
            winner = max(round_contenders, key=operator.attrgetter("fitness"))
            round_contenders.remove(winner)

            champions.append(winner)
            contenders.update(round_contenders)

        if len(contenders) != 0 and round_num != n_offsprings*2 -1:
            champions.extend(sorted(contenders, key=operator.attrgetter("fitness"), reverse=True))

        return _get_parents_from(champions)

    @staticmethod
    def roulette_wheel(population: Population, n_offsprings: int):
        non_visited = set(population.individuals)
        sorted_fitnesses = sorted(map(operator.attrgetter("fitness"), non_visited), reverse=True)
        total_fitness = sum(map(operator.attrgetter("fitness"), non_visited))

        # TODO: Remove redundant statements
        # (Heavily) relies on mutation to keep the algorithm fast
        def _choice_by_roulette(non_visited, sorted_fitnesses, total_fitness) -> tuple:
            lowest_fitness = sorted_fitnesses.pop()

            offset = 0
            if lowest_fitness < 0:
                offset = -lowest_fitness

            total_normalised_fitness = total_fitness + offset*len(non_visited)

            if total_normalised_fitness == 0:
                individual = non_visited.pop()
                total_fitness -= individual.fitness
                return total_fitness, individual 

            draw = random.random()
            accumulated = 0

            for individual in non_visited:
                normalised_fitness = individual.fitness + offset
                probability = normalised_fitness / total_normalised_fitness
                accumulated += probability

                if accumulated >= draw:
                    total_fitness -= individual.fitness
                    non_visited.remove(individual)
                    return total_fitness, individual

            if len(non_visited) > 0:
                individual = non_visited.pop()
                total_fitness -= individual.fitness
                return total_fitness, individual

        fathers = []
        mothers = []
        for _ in range(n_offsprings):
            total_fitness, mother = _choice_by_roulette(non_visited, sorted_fitnesses, total_fitness)
            total_fitness, father = _choice_by_roulette(non_visited, sorted_fitnesses, total_fitness)
            mothers.append(mother)
            fathers.append(father)

        return mothers, fathers

    @staticmethod
    def random(population: Population, n_offsprings: int):
        return _get_parents_from(random.sample(population.individuals, n_offsprings*2))

    @staticmethod
    def fittest(population: Population, n_offsprings: int):
        """
        Pairs the fittest of the population.

        Args:
            population:
            n_offsprings: The desired number of offsprings after pairing (parents = 2*n_offsprings)

        Returns:
            The mating pool: Tuple (mothers, fathers)
        """
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
                 fitness_func=None,
                 kill_before_spawn=True):

        """Creates a new evolution framework

        Args:
            individual_class: The type of the individuals of the population
            size: The size of the population => How many individuals
            n_offsprings: How many pairings per round
            init_params: The argument passed to the individuals create(...) method
            mutate_params: The argument passed to the individuals mutate(...) method
            pair_params: The argument passed to the individuals pair(...) method
            selection_method: A method that decides which individuals to pair together
            fitness_func: A function calculating the fitness of an individual
            kill_before_spawn: If true, the weakest individuals will be killed before adding the new offsprings
        """

        assert n_offsprings <= size / 2
        assert size >= 2
        assert n_offsprings >= 1
        assert selection_method is not None

        self.gen_number = 0
        self.stall_gens = 0

        self.mutate_params = mutate_params
        self.pair_params = pair_params
        self.n_offsprings = n_offsprings

        self.selection_method = selection_method

        self.population = Population(individual_class, fitness_func, size, init_params)
        self.kill_before_spawn = kill_before_spawn

    def get_best_fitness(self):
        return self.get_best_n(1)[0].fitness

    def get_best_n(self, n):
        self.population.sort_by_fitness()
        return self.population.individuals[:n]

    def _offsprings_from_parents(self, mother, father):
        offsprings = mother.pair(father, self.pair_params)

        # Sometimes, more than just one offspring is created
        if isinstance(offsprings, (list, tuple)):
            for o in offsprings:
                o.mutate(self.mutate_params)
                self.population.setup(o)
            return offsprings
        else:
            # Only one offspring
            offsprings.mutate(self.mutate_params)
            self.population.setup(offsprings)
            return [offsprings]

    def _offsprings_from_pool(self, mothers, fathers):
        nested = [
            self._offsprings_from_parents(mother, father)
            for mother, father
            in zip(mothers, fathers)]

        new_offsprings = list(itertools.chain.from_iterable(nested))
        return new_offsprings

    def evolve(self, iterations=1):
        for _ in range(iterations):
            self.next_gen()

        return self.population.individuals

    def next_gen(self):
        prev_best_fitness = self.get_best_fitness()

        mothers, fathers = self.selection_method(self.population, self.n_offsprings)
        new_offsprings = self._offsprings_from_pool(mothers, fathers)

        if self.kill_before_spawn:
            self.population.kill_weakest(len(new_offsprings))
        self.population.add(new_offsprings)
        if not self.kill_before_spawn:
            self.population.kill_weakest(len(new_offsprings))

        self.gen_number += 1

        current_best_fitness = self.get_best_fitness()
        if current_best_fitness > prev_best_fitness:
            self.stall_gens = 0
        else:
            self.stall_gens += 1

        return self.population.individuals
