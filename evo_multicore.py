
import multiprocessing
import itertools
import functools

from evo2 import Evolution


worker_pool: multiprocessing.Pool = None


def _parallel_offspring_processing(evo, mothers, father):
    nested = worker_pool.starmap(
        func=evo._offsprings_from_parents,
        iterable=[(m, f) for m, f in zip(mothers, father)],
        chunksize=10
    )

    new_offsprings = list(itertools.chain.from_iterable(nested))
    return new_offsprings


def parallelise(evo: Evolution, num_cpus=6):
    global worker_pool
    worker_pool = multiprocessing.Pool(num_cpus)
    evo._offsprings_from_pool = functools.partial(_parallel_offspring_processing, evo)
