
import multiprocessing
import itertools
import functools


_original_func_name = "_multicore_original_func"
_worker_pool: multiprocessing.Pool = None


# TODO: Fix for the use of lambdas in evo.fitness_func (Remove that dependency)


class _OnExit:
    def __init__(self, then):
        self.then = then

    def __exit__(self):
        self.then()


def _parallel_offspring_processing(evo, mothers, father):
    nested = worker_pool.starmap(
        func=evo._offsprings_from_parents,
        iterable=[(m, f) for m, f in zip(mothers, father)]
    )

    new_offsprings = list(itertools.chain.from_iterable(nested))
    return new_offsprings


def deparallelise(evo):
    if hasattr(evo, _original_func_name):
        evo._offsprings_from_pool = getattr(evo, _original_func_name)
        delattr(evo, _original_func_name)
    worker_pool.close()


def parallelise(evo, num_cpus=6):
    global worker_pool
    worker_pool = multiprocessing.Pool(num_cpus)

    setattr(evo, _original_func_name, evo._offsprings_from_pool)
    evo._offsprings_from_pool = functools.partial(_parallel_offspring_processing, evo)

    return _OnExit(lambda: deparallelise(evo))
