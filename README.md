# Evo

A lightweight library for creating genetic algorithms with ease in Python.

- 100% pure python

- Lightweight: No external dependencies / packages

- Quick start template for multi-parameter optimisation problems

## Getting started

1. Copy the `evo2.py` file into your own project

2. Import the file and create a class which will represent an individual of the population

```python
from evo2 import Individual, Evolution, Selection
import random


class Optimisation (Individual):
   # __slots__ makes the individual class use less memory
   __slots__ = ("x")

   def __init__(self):
       super().__init__()
       self.x = 0

   # You can pass in your own data for initialisation
   # Although a dictionary is handy for that, any data type can be used
   def create(self, init_params):
       # Here you randomly initialise the individual
       self.x = random.uniform(init_params["lower"], init_params["upper"])

   # Mutate is used for introducing some randomness after pairing
   def mutate(self, mutate_params):
       self.x += random.random() * mutate_params["intensity"]
       # Clamp the x value into the desired range, if it goes over
       self.x = min(mutate_params["upper"], min(mutate_params["lower"], self.x))

   # Create a new offspring (Also known as the crossover operator)
   def pair(self, other, pair_params):
       offspring = Optimisation()
       offspring.x = (self.x + other.x) / 2
```

3. Create a fitness function

```python
def curve(x):
    return -x*(x-1)*(x-2)*(x-3)*(x-4)
```

4. Initialise a population

```python
evo = Evolution(
    Optimisation,
    size=20,
    n_offsprings=10,
    selection_method=Selection.tournament
    init_params={"lower": 0, "upper": 4},
    mutate_params={"lower": 0, "upper": 4},
    fitness_func=lambda obj: curve(obj.x))
```

5. Run the algorithm

```python
# Run 100 generations
evo.evolve(100)
```

6. Get the best individual

```python
best = evo.get_best_n(1)[0]
best_fitness = best.fitness
```

### Other features

- Computing the diversity: Evo uses the standard deviation of the fitness values to describe the diversity of the population

```python
diversity = evo.population.compute_diversity()
```

- Stopping after a certain number of generations without improvement

```python
while True:
    evo.evolve()

    # Stop after 50 generations of no improvement
    if evo.stall_gens > 50:
        break
```

- Preserving diversity by using "social disasters"

```python
from evo2 import SocialDisasters

for i in range(100):
    diversity = evo.population.compute_diversity()

    if diversity < 50:
        # Randomly re-initialise individuals that are too similar
        SocialDisasters.packing(evo.population, 10)

        # OR only keep the best individual and randomly re-initialise all others
        SocialDisasters.judgement_day(evo.population)
```

- Getting the generation number:
  
  ```python
  print(f"Generation #{ evo.gen_number }")
  ```



### Using the optimisation template

The Evo library has a builtin template for multi parameter optimisation problems

1. Copy the `evo_templates.py` file in to your workspace (as well as the main Evo file itself)

2. Define the problem
   
   ```python
   # 3 parameter function, taking 3 floats
   def cost(a, b, c):
       return (a+1)*(b+2)*(c+3)*(a-b-c)*(c-b-a)
   ```

3. Let the library optimise for you
   
   ```python
   from evo_templates import maximise_multi_param
   
   a, b, c = maximise_multi_param(cost, lower_bounds=[-2, -2, -2], upper_bounds=[2, 2, 2]))
   print(a, b, c)
   ```
   
   
