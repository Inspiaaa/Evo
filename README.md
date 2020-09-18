# Evo

A lightweight library for creating genetic algorithms with ease in Python.

- 100% pure python

- Lightweight: No external dependencies / packages

- Quick start template for multi-parameter optimisation problems



## Getting started

1. Copy the `evo2.py` file into your own project

2. Import the file and create a class which will represent an individual of the population
   
   ```python
   from evo2 import Individual, Evolution
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


