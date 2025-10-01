import numpy as np
from typing import Dict, Tuple

from evga.fitness import evaluate_solution
from evga.operators import (
    initialize_population,
    crossover,
    mutation,
    tournament_selection,
)


class GeneticAlgorithm:
    """
    A simple class-based Genetic Algorithm for the EVGA problem.
    """

    def __init__(
        self,
        data: Dict,
        pop_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        tournament_k: int = 3,
        random_seed: int = None,
    ):
        """
        Initialize GA parameters.
        """
        self.data = data
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k

        if random_seed is not None:
            np.random.seed(random_seed)

        # Internal state
        self.population = None
        self.fitnesses = None
        self.best_solution = None
        self.best_fitness = np.inf  # lower is better

    def initialize(self):
        """
        Create initial population and evaluate fitness.
        """
        self.population = initialize_population(self.pop_size, self.data)
        self.fitnesses = np.array(
            [evaluate_solution(ind, self.data) for ind in self.population]
        )
        self.update_best()

    def update_best(self):
        """
        Update best solution so far.
        """
        min_idx = np.argmin(self.fitnesses)
        if self.fitnesses[min_idx] < self.best_fitness:
            self.best_fitness = self.fitnesses[min_idx]
            self.best_solution = self.population[min_idx].copy()

    def evolve(self):
        """
        Run one generation: selection, crossover, mutation, evaluation.
        """
        new_population = []

        while len(new_population) < self.pop_size:
            # Selection
            parent1 = tournament_selection(
                self.population, self.fitnesses, self.tournament_k
            )
            parent2 = tournament_selection(
                self.population, self.fitnesses, self.tournament_k
            )

            # Crossover
            if np.random.rand() < self.crossover_rate:
                children = crossover(parent1, parent2)
            else:
                children = [parent1.copy(), parent2.copy()]

            # Mutation
            children = [
                mutation(child, self.data, self.mutation_rate) for child in children
            ]

            new_population.extend(children)

        # Trim if overfilled
        self.population = np.array(new_population[: self.pop_size])
        self.fitnesses = np.array(
            [evaluate_solution(ind, self.data) for ind in self.population]
        )
        self.update_best()

    def run(self) -> Tuple[np.ndarray, float]:
        """
        Run the genetic algorithm.

        Returns:
            (best_solution, best_fitness)
        """
        self.initialize()
        for gen in range(self.generations):
            self.evolve()
            print(
                f"Generation {gen+1}/{self.generations}, "
                f"best fitness so far: {self.best_fitness:.2f}"
            )
            # Print the population of the current generation
            print("Population (first 5 individuals):")
            print(self.population[:5])  # printing only first 5 for readability
            print("-----")

        return self.best_solution, self.best_fitness

