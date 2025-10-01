import numpy as np
from typing import Dict, List


def initialize_population(pop_size: int, data: Dict) -> np.ndarray:
    """
    Initialize a population of random solutions.

    Args:
        pop_size: number of individuals.
        data: problem data dict (expects 'n_sites', 'max_outlets_per_site').

    Returns:
        Population as np.ndarray of shape (pop_size, n_sites).
    """
    n_sites = data["n_sites"]
    max_outlets = np.array(data["max_outlets_per_site"], dtype=int)

    population = np.zeros((pop_size, n_sites), dtype=int)
    for i in range(pop_size):
        for j in range(n_sites):
            population[i, j] = np.random.randint(0, max_outlets[j] + 1)
    return population


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> List[np.ndarray]:
    """
    Perform one-point crossover.

    Args:
        parent1: np.ndarray, first parent.
        parent2: np.ndarray, second parent.

    Returns:
        Two offspring solutions.
    """
    n_sites = len(parent1)
    if n_sites < 2:
        return [parent1.copy(), parent2.copy()]

    point = np.random.randint(1, n_sites)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return [child1, child2]


def mutation(individual: np.ndarray, data: Dict, mutation_rate: float = 0.1) -> np.ndarray:
    """
    Mutate an individual by randomly adjusting outlet counts.

    Args:
        individual: np.ndarray representing solution.
        data: problem data dict (expects 'max_outlets_per_site').
        mutation_rate: probability of mutating each site.

    Returns:
        Mutated individual (new copy).
    """
    max_outlets = np.array(data["max_outlets_per_site"], dtype=int)
    mutant = individual.copy()

    for j in range(len(mutant)):
        if np.random.rand() < mutation_rate:
            mutant[j] = np.random.randint(0, max_outlets[j] + 1)
    return mutant


def tournament_selection(
    population: np.ndarray, fitnesses: np.ndarray, k: int = 3
) -> np.ndarray:
    """
    Tournament selection.

    Args:
        population: np.ndarray of shape (pop_size, n_sites).
        fitnesses: np.ndarray of shape (pop_size,).
        k: tournament size.

    Returns:
        Selected individual (copy).
    """
    pop_size = population.shape[0]
    idxs = np.random.choice(pop_size, k, replace=False)
    best_idx = idxs[np.argmin(fitnesses[idxs])]  # lower is better
    return population[best_idx].copy()
