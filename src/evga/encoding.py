import numpy as np
from typing import List


def random_solution(max_outlets: List[int], rng: np.random.Generator) -> np.ndarray:
    """
    Generate a random solution where each gene is in [0, max_outlets[i]].

    Args:
        max_outlets: list of maximum outlets allowed at each site.
        rng: numpy random generator.

    Returns:
        np.ndarray of shape (n_sites,) with integer values.
    """
    max_arr = np.array(max_outlets, dtype=int)
    solution = rng.integers(0, max_arr + 1)  # upper bound exclusive, so +1
    return solution


def random_population(
    pop_size: int, max_outlets: List[int], rng: np.random.Generator
) -> np.ndarray:
    """
    Generate a population of random solutions.

    Args:
        pop_size: number of individuals in the population.
        max_outlets: list of maximum outlets per site.
        rng: numpy random generator.

    Returns:
        np.ndarray of shape (pop_size, n_sites).
    """
    n_sites = len(max_outlets)
    pop = np.zeros((pop_size, n_sites), dtype=int)
    for i in range(pop_size):
        pop[i] = random_solution(max_outlets, rng)
    return pop


def is_feasible(solution: np.ndarray, data: dict) -> bool:
    """
    Check feasibility of a solution based on max_outlets.

    Args:
        solution: array of outlets per site.
        data: problem data dictionary (expects 'max_outlets').

    Returns:
        True if all genes are within [0, max_outlets[i]], else False.
    """
    max_outlets = np.array(data["max_outlets"], dtype=int)
    return np.all((solution >= 0) & (solution <= max_outlets))

def decode_solution(solution: np.ndarray, data: dict) -> dict:
    """
    Convert solution array into human-readable dictionary.

    Args:
        solution: np.ndarray with number of outlets per site
        data: problem data dict (expects 'sites')

    Returns:
        dict with detailed info
    """
    sites_info = []
    for i, outlets in enumerate(solution):
        site = data["sites"][i]
        sites_info.append({
            "id": site["id"],
            "lat": site["lat"],
            "lon": site["lon"],
            "outlets": int(outlets)
        })

    decoded = {
        "sites": sites_info,
        "total_outlets": int(solution.sum())
    }
    return decoded

