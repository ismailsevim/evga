import numpy as np
import json
from evga.operators import (
    initialize_population,
    crossover,
    mutation,
    tournament_selection,
)


def load_data():
    with open("data/sample_data_leplateaumontroyal.json") as f:
        return json.load(f)


def test_initialize_population_shape_and_limits():
    """Population should have correct shape and respect outlet limits."""
    data = load_data()
    pop = initialize_population(10, data)
    assert pop.shape == (10, data["n_sites"])
    max_outlets = np.array(data["max_outlets"])
    assert np.all(pop >= 0)
    assert np.all(pop <= max_outlets)


def test_crossover_produces_children_and_shape():
    """Crossover should produce exactly two children of correct length."""
    p1 = np.array([0, 1, 2, 3])
    p2 = np.array([3, 2, 1, 0])
    children = crossover(p1, p2)
    assert len(children) == 2
    assert all(isinstance(c, np.ndarray) for c in children)
    assert all(c.shape == p1.shape for c in children)


def test_mutation_respects_limits():
    """Mutation must keep outlet counts within [0, max_outlets]."""
    data = load_data()
    indiv = np.zeros(data["n_sites"], dtype=int)
    mutant = mutation(indiv, data, mutation_rate=1.0)  # force mutation
    max_outlets = np.array(data["max_outlets"])
    assert np.all(mutant >= 0)
    assert np.all(mutant <= max_outlets)


def test_tournament_selection_returns_valid():
    """Tournament selection should return a valid individual from the population."""
    data = load_data()
    pop = initialize_population(5, data)
    fits = np.arange(5, dtype=float)  # lower is better
    selected = tournament_selection(pop, fits, k=3)
    assert isinstance(selected, np.ndarray)
    assert selected.shape == (data["n_sites"],)
    # The selected individual must exist in the population
    assert any(np.array_equal(selected, ind) for ind in pop)
