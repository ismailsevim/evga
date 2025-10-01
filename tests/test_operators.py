import numpy as np
import json
from evga.operators import (
    initialize_population,
    crossover,
    mutation,
    tournament_selection,
)


def load_data():
    with open("data/sample_data.json") as f:
        return json.load(f)


def test_initialize_population_shape():
    data = load_data()
    pop = initialize_population(10, data)
    assert pop.shape == (10, data["n_sites"])


def test_crossover_produces_children():
    p1 = np.array([0, 1, 2])
    p2 = np.array([2, 1, 0])
    children = crossover(p1, p2)
    assert len(children) == 2
    assert all(isinstance(c, np.ndarray) for c in children)


def test_mutation_respects_limits():
    data = load_data()
    indiv = np.zeros(data["n_sites"], dtype=int)
    mutant = mutation(indiv, data, mutation_rate=1.0)  # force mutation
    assert np.all(mutant >= 0)
    assert np.all(mutant <= np.array(data["max_outlets_per_site"]))


def test_tournament_selection_returns_valid():
    data = load_data()
    pop = initialize_population(5, data)
    fits = np.arange(5, dtype=float)  # 0 is best, 4 is worst
    selected = tournament_selection(pop, fits, k=3)
    assert isinstance(selected, np.ndarray)
    assert selected.shape == (data["n_sites"],)

