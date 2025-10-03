import numpy as np
import json
from evga.fitness import evaluate_solution, compute_total_cost, compute_served_demand


def load_data():
    with open("data/sample_data_leplateaumontroyal.json") as f:
        return json.load(f)


def test_evaluate_solution_all_zero():
    """If all sites are closed, fitness should be +inf (no demand is served)."""
    data = load_data()
    solution = np.zeros(len(data["max_outlets"]), dtype=int)
    fitness = evaluate_solution(solution, data)
    assert isinstance(fitness, float)
    assert fitness == float("inf") or fitness > 0  # should be "bad" fitness


def test_evaluate_solution_feasible():
    """If at least one site is open, fitness should be finite."""
    data = load_data()
    solution = np.zeros(len(data["max_outlets"]), dtype=int)
    solution[0] = 1  # open 1 outlet at the first site
    fitness = evaluate_solution(solution, data)
    assert isinstance(fitness, float)
    assert np.isfinite(fitness)


def test_evaluate_solution_with_penalty():
    """Opening max outlets should trigger budget penalty."""
    data = load_data()
    solution = np.array(data["max_outlets"], dtype=int)  # max at every site
    fitness = evaluate_solution(solution, data, penalty_coeff=1e6)
    # Penalty should dominate -> fitness becomes very large positive
    assert fitness > 0


def test_compute_total_cost_and_served():
    """Check cost and served demand behave consistently."""
    data = load_data()
    solution = np.zeros(len(data["max_outlets"]), dtype=int)
    solution[0] = 1  # one outlet at site 0

    cost = compute_total_cost(solution, data)
    served = compute_served_demand(solution, data)

    assert isinstance(cost, float)
    assert isinstance(served, float)
    assert cost >= 0
    assert served >= 0
