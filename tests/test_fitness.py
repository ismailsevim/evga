import numpy as np
import json
from evga.fitness import evaluate_solution


def load_data():
    with open("data/sample_data.json") as f:
        return json.load(f)


def test_evaluate_solution_feasible():
    data = load_data()
    # A feasible solution (all zeros, cost = 0, within budget)
    solution = np.zeros(data["n_sites"], dtype=int)
    fitness = evaluate_solution(solution, data)
    assert isinstance(fitness, float)


def test_evaluate_solution_infeasible():
    data = load_data()
    # Too many outlets, should exceed budget and add penalty
    solution = np.array(data["max_outlets_per_site"])
    fitness = evaluate_solution(solution, data, penalty_coeff=1e6)
    assert fitness > 0  # penalty makes it worse
