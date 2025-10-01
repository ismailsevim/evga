import numpy as np
from typing import Dict


def compute_total_cost(solution: np.ndarray, data: dict) -> float:
    """
    Fitness = minimize total weighted distance of demand points 
              to the nearest charging site with an outlet.
    Distances come directly from precomputed distance_matrix (km).
    """
    dist_matrix = np.array(data["distance_matrix"])  # shape (n_sites, n_demands)
    demand_points = data["demand_points"]

    if solution.sum() == 0:
        return float("inf")

    total_cost = 0.0
    for j, dp in enumerate(demand_points):
        dp_demand = dp["demand"]

        # Consider only sites that have outlets
        open_sites = np.where(solution > 0)[0]
        if len(open_sites) == 0:
            return float("inf")

        # Distances from this demand point to all open sites
        dists = dist_matrix[open_sites, j]
        min_dist = np.min(dists)

        total_cost += dp_demand * min_dist

    return total_cost


def compute_served_demand(solution: np.ndarray, data: Dict) -> float:
    """
    Assign demand points to nearest open site within service radius,
    respecting outlet capacity. Uses distance_matrix.
    """
    dist_matrix = np.array(data["distance_matrix"])  # (n_sites, n_demands)
    demand_points = data["demand_points"]
    capacity_per_outlet = np.array(data["capacity_per_outlet"], dtype=float)
    service_radius = data.get("service_radius", float("inf"))  # default: no limit

    # Capacity available per site
    capacity = solution * capacity_per_outlet
    served = 0.0

    for j, dp in enumerate(demand_points):
        dp_demand = dp["demand"]

        nearest_idx = None
        nearest_dist = float("inf")

        # Check all open sites
        for i in range(len(capacity)):
            if solution[i] == 0 or capacity[i] <= 0:
                continue
            dist = dist_matrix[i, j]
            if dist < service_radius and dist < nearest_dist:
                nearest_idx = i
                nearest_dist = dist

        if nearest_idx is not None:
            allocated = min(dp_demand, capacity[nearest_idx])
            served += allocated
            capacity[nearest_idx] -= allocated

    return served


def evaluate_solution(
    solution: np.ndarray, data: Dict, penalty_coeff: float = 10
) -> float:
    """
    Evaluate fitness of a solution (lower is better).
    Fitness = -served_demand + penalty for budget violation.
    """
    cost = compute_total_cost(solution, data)  # km·demand
    served = compute_served_demand(solution, data)

    penalty = 0.0
    if cost > data["budget"]:
        penalty += penalty_coeff * (cost - data["budget"])

    return -served + penalty
