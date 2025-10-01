import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

import json
import numpy as np
from evga.ga import GeneticAlgorithm


def main():
    # Load problem data
    with open("data/sample_data_leplateaumontroyal.json") as f:
        data = json.load(f)

    # Set GA hyperparameters
    ga = GeneticAlgorithm(
        data=data,
        pop_size=20,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        tournament_k=3,
        random_seed=420,
    )

    # Run GA
    best_solution, best_fitness = ga.run()

    print("\n=== EVGA Result ===")
    print("Best solution (outlets per site):", best_solution)
    print("Fitness value:", best_fitness)

    # Decode for readability
    from evga.encoding import decode_solution
    print("\nDecoded solution:", decode_solution(best_solution, data))


if __name__ == "__main__":
    main()

