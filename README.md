# EVGA: Electric Vehicle Charging Site Optimization

EVGA is a Python-based framework for optimizing the location and capacity of electric vehicle (EV) charging stations using a **Genetic Algorithm (GA)**. It allows you to generate realistic datasets, evaluate solutions, and run experiments with precomputed distance matrices.

---

## 🚀 Features

- Genetic Algorithm for EV site placement optimization
- Fitness evaluation based on:
  - Total weighted distance from demand points to charging sites
  - Served demand respecting outlet capacity and service radius
  - Budget constraints with penalties
- Supports real-world data from **OpenStreetMap** via `osmnx`
- Precomputed distance matrices for realistic travel distances
- Easy-to-extend modular architecture:
  - `encoding.py` – solution representation and decoding
  - `fitness.py` – evaluation functions
  - `operators.py` – GA operators (selection, crossover, mutation)
  - `ga.py` – main GA class
  - `tests/` – unit tests for all modules

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/ismailsevim/evga.git
cd evga
```

2. Install in editable mode:
   
```
pip install -e .
```
---

##  🧪 Running the GA

```
from evga.ga import GeneticAlgorithm
import json

# Load dataset
with open("data/sample_data_leplateaumontroyal.json") as f:
    data = json.load(f)

# Run GA
ga = GeneticAlgorithm(data, pop_size=50, generations=100)
best_solution, best_fitness = ga.run()

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
```

## 🗃️ Generating Realistic Datasets

Use dataset_generator.py to generate datasets with site and demand points from **OpenStreetMap**. Make sure that `osmnx` is installed:

```
from dataset_generator import generate_dataset

data = generate_dataset(
    place="Hampstead, Canada",
    n_sites=10,
    n_demands=50,
    budget=300
)
```

This produces a JSON with:
- `sites` – candidate charging locations
- `demand_points` – EV demand points
- `distance_matrix` – shortest path distances in km
- `capacity_per_outlet`, `max_outlets`, `budget`

## ✅ Tests

Run all tests to ensure functionality:
```
pytest tests/
```







