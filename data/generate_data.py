import osmnx as ox
import networkx as nx
import numpy as np
import json
import random

def generate_dataset(
    place: str,
    n_sites: int = 10,
    n_demands: int = 50,
    budget: int = 300,
    seed: int = 42,
    output_file: str = "evga_data.json"
):
    """
    Generate dataset for EVGA with real OSM locations and a precomputed distance matrix.
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. Download street network
    print(f"Downloading street network for {place}...")
    G = ox.graph_from_place(place, network_type="drive")
    
    # Visualize the graph if needed.
    ## ox.plot_graph(G)

    # Convert to undirected for simplicity
    G = nx.Graph(G)

    # 2. Sample site locations (candidate stations)
    nodes = list(G.nodes)
    site_nodes = random.sample(nodes, n_sites)

    sites = []
    for i, node in enumerate(site_nodes):
        x, y = G.nodes[node]["x"], G.nodes[node]["y"]
        sites.append({"id": i, "lat": y, "lon": x})

    # 3. Sample demand points
    demand_nodes = random.sample(nodes, n_demands)
    demand_points = []
    for i, node in enumerate(demand_nodes):
        x, y = G.nodes[node]["x"], G.nodes[node]["y"]
        demand = random.randint(10, 50)  # random demand value
        demand_points.append({"id": i, "lat": y, "lon": x, "demand": demand})

    # 4. Compute distance matrix (site ↔ demand shortest path distance in km)
    dist_matrix = np.zeros((n_sites, n_demands))
    for i, site_node in enumerate(site_nodes):
        for j, demand_node in enumerate(demand_nodes):
            try:
                length = nx.shortest_path_length(G, site_node, demand_node, weight="length")
                dist_matrix[i, j] = length / 1000.0  # meters → km
            except nx.NetworkXNoPath:
                dist_matrix[i, j] = float("inf")

    # 5. Add capacity/costs for sites
    site_costs = np.random.randint(20, 50, size=n_sites).tolist()
    capacity_per_outlet = np.random.randint(30, 80, size=n_sites).tolist()
    max_outlets = np.random.randint(2, 5, size=n_sites).tolist()

    # 6. Bundle into dict
    data = {
        "budget": budget,
        "site_costs": site_costs,
        "capacity_per_outlet": capacity_per_outlet,
        "max_outlets": max_outlets,
        "n_sites": n_sites,
        "max_outlets_per_site": max_outlets,  # same for now
        "sites": sites,
        "demand_points": demand_points,
        "distance_matrix": dist_matrix.tolist(),
    }

    # 7. Save to JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Dataset saved to {output_file}")
    return data


if __name__ == "__main__":
    dataset = generate_dataset("Hampstead, Canada", n_sites=10, n_demands=50)