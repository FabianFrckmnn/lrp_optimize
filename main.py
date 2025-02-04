import math
import random
import pickle
import os
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms
from scipy.spatial import Voronoi, voronoi_plot_2d

from config.config import DISTANCES_FILE
from data.cities import CITIES


MAX_RADIUS = 100
MAX_ROUND_TRIP = 200
EMPLOYEES_PER_TRUCK = 2
EMPLOYEE_WAGE = 100
TRUCK_COST = 50
TRUCK_CAPACITY = 500
LOGISTICS_CENTER_COST = 500
SPECIAL_CITIES = {"Berlin": 0.8, "Stuttgart": 0.8}
SPECIAL_WAGE = {"Berlin": 1.1, "Stuttgart": 1.1}
CITY_DEMAND = {city: pop / 1000 for city, (_, _, pop) in CITIES.items()}


def save_model(distances):
    with open(DISTANCES_FILE, "wb") as f:
        pickle.dump(distances, f)


def load_model():
    if  os.path.exists(DISTANCES_FILE):
        with open(DISTANCES_FILE, "rb") as f:
            distances = pickle.load(f)
        return distances
    return None


def plot_voronoi(centers, title="Initial Voronoi Diagram"):
    points = np.array([[CITIES[city][0], CITIES[city][1]] for city in centers])
    vor = Voronoi(points)

    _, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=1.5)
    ax.scatter(points[:, 0], points[:, 1], c='red', marker='o', label="Logistics Centers")

    for city in centers:
        plt.text(CITIES[city][0], CITIES[city][1], city, fontsize=8, ha='right')

    plt.title(title)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.legend()
    plt.show()


def plot_network(centers):
    G = nx.Graph()

    for city in centers:
        G.add_node(city, pos=(CITIES[city][1], CITIES[city][0]))  # (longitude, latitude)

    for city1 in centers:
        for city2 in centers:
            if city1 != city2 and shortest_paths[city1][city2] <= MAX_RADIUS:
                G.add_edge(city1, city2, weight=shortest_paths[city1][city2])

    plt.figure(figsize=(10, 6))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=8, edge_color="gray")
    plt.title("Optimized Logistics Network")
    plt.show()


def get_real_distances(cities):
    distances = load_model()
    if distances is not None:
        return distances

    print("Downloading road network...")
    G = ox.graph_from_place("Germany", network_type="drive")

    city_locs = {city: ox.distance.nearest_nodes(G, lon, lat) for city, (lat, lon, _) in cities.items()}
    distances = {}

    for city1, node1 in city_locs.items():
        distances[city1] = {}
        for city2, node2 in city_locs.items():
            if city1 != city2:
                distances[city1][city2] = ox.distance.shortest_path_length(G, node1, node2, weight="length") / 1000

    save_model(distances)

    return distances


def evaluate(individual):
    centers = [list(CITIES.keys())[i] for i in range(len(individual)) if individual[i] == 1]

    # penalty for infeasible solutions
    if not centers:
        return (float('inf'),)

    total_cost = sum(
        LOGISTICS_CENTER_COST * SPECIAL_CITIES.get(center, 1.0) for center in centers
    )

    employees_needed = len(centers) * EMPLOYEES_PER_TRUCK
    employee_costs = sum(
        EMPLOYEE_WAGE * SPECIAL_WAGE.get(center, 1.0) for center in centers
    )

    truck_costs = 0
    total_time = 0
    total_trucks_used = 0

    satisfied_demand = {city: 0 for city in CITIES.keys()}

    for city in CITIES.keys():
        min_distance = float('inf')
        best_center = None

        for center in centers:
            if center in shortest_paths[city] and shortest_paths[city][center] <= MAX_RADIUS:
                if shortest_paths[city][center] < min_distance:
                    min_distance = shortest_paths[city][center]
                    best_center = center

        # penalty for infeasible solutions
        if min_distance == float('inf'):
            return (float('inf'),)

        total_time += min_distance
        trucks_needed = math.ceil(CITY_DEMAND[city] / TRUCK_CAPACITY)
        truck_costs += trucks_needed * TRUCK_COST
        total_trucks_used += trucks_needed

        # penalty for round-trip constraint violation
        if 2 * min_distance > MAX_ROUND_TRIP:
            return (float('inf'),)

        satisfied_demand[city] += CITY_DEMAND[city]

    # penalty for missing employee requirements
    if total_trucks_used > (employees_needed / 2):
        return (float('inf'),)

    # penalty for not fully meeting demand
    unmet_demand = sum(max(0, CITY_DEMAND[city] - satisfied_demand[city]) for city in CITIES.keys())
    if unmet_demand > 0:
        return (float('inf'),)

    total_cost += truck_costs + employee_costs
    weighted_sum = total_cost + total_time

    return (weighted_sum,)


def run_ga():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof,
                                   verbose=True)
    return hof[0]


if __name__ == '__main__':
    print("Precomputing shortest paths...")
    shortest_paths = get_real_distances(CITIES)

    plot_voronoi(list(CITIES.keys()))

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(CITIES))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    best_solution = run_ga()
    selected_centers = [list(CITIES.keys())[i] for i in range(len(best_solution)) if best_solution[i] == 1]
    print("Best Logistics Centers: ", selected_centers)

    plot_voronoi(selected_centers, title="Optimized Logistics Centers (Voronoi Diagram)")

    plot_network(selected_centers)
