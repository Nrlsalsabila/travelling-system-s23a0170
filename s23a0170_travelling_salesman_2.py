import random
from itertools import permutations

# Set non-interactive Matplotlib backend BEFORE importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(
    page_title="TSP Solver using Genetic Algorithm", layout="wide"
)

st.title("Travelling Salesman Problem (TSP) Solver using GA")

# ---------------------------------------------------------
# Sidebar UI Components for Hyperparameters
# ---------------------------------------------------------
st.sidebar.header("Genetic Algorithm Parameters")

n_population = st.sidebar.slider(
    "Population Size", min_value=10, max_value=500, value=250, step=10
)
n_generations = st.sidebar.slider(
    "Generations", min_value=1, max_value=300, value=100, step=5
)
crossover_per = st.sidebar.slider(
    "Crossover Rate", min_value=0.1, max_value=1.0, value=0.8, step=0.05
)
mutation_per = st.sidebar.slider(
    "Mutation Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01
)

# ---------------------------------------------------------
# City Coordinates and Initial Setup
# ---------------------------------------------------------
city_coords = {
    "Gliwice": (0, 1),
    "Cairo": (3, 2),
    "Rome": (6, 1),
    "Krakow": (9, 4.5),
    "Paris": (15, -1),
    "Alexandria": (10, 2.5),
    "Berlin": (16, 11),
    "Tokyo": (5, 4),
    "Rio": (8, 9),
    "Budapest": (1.5, 12),
}

cities_names = list(city_coords.keys())
x = [coords[0] for coords in city_coords.values()]
y = [coords[1] for coords in city_coords.values()]

colors = sns.color_palette("pastel", len(cities_names))

city_icons = {
    "Gliwice": "♕",
    "Cairo": "♖",
    "Rome": "♗",
    "Krakow": "♘",
    "Paris": "♙",
    "Alexandria": "♔",
    "Berlin": "♚",
    "Tokyo": "♛",
    "Rio": "♜",
    "Budapest": "♝",
}

# ---------------------------------------------------------
# Genetic Algorithm Helper Functions
# ---------------------------------------------------------


def initial_population(cities_list, n_pop=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    # Safeguard if requested population exceeds max possible permutations
    sample_size = min(n_pop, len(possible_perms))
    random_ids = random.sample(range(0, len(possible_perms)), sample_size)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms


def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(
        np.sum((np.array(city_1_coords) - np.array(city_2_coords)) ** 2)
    )


def total_dist_individual(individual):
    total_dist = 0
    for i in range(0, len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i + 1])
    return total_dist


def fitness_prob(population):
    total_dist_all_individuals = [
        total_dist_individual(ind) for ind in population
    ]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(
        total_dist_all_individuals
    )
    population_fitness_sum = sum(population_fitness)

    if population_fitness_sum == 0:
        return np.ones(len(population)) / len(population)

    return population_fitness / population_fitness_sum


def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    bool_prob_array = (
        population_fitness_probs_cumsum < np.random.uniform(0, 1, 1)[0]
    )
    selected_individual_index = len(bool_prob_array[bool_prob_array == True])
    if selected_individual_index >= len(population):
        selected_individual_index = len(population) - 1
    return population[selected_individual_index]


def crossover(parent_1, parent_2):
    n_cities_cut = len(cities_names) - 1
    cut = round(random.uniform(1, n_cities_cut))

    offspring_1 = parent_1[0:cut]
    offspring_1 += [city for city in parent_2 if city not in offspring_1]

    offspring_2 = parent_2[0:cut]
    offspring_2 += [city for city in parent_1 if city not in offspring_2]

    return offspring_1, offspring_2


def mutation(offspring):
    n_cities_cut = len(cities_names) - 1
    index_1 = round(random.uniform(0, n_cities_cut))
    index_2 = round(random.uniform(0, n_cities_cut))

    offspring[index_1], offspring[index_2] = (
        offspring[index_2],
        offspring[index_1],
    )
    return offspring


def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)
    fitness_probs = fitness_prob(population)

    parents_list = []
    for _ in range(0, int(crossover_per * n_population)):
        parents_list.append(roulette_wheel(population, fitness_probs))

    offspring_list = []
    for i in range(0, len(parents_list) - 1, 2):
        offspring_1, offspring_2 = crossover(
            parents_list[i], parents_list[i + 1]
        )

        if random.random() > (1 - mutation_per):
            offspring_1 = mutation(offspring_1)
        if random.random() > (1 - mutation_per):
            offspring_2 = mutation(offspring_2)

        offspring_list.append(offspring_1)
        offspring_list.append(offspring_2)

    mixed_offspring = parents_list + offspring_list
    fitness_probs = fitness_prob(mixed_offspring)
    sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
    best_fitness_indices = sorted_fitness_indices[0:n_population]

    best_mixed_offspring = [
        mixed_offspring[idx] for idx in best_fitness_indices
    ]

    for _ in range(0, n_generations):
        fitness_probs = fitness_prob(best_mixed_offspring)
        parents_list = []
        for _ in range(0, int(crossover_per * n_population)):
            parents_list.append(
                roulette_wheel(best_mixed_offspring, fitness_probs)
            )

        offspring_list = []
        for i in range(0, len(parents_list) - 1, 2):
            offspring_1, offspring_2 = crossover(
                parents_list[i], parents_list[i + 1]
            )

            if random.random() > (1 - mutation_per):
                offspring_1 = mutation(offspring_1)
            if random.random() > (1 - mutation_per):
                offspring_2 = mutation(offspring_2)

            offspring_list.append(offspring_1)
            offspring_list.append(offspring_2)

        mixed_offspring = parents_list + offspring_list
        fitness_probs = fitness_prob(mixed_offspring)
        sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
        best_fitness_indices = sorted_fitness_indices[
            0 : int(0.8 * n_population)
        ]

        best_mixed_offspring = [
            mixed_offspring[idx] for idx in best_fitness_indices
        ]

        old_population_indices = [
            random.randint(0, (len(population) - 1))
            for _ in range(int(0.2 * n_population))
        ]
        for idx in old_population_indices:
            best_mixed_offspring.append(population[idx])

        random.shuffle(best_mixed_offspring)

    return best_mixed_offspring


# ---------------------------------------------------------
# Execution & Visualization
# ---------------------------------------------------------
if st.sidebar.button("Run Genetic Algorithm"):
    with st.spinner("Calculating optimal route..."):
        best_mixed_offspring = run_ga(
            cities_names,
            n_population,
            n_generations,
            crossover_per,
            mutation_per,
        )

        total_dist_all_individuals = [
            total_dist_individual(ind) for ind in best_mixed_offspring
        ]
        index_minimum = np.argmin(total_dist_all_individuals)
        minimum_distance = total_dist_all_individuals[index_minimum]
        shortest_path = best_mixed_offspring[index_minimum]

        x_shortest = [city_coords[city][0] for city in shortest_path]
        y_shortest = [city_coords[city][1] for city in shortest_path]
        x_shortest.append(x_shortest[0])
        y_shortest.append(y_shortest[0])

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot city locations & icons
        for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
            color = colors[i]
            icon = city_icons[city]
            ax.scatter(city_x, city_y, c=[color], s=1000, zorder=2)
            ax.annotate(
                icon,
                (city_x, city_y),
                fontsize=30,
                ha="center",
                va="center",
                zorder=3,
            )
            ax.annotate(
                city,
                (city_x, city_y),
                fontsize=11,
                ha="center",
                va="bottom",
                xytext=(0, -25),
                textcoords="offset points",
            )

        # Draw best route
        ax.plot(
            x_shortest, y_shortest, "--go", label="Best Route", linewidth=2.5
        )

        # Draw background grid lines connecting all cities
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                ax.plot(
                    [x[i], x[j]], [y[i], y[j]], "k-", alpha=0.09, linewidth=1
                )

        ax.set_title(
            "TSP Best Route Using Genetic Algorithm", fontsize=18, pad=15
        )
        ax.legend()

        # Display results in Streamlit
        st.pyplot(fig)

        st.success(f"**Shortest Total Distance:** {round(minimum_distance, 3)}")
        st.write(
            "**Optimal City Order:** "
            + " ➔ ".join([f"{i+1}. {c}" for i, c in enumerate(shortest_path)])
        )
