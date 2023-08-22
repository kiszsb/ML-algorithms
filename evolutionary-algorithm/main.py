# Monika Kisz
# travelling salesman problem

import pandas as pd
import plotly.express as px
import numpy as np
import time

np.random.seed(42)
n_cities = 30

# random coordinates
x = list(np.random.randint(0, 100, n_cities))
y = list(np.random.randint(0, 100, n_cities))

city = []
for i in range(n_cities):
    city.append(i)

df = pd.DataFrame(list(zip(x, y, city)), columns=['x', 'y', 'point'])


def euclidean_distance(x1, x2, y1, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# generating matrix of distances between cities
distances = []
for i in range(0, len(df)):
    if i >= len(df):
        break
    distance_b = []
    for idx in range(0, len(df)):
        if idx >= len(df):
            break
        distance = round(euclidean_distance(df['x'].loc[idx], df['x'].loc[i],
                                            df['y'].loc[idx], df['y'].loc[i]), 2)
        distance_b.append(distance)
    distances.append(distance_b)


class Population:
    def __init__(self, full_population, adjacency_mat):
        self.full_population = full_population
        self.parents = []
        self.score = 0  # score of the best chromosome in the population
        self.best = None  # to store best chromosome
        self.adjacency_mat = adjacency_mat  # to calculate the distance between cities

    # Measuring distances between all cities
    def fitness(self, chromosome):
        suma = 0
        for i in range(len(chromosome-1)-1):
            suma += self.adjacency_mat[chromosome[i]][chromosome[i+1]]
        return suma

    # Evaluating distances
    def evaluate(self):
        distances = np.array(
            [self.fitness(chromosome) for chromosome in self.full_population]
        )
        self.score = np.min(distances)
        self.best = self.full_population[distances.tolist().index(self.score)]
        if False in (distances[0] == distances):
            distances = np.max(distances) - distances
        return distances / np.sum(distances)

    # Tournament selection
    def select(self):
        fit = self.evaluate()
        for i in range(len(fit)):
            idx = np.random.randint(0, len(fit))
            idy = np.random.randint(0, len(fit))
            # tournament selection
            if fit[idx] > fit[idy]:
                self.parents.append(self.full_population[idx])
            else:
                self.parents.append((self.full_population[idy]))
        self.parents = np.array(self.parents)

    def swap(self, chromosome):
        a, b = np.random.choice(len(chromosome), 2)
        chromosome[a], chromosome[b] = (
            chromosome[b],
            chromosome[a],
        )
        return chromosome

    def mutate(self, p_mut=0.1):
        next_bag = []
        children = self.parents
        for child in children:
            if np.random.rand() < p_mut:
                next_bag.append(self.swap(child))
            else:
                next_bag.append(child)
        return next_bag

def genetic_algorithm(
    cities,
    adjacency_mat,
    n_population=200,
    n_iter=50,
    p_mut=0.1,
    print_interval=100,
    return_history=False,
    verbose=False,
):
    pop = init_population(cities, adjacency_mat, n_population)
    best = pop.best
    score = float("inf")
    history = []
    for i in range(n_iter):
        pop.select()
        history.append(pop.score)
        if verbose:
            print(f"Generation {i}: {pop.score}")
        elif i % print_interval == 0:
            print(f"Generation {i}: {pop.score}")
        if pop.score < score:
            best = pop.best
            score = pop.score
        children = pop.mutate(p_mut)
        pop = Population(children, pop.adjacency_mat)
    if return_history:
        return best, history
    return best


# Randomly initiating first generation
def init_population(cities, adjacency_mat, n_population):
    population = []
    for _ in range(n_population):
        population.append(np.random.permutation(cities))
    return Population(population, adjacency_mat)


# start to measure time of algorithm
start_time = time.time()
best, history = genetic_algorithm(city, distances, verbose=True, return_history=True)

print("Processing time of : %.2f seconds."% (time.time()-start_time))


def Average(lst):
    return sum(lst) / len(lst)


# Printing statistics
average = Average(history)
print(f'average: {round(average,2)}')
print(f'minimum: {round(min(history),2)}')
print(f'maximum: {round(max(history),2)}')
print(f'standard deviation: {round(np.std(history),2)}')
new_x = []
new_y = []
new_cities = []
for i in best:
    new_x.append(x[i])
    new_y.append(y[i])
    new_cities.append(i)

new_df = pd.DataFrame(list(zip(new_x, new_y, new_cities)), columns=['x', 'y', 'point'])
fig = px.line(new_df, x='x', y='y', text='point', title='City Map')
fig.update_layout(template="simple_white", width=800, title_x=0.5, font=dict(size=20))
fig.update_traces(textposition='top center')
fig.show()

