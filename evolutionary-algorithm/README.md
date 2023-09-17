# Evolutionary algorithm
Welcome to the "Evolutionary Algorithm" repository! This project showcases the implementation of an evolutionary algorithm in Python. Evolutionary algorithms are a 
class of optimization algorithms inspired by the principles of natural selection and evolution. This repository serves as a resource for understanding, using, and 
customizing evolutionary algorithms for various optimization tasks.

## Introduction 
Evolutionary algorithms are a family of stochastic optimization techniques that draw inspiration from the process of biological evolution. They are designed to find 
optimal or near-optimal solutions to complex problems through a process of selection, reproduction, and mutation. These algorithms are particularly well-suited for 
problems with non-linear, discontinuous, or poorly defined objective functions.

In this example classic evolution algorithm was implemented. Presented algorithm in this case uses tournament selection and generational succession, crossover was not implemented.
Algorithm solves traveling salesman problem based on created list of cities and their co-ordinates to find shortest way.

## Description
Evolutionary algorithm operates as follows:
- Initialization: A population of potential solutions (individuals) is created, often randomly or based on prior knowledge.
- Evaluation: Each individual in the population is evaluated according to the objective function of the problem being optimized. This function quantifies how well each individual solves the problem.
- Selection: Individuals are selected from the current population to serve as parents for the next generation. The probability of selection is typically proportional to an individual's fitness, which is a measure of how well they perform.
- Crossover: Pairs of parents are combined to create one or more offspring. Crossover methods vary but typically involve exchanging genetic information between parents.
- Mutation: With a certain probability, the offspring undergo random changes or mutations in their genetic makeup.
- Replacement: The offspring replace some individuals in the current population, often based on their fitness, to form the next generation.
- Termination: The algorithm continues to iterate through generations until a termination condition is met, such as a maximum number of generations reached or a satisfactory solution found.

## Use cases
Evolutionary algorithms are highly versatile and can be applied to a wide range of optimization problems, including:
- Function Optimization: Finding the global minimum or maximum of mathematical functions.
- Parameter Tuning: Optimizing parameters for machine learning algorithms or simulations.
- Combinatorial Optimization: Solving problems like the traveling salesman problem and job scheduling.
- Feature Selection: Selecting the most relevant features for machine learning models.
