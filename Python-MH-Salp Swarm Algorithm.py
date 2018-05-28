############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Salp Swarm Algorithm

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Salp_Swarm_Algorithm, File: Python-MH-Salp Swarm Algorithm.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Salp_Swarm_Algorithm>

############################################################################

# Required Libraries
import pandas as pd
import numpy  as np
import math
import random
import os

# Function: Initialize Variables
def initial_position(swarm_size = 5, min_values = [-5,-5], max_values = [5,5]):
    position = pd.DataFrame(np.zeros((swarm_size, len(min_values))))
    position['Fitness'] = 0.0
    for i in range(0, swarm_size):
        for j in range(0, len(min_values)):
             position.iloc[i,j] = random.uniform(min_values[j], max_values[j])
        position.iloc[i,-1] = target_function(position.iloc[i,0:position.shape[1]-1])
    return position

# Function: Initialize Food Position
def food_position(dimension = 2):
    food = pd.DataFrame(np.zeros((1, dimension)))
    food['Fitness'] = 0.0
    for j in range(0, dimension):
        food.iloc[0,j] = 0.0
    food.iloc[0,-1] = target_function(food.iloc[0,0:food.shape[1]-1])
    return food

# Function: Updtade Food Position by Fitness
def update_food(position, food):
    updated_position = position.copy(deep = True)
    for i in range(0, position.shape[0]):
        if (updated_position.iloc[i,-1] < food.iloc[0,-1]):
            for j in range(0, updated_position.shape[1]):
                food.iloc[0,j] = updated_position.iloc[i,j]
    return food

# Function: Updtade Position
def update_position(position, food, c1 = 1, min_values = [-5,-5], max_values = [5,5]):
    updated_position = position.copy(deep = True)
    
    for i in range(0, updated_position.shape[0]):
        if (i <= updated_position.shape[0]/2):
            for j in range (0, len(min_values)):
                c2 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                c3 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                if (c3 >= 0.5):
                    updated_position.iloc[i,j] = food.iloc[0,j] + c1*((max_values[j] - min_values[j])*c2 + min_values[j])
                    
                    if (updated_position.iloc[i,j] > max_values[j]):
                        updated_position.iloc[i,j] = max_values[j]
                    elif (updated_position.iloc[i,j] < min_values[j]):
                        updated_position.iloc[i,j] = min_values[j] 
                else:
                    updated_position.iloc[i,j] = food.iloc[0,j] - c1*((max_values[j] - min_values[j])*c2 + min_values[j])
                    
                    if (updated_position.iloc[i,j] > max_values[j]):
                        updated_position.iloc[i,j] = max_values[j]
                    elif (updated_position.iloc[i,j] < min_values[j]):
                        updated_position.iloc[i,j] = min_values[j]
                        
        elif (i > updated_position.shape[0]/2 and i < updated_position.shape[0] + 1):
            for j in range (0, len(min_values)):
                updated_position.iloc[i,j] = (updated_position.iloc[i - 1,j] + updated_position.iloc[i,j])/2    
                if (updated_position.iloc[i,j] > max_values[j]):
                    updated_position.iloc[i,j] = max_values[j]
                elif (updated_position.iloc[i,j] < min_values[j]):
                    updated_position.iloc[i,j] = min_values[j]        
        
        updated_position.iloc[i,-1] = target_function(updated_position.iloc[i,0:updated_position.shape[1]-1])
            
    return updated_position

# SSA Function
def salp_swarm_algorithm(swarm_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50):    
    count = 0
    position = initial_position(swarm_size = swarm_size, min_values = min_values, max_values = max_values)
    food = food_position(dimension = len(min_values))

    while (count <= iterations):
        
        print("Iteration = ", count, " f(x) = ", food.iloc[food['Fitness'].idxmin(),-1])
        
        c1 = 2*math.exp(-(4*(count/iterations))**2)

        food = update_food(position, food)        
        position = update_position(position, food, c1 = c1, min_values = min_values, max_values = max_values)
        
        count = count + 1 
        
    print(food.iloc[food['Fitness'].idxmin(),:].copy(deep = True))    
    return food.iloc[food['Fitness'].idxmin(),:].copy(deep = True)

######################## Part 1 - Usage ####################################

# Function to be Minimized. Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def target_function (variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

ssa = salp_swarm_algorithm(swarm_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 100)
