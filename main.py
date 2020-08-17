############################################################################

# Required Libraries
import pandas as pd

from src.pyVRP import  build_coordinates, build_distance_matrix, genetic_algorithm_vrp, plot_tour_coordinates

######################## EXAMPLE - 01 ####################################

# Load Dataset Case 1 - Distance Matrix (YES); Coordinates (NO)
distance_matrix = pd.read_csv('VRP-01-Distance Matrix.txt', sep = '\t') 
distance_matrix = distance_matrix.values
coordinates     = build_coordinates(distance_matrix)
parameters      = pd.read_csv('VRP-01-Parameters.txt', sep = '\t') 
parameters      = parameters.values

# Parameters - Model
n_depots    =  2           # The first n rows of the distance_matrix or coordinates
time_window = 'without'    # 'with', 'without'
route       = 'open'       # 'open', 'closed'
model       = 'vrp'        # 'tsp', 'mtsp', 'vrp'
graph       = True         # True, False

# Parameters - Vehicle
vehicle_types = 2          # Two Types of Vehicles: A and B
fixed_cost    = [12 , 25]  # Fixed Cost for Vehicle A = 12; Fixed Cost for Vehicle B = 25
variable_cost = [0.5,  1]  # Variable Cost for Vehicle A = 0.5; Variable Cost for Vehicle B = 1
capacity      = [9  ,  8]  # Capacity of Vehicle A = 9; Capacity of Vehicle B = 8
velocity      = [50 , 70]  # Average Velocity of Vehicle A = 50; Average Velocity of Vehicle B = 70. The Average Velocity Value is Used as a Constant that Divides the Distance Matrix.

# Parameters - GA
penalty_value   = 10000    # GA Target Function Penalty Value for Violating the Problem Constraints
population_size = 15       # GA Population Size
mutation_rate   = 0.10     # GA Mutation Rate
elite           = 1        # GA Elite Member(s) - Total Number of Best Individual(s) that (is)are Maintained 
generations     = 1000     # GA Number of Generations

# Call GA Function
ga_report, ga_vrp = genetic_algorithm_vrp(coordinates, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, population_size, vehicle_types, n_depots, route, model, time_window, mutation_rate, elite, generations, penalty_value, graph)

# Plot Solution
plot_tour_coordinates(coordinates, ga_vrp, n_depots = n_depots, route = route)

# Solution Report
print(ga_report)

# Save Solution Report
ga_report.to_csv('VRP-01-Report.csv', sep = ';', index = False)

######################## EXAMPLE - 02 ####################################

# Load Dataset Case 2 - Distance Matrix (NO); Coordinates (YES)
coordinates     = pd.read_csv('VRP-02-Coordinates.txt', sep = '\t') 
coordinates     = coordinates.values
distance_matrix = build_distance_matrix(coordinates)
parameters      = pd.read_csv('VRP-02-Parameters.txt', sep = '\t') 
parameters      = parameters.values

# Parameters - Model
n_depots    =  1         # The first n rows of the distance_matrix or coordinates
time_window = 'with'     # 'with', 'without'
route       = 'closed'   # 'open', 'closed'
model       = 'vrp'      # 'tsp', 'mtsp', 'vrp'
graph       = True       # True, False

# Parameters - Vehicle
vehicle_types = 1        # One Type of Vehicle: A
fixed_cost    = [ 30]    # Fixed Cost for Vehicle A = 30
variable_cost = [  2]    # Variable Cost for Vehicle A = 2
capacity      = [150]    # Capacity of Vehicle A = 150
velocity      = [ 70]    # Average Velocity of Vehicle A = 70. The Average Velocity Value is Used as a Constant that Divides the Distance Matrix.

# Parameters - GA
penalty_value   = 10000  # GA Target Function Penalty Value for Violating the Problem Constraints
population_size = 50     # GA Population Size
mutation_rate   = 0.10   # GA Mutation Rate
elite           = 1      # GA Elite Member(s) - Total Number of Best Individual(s) that (is)are Maintained in Each Generation
generations     = 1000   # GA Number of Generations

# Call GA Function
ga_report, ga_vrp = genetic_algorithm_vrp(coordinates, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, population_size, vehicle_types, n_depots, route, model, time_window, mutation_rate, elite, generations, penalty_value, graph)

# Plot Solution
plot_tour_coordinates(coordinates, ga_vrp, n_depots = n_depots, route = route)

# Solution Report
print(ga_report)

# Save Solution Report
ga_report.to_csv('VRP-02-Report.csv', sep = ';', index = False)

############################################################################
