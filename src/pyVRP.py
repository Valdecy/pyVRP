############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Genetic Algorithm

# Citation:
# PEREIRA, V. (2018). Project: Metaheuristic-Genetic_Algorithm, File: Python-MH-Genetic Algorithm.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Genetic_Algorithm>

############################################################################

# Required Libraries
import folium
import folium.plugins
import pandas as pd
import random
import numpy as np
import copy
import os
import time as tm

from itertools import cycle
from matplotlib import pyplot as plt
plt.style.use('bmh')

############################################################################

# Function: Build Coordinates
def build_coordinates(distance_matrix):  
    a           = distance_matrix[0,:].reshape(distance_matrix.shape[0], 1)
    b           = distance_matrix[:,0].reshape(1, distance_matrix.shape[0])
    m           = (1/2)*(a**2 + b**2 - distance_matrix**2)
    w, u        = np.linalg.eig(np.matmul(m.T, m))
    s           = (np.diag(np.sort(w)[::-1]))**(1/2) 
    coordinates = np.matmul(u, s**(1/2))
    coordinates = coordinates.real[:,0:2]
    return coordinates

# Function: Build Distance Matrix
def build_distance_matrix(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Tour Plot
def plot_tour_coordinates (coordinates, solution, n_depots, route, size_x = 10, size_y = 10):
    depot     = solution[0]
    city_tour = solution[1]
    cycol     = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#bf77f6', '#ff9408', 
                       '#d1ffbd', '#c85a53', '#3a18b1', '#ff796c', '#04d8b2', '#ffb07c', '#aaa662', '#0485d1', '#fffe7a', '#b0dd16', '#d85679', '#12e193', 
                       '#82cafc', '#ac9362', '#f8481c', '#c292a1', '#c0fa8b', '#ca7b80', '#f4d054', '#fbdd7e', '#ffff7e', '#cd7584', '#f9bc08', '#c7c10c'])
    plt.figure(figsize = [size_x, size_y])
    for j in range(0, len(city_tour)):
        if (route == 'closed'):
            xy = np.zeros((len(city_tour[j]) + 2, 2))
        else:
            xy = np.zeros((len(city_tour[j]) + 1, 2))
        for i in range(0, xy.shape[0]):
            if (i == 0):
                xy[ i, 0] = coordinates[depot[j][i], 0]
                xy[ i, 1] = coordinates[depot[j][i], 1]
                if (route == 'closed'):
                    xy[-1, 0] = coordinates[depot[j][i], 0]
                    xy[-1, 1] = coordinates[depot[j][i], 1]
            if (i > 0 and i < len(city_tour[j])+1):
                xy[i, 0] = coordinates[city_tour[j][i-1], 0]
                xy[i, 1] = coordinates[city_tour[j][i-1], 1]
        plt.plot(xy[:,0], xy[:,1], marker = 's', alpha = 0.5, markersize = 5, color = next(cycol))
    for i in range(0, coordinates.shape[0]):
        if (i < n_depots):
            plt.plot(coordinates[i,0], coordinates[i,1], marker = 's', alpha = 1.0, markersize = 7, color = 'k')[0]
            plt.text(coordinates[i,0], coordinates[i,1] + 0.04, i, ha = 'center', va = 'bottom', color = 'k', fontsize = 7)
        else:
            plt.text(coordinates[i,0],  coordinates[i,1] + 0.04, i, ha = 'center', va = 'bottom', color = 'k', fontsize = 7)
    return

# Function: Tour Plot - Lat Long
def plot_tour_latlong (lat_long, solution, n_depots, route):
    m       = folium.Map(location = (lat_long.iloc[0][0], lat_long.iloc[0][1]), zoom_start = 14)
    clients = folium.plugins.MarkerCluster(name = 'Clients').add_to(m)
    depots  = folium.plugins.MarkerCluster(name = 'Depots').add_to(m)
    for i in range(0, lat_long.shape[0]):
        if (i < n_depots):
            folium.Marker(location = [lat_long.iloc[i][0], lat_long.iloc[i][1]], popup = '<b>Client: </b>%s</br> <b>Adress: </b>%s</br>'%(int(i), 'D'), icon = folium.Icon(color = 'black', icon = 'home')).add_to(depots)
        else:
            folium.Marker(location = [lat_long.iloc[i][0], lat_long.iloc[i][1]], popup = '<b>Client: </b>%s</br> <b>Adress: </b>%s</br>'%(int(i), 'C'), icon = folium.Icon(color = 'blue')).add_to(clients)
    depot     = solution[0]
    city_tour = solution[1]
    cycol     = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#bf77f6', '#ff9408', 
                       '#d1ffbd', '#c85a53', '#3a18b1', '#ff796c', '#04d8b2', '#ffb07c', '#aaa662', '#0485d1', '#fffe7a', '#b0dd16', '#d85679', '#12e193', 
                       '#82cafc', '#ac9362', '#f8481c', '#c292a1', '#c0fa8b', '#ca7b80', '#f4d054', '#fbdd7e', '#ffff7e', '#cd7584', '#f9bc08', '#c7c10c'])
    for j in range(0, len(city_tour)):
        if (route == 'closed'):
            ltlng = np.zeros((len(city_tour[j]) + 2, 2))
        else:
            ltlng = np.zeros((len(city_tour[j]) + 1, 2))
        for i in range(0, ltlng.shape[0]):
            if (i == 0):
                ltlng[ i, 0] = lat_long.iloc[depot[j][i], 0]
                ltlng[ i, 1] = lat_long.iloc[depot[j][i], 1]
                if (route == 'closed'):
                    ltlng[-1, 0] = lat_long.iloc[depot[j][i], 0]
                    ltlng[-1, 1] = lat_long.iloc[depot[j][i], 1]
            if (i > 0 and i < len(city_tour[j])+1):
                ltlng[i, 0] = lat_long.iloc[city_tour[j][i-1], 0]
                ltlng[i, 1] = lat_long.iloc[city_tour[j][i-1], 1]
        c = next(cycol)
        for i in range(0, ltlng.shape[0]-1):
          locations = [ (ltlng[i,0], ltlng[i,1]), (ltlng[i+1,0], ltlng[i+1,1])]
          folium.PolyLine(locations , color = c, weight = 1.5, opacity = 1).add_to(m)
    return m

# Function: Subroute Distance
def evaluate_distance(distance_matrix, depot, subroute):    
    subroute_i    = depot + subroute
    subroute_j    = subroute + depot
    subroute_ij   = [(subroute_i[i], subroute_j[i]) for i in range(0, len(subroute_i))]
    distance      = list(np.cumsum(distance_matrix[tuple(np.array(subroute_ij).T)]))
    distance[0:0] = [0.0]
    return distance

# Function: Subroute Time
def evaluate_time(distance_matrix, parameters, depot, subroute, velocity):  
    tw_early   = parameters[:, 1]
    tw_st      = parameters[:, 3]
    subroute_i = depot + subroute
    subroute_j = subroute + depot
    wait       = [0]*len(subroute_j)
    time       = [0]*len(subroute_j)
    for i in range(0, len(time)):
        time[i] = time[i] + distance_matrix[(subroute_i[i], subroute_j[i])]/velocity[0]
        if (time[i] < tw_early[subroute_j][i]):
            wait[i] = tw_early[subroute_j][i] - time[i]
            time[i] = tw_early[subroute_j][i]
        time[i] = time[i] + tw_st[subroute_j][i]
        if (i + 1 <= len(time) - 1):
            time[i+1] = time[i]
    time[0:0] = [0]
    wait[0:0] = [0]
    return wait, time

# Function: Subroute Capacity
def evaluate_capacity(parameters, depot, subroute): 
    demand    = parameters[:, 0]
    subroute_ = depot + subroute + depot
    capacity  = list(np.cumsum(demand[subroute_]))
    return capacity 

# Function: Subroute Cost
def evaluate_cost(dist, wait, parameters, depot, subroute, fixed_cost, variable_cost, time_window):
    tw_wc     = parameters[:, 4]
    subroute_ = depot + subroute + depot
    cost      = [0]*len(subroute_)
    if (time_window == 'with'):
        cost = [fixed_cost[0] + y*z if x == 0 else fixed_cost[0] + x*variable_cost[0] + y*z for x, y, z in zip(dist, wait, tw_wc[subroute_])]
    else:
        cost = [fixed_cost[0]  if x == 0 else fixed_cost[0] + x*variable_cost[0]  for x in dist]
    return cost

# Function: Subroute Cost
def evaluate_cost_penalty(dist, time, wait, cap, capacity, parameters, depot, subroute, fixed_cost, variable_cost, penalty_value, time_window, route):
    tw_late = parameters[:, 2]
    tw_st   = parameters[:, 3]
    tw_wc   = parameters[:, 4]
    if (route == 'open'):
        subroute_ = depot + subroute
    else:
        subroute_ = depot + subroute + depot
    pnlt = 0
    cost = [0]*len(subroute_)
    pnlt = pnlt + sum( x > capacity for x in cap[0:len(subroute_)] )
    if(time_window == 'with'):
        pnlt = pnlt + sum(x > y + z for x, y, z in zip(time, tw_late[subroute_] , tw_st[subroute_]))  
        cost = [fixed_cost[0] + y*z if x == 0 else cost[0] + x*variable_cost[0] + y*z for x, y, z in zip(dist, wait, tw_wc[subroute_])]
    else:
        cost = [fixed_cost[0] if x == 0 else cost[0] + x*variable_cost[0] for x in dist]        
    cost[-1] = cost[-1] + pnlt*penalty_value
    return cost[-1]

# Function: Routes Nearest Depot
def evaluate_depot(n_depots, individual, distance_matrix):
    d_1 = float('+inf')
    for i in range(0, n_depots):
        for j in range(0, len(individual[1])):
            d_2 = evaluate_distance(distance_matrix, [i], individual[1][j])[-1]
            if (d_2 < d_1):
                d_1 = d_2
                individual[0][j] = [i]
    return individual

# Function: Routes Best Vehicle
def evaluate_vehicle(vehicle_types, individual, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, penalty_value, time_window, route, fleet_size):
    cost, _     = target_function([individual], distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, penalty_value, time_window, route, fleet_size) 
    individual_ = copy.deepcopy(individual)
    for i in range(0, len(individual[0])):
        for j in range(0, vehicle_types):
            individual_[2][i] = [j]
            cost_, _          = target_function([individual_], distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, penalty_value, time_window, route, fleet_size) 
            if (cost_ < cost):
                cost             = cost_
                individual[2][i] = [j]     
                individual_      = copy.deepcopy(individual)
            else:
                individual_      = copy.deepcopy(individual)
    return individual

# Function: Routes Break Capacity
def cap_break(vehicle_types, individual, parameters, capacity):
    go_on = True
    while (go_on):
        individual_ = copy.deepcopy(individual)
        solution    = [[], [], []]
        for i in range(0, len(individual_[0])):
            cap   = evaluate_capacity(parameters, individual_[0][i], individual_[1][i]) 
            sep   = [x >  capacity[individual_[2][i][0]] for x in cap[1:-1] ]
            sep_f = [individual_[1][i][x] for x in range(0, len(individual_[1][i])) if sep[x] == False]
            sep_t = [individual_[1][i][x] for x in range(0, len(individual_[1][i])) if sep[x] == True ]
            if (len(sep_t) > 0 and len(sep_f) > 0):
                solution[0].append(individual_[0][i])
                solution[0].append(individual_[0][i])
                solution[1].append(sep_f)
                solution[1].append(sep_t)
                solution[2].append(individual_[2][i])
                solution[2].append(individual_[2][i])
            if (len(sep_t) > 0 and len(sep_f) == 0):
                solution[0].append(individual_[0][i])
                solution[1].append(sep_t)
                solution[2].append(individual_[2][i])
            if (len(sep_t) == 0 and len(sep_f) > 0):
                solution[0].append(individual_[0][i])
                solution[1].append(sep_f)
                solution[2].append(individual_[2][i])
        individual_ = copy.deepcopy(solution)
        if (individual == individual_):
            go_on      = False
        else:
            go_on      = True
            individual = copy.deepcopy(solution)
    return individual

# Function: Solution Report
def show_report(solution, distance_matrix, parameters, velocity, fixed_cost, variable_cost, route, time_window):
    column_names = ['Route', 'Vehicle', 'Activity', 'Job', 'Arrive_Load', 'Leave_Load', 'Wait_Time', 'Arrive_Time','Leave_Time', 'Distance', 'Costs']
    tt           = 0
    td           = 0 
    tc           = 0
    tw_st        = parameters[:, 3]
    report_lst   = []
    for i in range(0, len(solution[1])):
        dist         = evaluate_distance(distance_matrix, solution[0][i], solution[1][i])
        wait, time   = evaluate_time(distance_matrix, parameters, solution[0][i], solution[1][i], velocity = [velocity[solution[2][i][0]]])
        reversed_sol = copy.deepcopy(solution[1][i])
        reversed_sol.reverse()
        cap          = evaluate_capacity(parameters, solution[0][i], reversed_sol) 
        cap.reverse()
        leave_cap = copy.deepcopy(cap)
        for n in range(1, len(leave_cap)-1):
            leave_cap[n] = cap[n+1] 
        cost = evaluate_cost(dist, wait, parameters, solution[0][i], solution[1][i], fixed_cost = [fixed_cost[solution[2][i][0]]], variable_cost = [variable_cost[solution[2][i][0]]], time_window = time_window)
        if (route == 'closed'):
            subroute = [solution[0][i] + solution[1][i] + solution[0][i] ]
        elif (route == 'open'):
            subroute = [solution[0][i] + solution[1][i] ]
        for j in range(0, len(subroute[0])):
            if (j == 0):
                activity    = 'start'
                arrive_time = round(time[j],2)
            else:
                arrive_time = round(time[j] - tw_st[subroute[0][j]] - wait[j],2)
            if (j > 0 and j < len(subroute[0]) - 1):
                activity = 'service'  
            if (j == len(subroute[0]) - 1):
                activity = 'finish'
                if (time[j] > tt):
                    tt = time[j]
                td = td + dist[j]
                tc = tc + cost[j]
            report_lst.append(['#' + str(i+1), solution[2][i][0], activity, subroute[0][j], cap[j], leave_cap[j], round(wait[j],2), arrive_time, round(time[j],2), round(dist[j],2), round(cost[j],2) ])
        report_lst.append(['-//-', '-//-', '-//-', '-//-','-//-', '-//-', '-//-', '-//-', '-//-', '-//-', '-//-'])
    report_lst.append(['MAX TIME', '', '','', '', '', '', '', round(tt,2), '', ''])
    report_lst.append(['TOTAL', '', '','', '', '', '', '', '', round(td,2), round(tc,2)])
    report_df = pd.DataFrame(report_lst, columns = column_names)
    return report_df

# Function: Route Evalution & Correction
def target_function(population, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, penalty_value, time_window, route, fleet_size = []):
    cost     = [[0] for i in range(len(population))]
    tw_late  = parameters[:, 2]
    tw_st    = parameters[:, 3]
    flt_cnt  = [0]*len(fleet_size)
    if (route == 'open'):
        end = 2 
    else:
        end = 1
    for k in range(0, len(population)): # k individuals
        individual = copy.deepcopy(population[k])  
        size       = len(individual[1])
        i          = 0
        pnlt       = 0
        flt_cnt    = [0]*len(fleet_size)
        while (size > i): # i subroutes 
            dist = evaluate_distance(distance_matrix, individual[0][i], individual[1][i])
            if(time_window == 'with'):
                wait, time = evaluate_time(distance_matrix, parameters, depot = individual[0][i], subroute = individual[1][i], velocity = [velocity[individual[2][i][0]]])
            else:
                wait       = []
                time       = []
            cap    = evaluate_capacity(parameters, depot = individual[0][i], subroute = individual[1][i])
            cost_s = evaluate_cost(dist, wait, parameters, depot = individual[0][i], subroute = individual[1][i], fixed_cost = [fixed_cost[individual[2][i][0]]], variable_cost = [variable_cost[individual[2][i][0]]], time_window = time_window)      
            pnlt   = pnlt + sum( x >  capacity[individual[2][i][0]] for x in cap[0:-1] )
            if(time_window == 'with'):
                if (route == 'open'):
                    subroute_ = individual[0][i] + individual[1][i]
                else:
                    subroute_ = individual[0][i] + individual[1][i] + individual[0][i]
                pnlt = pnlt + sum(x > y + z for x, y, z in zip(time, tw_late[subroute_] , tw_st[subroute_]))                      
            if (len(fleet_size) > 0):
                flt_cnt[individual[2][i][0]] = flt_cnt[individual[2][i][0]] + 1 
            if (size <= i + 1):
                for v in range(0, len(fleet_size)):
                    v_sum = flt_cnt[v] - fleet_size[v]
                    if (v_sum > 0):
                        pnlt = pnlt + v_sum
            cost[k][0] = cost[k][0] + cost_s[-end] + pnlt*penalty_value
            size       = len(individual[1])
            i          = i + 1
    cost_total = copy.deepcopy(cost)
    return cost_total, population

# Function: Initial Population
def initial_population(coordinates = 'none', distance_matrix = 'none', population_size = 5, vehicle_types = 1, n_depots = 1, model = 'vrp'):
    try:
        distance_matrix.shape[0]
    except:
        distance_matrix = build_distance_matrix(coordinates)
    if (model == 'tsp'):
        n_depots = 1
    depots     = [[i] for i in range(0, n_depots)]
    vehicles   = [[i] for i in range(0, vehicle_types)]
    clients    = list(range(n_depots, distance_matrix.shape[0]))
    population = []
    for i in range(0, population_size):
        clients_temp    = copy.deepcopy(clients)
        routes          = []
        routes_depot    = []
        routes_vehicles = []
        while (len(clients_temp) > 0):
            e = random.sample(vehicles, 1)[0]
            d = random.sample(depots, 1)[0]
            if (model == 'tsp'):
                c = random.sample(clients_temp, len(clients_temp))
            else:
                c = random.sample(clients_temp, random.randint(1, len(clients_temp)))
            routes_vehicles.append(e)
            routes_depot.append(d)
            routes.append(c)
            clients_temp = [item for item in clients_temp if item not in c]
        population.append([routes_depot, routes, routes_vehicles])
    return population

# Function: Fitness
def fitness_function(cost, population_size): 
    fitness = np.zeros((population_size, 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1 + cost[i][0] + abs(np.min(cost)))
    fit_sum      = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix     = 0
    random = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
          ix = i
          break
    return ix

# Function: TSP Crossover - BRBAX (Best Route Better Adjustment Recombination)
def crossover_tsp_brbax(parent_1, parent_2):
    offspring = copy.deepcopy(parent_2)
    cut       = random.sample(list(range(0,len(parent_1[1][0]))), 2)
    cut.sort()
    rand      = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    A         = parent_1[1][0][cut[0]:cut[1]]
    B         = [item for item in parent_2[1][0] if item not in A ] 
    if (rand > 0.5):
        A.reverse()
    offspring[1][0] = A + B
    return offspring

# Function: TSP Crossover - BCR (Best Cost Route Crossover)
def crossover_tsp_bcr(parent_1, parent_2, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window, parameters, route):
    offspring = copy.deepcopy(parent_2)
    cut       = random.sample(list(range(0,len(parent_1[1][0]))), 2)
    for i in range(0, 2):
        d_1            = float('+inf')
        A              = parent_1[1][0][cut[i]]
        best           = []
        parent_2[1][0] = [item for item in parent_2[1][0] if item not in [A] ]
        insertion      = copy.deepcopy([ parent_2[0][0], parent_2[1][0], parent_2[2][0] ])
        dist_list      = [evaluate_distance(distance_matrix, insertion[0], insertion[1][:n] + [A] + insertion[1][n:]) for n in range(0, len(parent_2[1][0]) + 1)]
        if(time_window == 'with'):
            wait_time_list = [evaluate_time(distance_matrix, parameters, insertion[0], insertion[1][:n] + [A] + insertion[1][n:], velocity = [velocity[parent_2[2][0][0]] ] ) for n in range(0, len(parent_2[1][0]) + 1)]
        else:
            wait_time_list = [[0, 0]]*len(dist_list)
        cap_list       = [evaluate_capacity(parameters, insertion[0], insertion[1][:n] + [A] + insertion[1][n:]) for n in range(0, len(parent_2[1][0]) + 1)]
        insertion_list = [insertion[1][:n] + [A] + insertion[1][n:] for n in range(0, len(parent_2[1][0]) + 1)]
        d_2_list       = [evaluate_cost_penalty(dist_list[n], wait_time_list[n][1], wait_time_list[n][0], cap_list[n], capacity[parent_2[2][0][0]], parameters, insertion[0], insertion_list[n], [fixed_cost[parent_2[2][0][0]]], [variable_cost[parent_2[2][0][0]]], penalty_value, time_window, route) for n in range(0, len(dist_list))]
        d_2            = min(d_2_list)
        if (d_2 <= d_1):
            d_1   = d_2
            best  = insertion_list[d_2_list.index(min(d_2_list))]
        parent_2[1][0] = best
        if (d_1 != float('+inf')):
            offspring = copy.deepcopy(parent_2)
    return offspring

# Function: VRP Crossover - BRBAX (Best Route Better Adjustment Recombination)
def crossover_vrp_brbax(parent_1, parent_2):
    s         = random.sample(list(range(0,len(parent_1[0]))), 1)[0]
    subroute  = [ parent_1[0][s], parent_1[1][s], parent_1[2][s] ]
    offspring = copy.deepcopy(parent_2)
    for k in range(len(parent_2[1])-1, -1, -1):
        offspring[1][k] = [item for item in offspring[1][k] if item not in subroute[1] ] 
        if (len(offspring[1][k]) == 0):
            del offspring[0][k]
            del offspring[1][k]
            del offspring[2][k]
    offspring[0].append(subroute[0])
    offspring[1].append(subroute[1])
    offspring[2].append(subroute[2])
    return offspring

# Function: VRP Crossover - BCR (Best Cost Route Crossover)
def crossover_vrp_bcr(parent_1, parent_2, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window, parameters, route):
    s         = random.sample(list(range(0,len(parent_1[0]))), 1)[0]
    offspring = copy.deepcopy(parent_2)
    if (len(parent_1[1][s]) > 1):
        cut  = random.sample(list(range(0,len(parent_1[1][s]))), 2)
        gene = 2
    else:
        cut  = [0, 0]
        gene = 1
    for i in range(0, gene):
        d_1   = float('+inf')
        ins_m = 0
        A     = parent_1[1][s][cut[i]]
        best  = []
        for m in range(0, len(parent_2[1])):
            parent_2[1][m] = [item for item in parent_2[1][m] if item not in [A] ]
            if (len(parent_2[1][m]) > 0):
                insertion      = copy.deepcopy([ parent_2[0][m], parent_2[1][m], parent_2[2][m] ])
                dist_list      = [evaluate_distance(distance_matrix, insertion[0], insertion[1][:n] + [A] + insertion[1][n:]) for n in range(0, len(parent_2[1][m]) + 1)]
                if(time_window == 'with'):
                    wait_time_list = [evaluate_time(distance_matrix, parameters, insertion[0], insertion[1][:n] + [A] + insertion[1][n:], velocity = [velocity[parent_2[2][m][0]] ] ) for n in range(0, len(parent_2[1][m]) + 1)]
                else:
                    wait_time_list = [[0, 0]]*len(dist_list)
                cap_list       = [evaluate_capacity(parameters, insertion[0], insertion[1][:n] + [A] + insertion[1][n:]) for n in range(0, len(parent_2[1][m]) + 1)]
                insertion_list = [insertion[1][:n] + [A] + insertion[1][n:] for n in range(0, len(parent_2[1][m]) + 1)]
                d_2_list       = [evaluate_cost_penalty(dist_list[n], wait_time_list[n][1], wait_time_list[n][0], cap_list[n], capacity[parent_2[2][m][0]], parameters, insertion[0], insertion_list[n], [fixed_cost[parent_2[2][m][0]]], [variable_cost[parent_2[2][m][0]]], penalty_value, time_window, route) for n in range(0, len(dist_list))]
                d_2 = min(d_2_list)
                if (d_2 <= d_1):
                    d_1   = d_2
                    ins_m = m
                    best  = insertion_list[d_2_list.index(min(d_2_list))]
        parent_2[1][ins_m] = best            
        if (d_1 != float('+inf')):
            offspring = copy.deepcopy(parent_2)
    for i in range(len(offspring[1])-1, -1, -1):
        if(len(offspring[1][i]) == 0):
            del offspring[0][i]
            del offspring[1][i]
            del offspring[2][i]
    return offspring

# Function: Breeding
def breeding(cost, population, fitness, distance_matrix, n_depots, elite, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window, parameters, route, vehicle_types, fleet_size):
    offspring = copy.deepcopy(population) 
    if (elite > 0):
        cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
        for i in range(0, elite):
            offspring[i] = copy.deepcopy(population[i])
    for i in range (elite, len(offspring)):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        parent_1 = copy.deepcopy(population[parent_1])  
        parent_2 = copy.deepcopy(population[parent_2])
        rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)  
        # TSP - Crossover
        if (len(parent_1[1]) == 1 and len(parent_2[1]) == 1):
            if (rand > 0.5):
                offspring[i] = crossover_tsp_brbax(parent_1, parent_2)
                offspring[i] = crossover_tsp_bcr(offspring[i], parent_2, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window = time_window, parameters = parameters, route = route)
            elif (rand <= 0.5): 
                offspring[i] = crossover_tsp_brbax(parent_2, parent_1)
                offspring[i] = crossover_tsp_bcr(offspring[i], parent_1, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window = time_window, parameters = parameters, route = route)
        # VRP - Crossover
        elif((len(parent_1[1]) > 1 and len(parent_2[1]) > 1)):
            if (rand > 0.5):
                offspring[i] = crossover_vrp_brbax(parent_1, parent_2)
                offspring[i] = crossover_vrp_bcr(offspring[i], parent_2, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window = time_window, parameters = parameters, route = route)              
            elif (rand <= 0.5): 
                offspring[i] = crossover_vrp_brbax(parent_2, parent_1)
                offspring[i] = crossover_vrp_bcr(offspring[i], parent_1, distance_matrix, velocity, capacity, fixed_cost, variable_cost, penalty_value, time_window = time_window, parameters = parameters, route = route)
        if (n_depots > 1):
            offspring[i] = evaluate_depot(n_depots, offspring[i], distance_matrix) 
        if (vehicle_types > 1):
            offspring[i] = evaluate_vehicle(vehicle_types, offspring[i], distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, penalty_value, time_window, route, fleet_size)
    offspring[i] = cap_break(vehicle_types, offspring[i], parameters, capacity)
    return offspring

# Function: Mutation - Swap
def mutation_tsp_vrp_swap(individual):
    if (len(individual[1]) == 1):
        k1 = random.sample(list(range(0, len(individual[1]))), 1)[0]
        k2 = k1
    else:
        k  = random.sample(list(range(0, len(individual[1]))), 2)
        k1 = k[0]
        k2 = k[1]  
    cut1                    = random.sample(list(range(0, len(individual[1][k1]))), 1)[0]
    cut2                    = random.sample(list(range(0, len(individual[1][k2]))), 1)[0]
    A                       = individual[1][k1][cut1]
    B                       = individual[1][k2][cut2]
    individual[1][k1][cut1] = B
    individual[1][k2][cut2] = A
    return individual

# Function: Mutation - Insertion
def mutation_tsp_vrp_insertion(individual):
    if (len(individual[1]) == 1):
        k1 = random.sample(list(range(0, len(individual[1]))), 1)[0]
        k2 = k1
    else:
        k  = random.sample(list(range(0, len(individual[1]))), 2)
        k1 = k[0]
        k2 = k[1]
    cut1 = random.sample(list(range(0, len(individual[1][k1])))  , 1)[0]
    cut2 = random.sample(list(range(0, len(individual[1][k2])+1)), 1)[0]
    A    = individual[1][k1][cut1]
    del individual[1][k1][cut1]
    individual[1][k2][cut2:cut2] = [A]
    if (len(individual[1][k1]) == 0):
        del individual[0][k1]
        del individual[1][k1]
        del individual[2][k1]
    return individual

# Function: Mutation
def mutation(offspring, mutation_rate, elite):
    for i in range(elite, len(offspring)):
        probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (probability <= mutation_rate):
            rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (rand <= 0.5):
                offspring[i] = mutation_tsp_vrp_insertion(offspring[i])
            elif(rand > 0.5):
                offspring[i] = mutation_tsp_vrp_swap(offspring[i])
        for k in range(0, len(offspring[i][1])):
            if (len(offspring[i][1][k]) >= 2):
                probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                if (probability <= mutation_rate):
                    rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    cut  = random.sample(list(range(0, len(offspring[i][1][k]))), 2)
                    cut.sort()
                    C    = offspring[i][1][k][cut[0]:cut[1]+1]
                    if (rand <= 0.5):
                        random.shuffle(C)
                    elif(rand > 0.5):
                        C.reverse()
                    offspring[i][1][k][cut[0]:cut[1]+1] = C
    return offspring

# Function: Elite Distance
def elite_distance(individual, distance_matrix, route):
    if (route == 'open'):
        end = 2
    else:
        end = 1
    td = 0
    for n in range(0, len(individual[1])):
        td = td + evaluate_distance(distance_matrix, depot = individual[0][n], subroute = individual[1][n])[-end]
    return round(td,2)

# GA-VRP Function
def genetic_algorithm_vrp(coordinates, distance_matrix, parameters, velocity, fixed_cost, variable_cost, capacity, population_size = 5, vehicle_types = 1, n_depots = 1, route = 'closed', model = 'vrp', time_window = 'without', fleet_size = [], mutation_rate = 0.1, elite = 0, generations = 50, penalty_value = 1000, graph = True, selection = 'rw'):    
    start           = tm.time()
    count           = 0
    solution_report = ['None']
    max_capacity    = copy.deepcopy(capacity)
    if (model == 'tsp'):
        n_depots = 1
        for i in range(0, len(max_capacity)):
            max_capacity[i] = float('+inf') 
    if (model == 'mtsp'):
        for i in range(0, len(max_capacity)):
            max_capacity[i] = float('+inf') 
    for i in range(0, n_depots):
        parameters[i, 0] = 0  
    population       = initial_population(coordinates, distance_matrix, population_size = population_size, vehicle_types = vehicle_types, n_depots = n_depots, model = model)
    cost, population = target_function(population, distance_matrix, parameters, velocity, fixed_cost, variable_cost, max_capacity, penalty_value, time_window = time_window, route = route, fleet_size = fleet_size) 
    cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
    if (selection == 'rw'):
        fitness          = fitness_function(cost, population_size)
    elif (selection == 'rb'):
        rank             = [[i] for i in range(1, len(cost)+1)]
        fitness          = fitness_function(rank, population_size)
    elite_ind        = elite_distance(population[0], distance_matrix, route = route)
    cost             = copy.deepcopy(cost)
    elite_cst        = copy.deepcopy(cost[0][0])
    solution         = copy.deepcopy(population[0])
    print('Generation = ', count, ' Distance = ', elite_ind, ' f(x) = ', round(elite_cst, 2)) 
    while (count <= generations-1): 
        offspring        = breeding(cost, population, fitness, distance_matrix, n_depots, elite, velocity, max_capacity, fixed_cost, variable_cost, penalty_value, time_window, parameters, route, vehicle_types, fleet_size)   
        offspring        = mutation(offspring, mutation_rate = mutation_rate, elite = elite)
        cost, population = target_function(offspring, distance_matrix, parameters, velocity, fixed_cost, variable_cost, max_capacity, penalty_value, time_window = time_window, route = route, fleet_size = fleet_size)
        cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
        elite_child      = elite_distance(population[0], distance_matrix, route = route)
        if (selection == 'rw'):
            fitness = fitness_function(cost, population_size)
        elif (selection == 'rb'):
            rank    = [[i] for i in range(1, len(cost)+1)]
            fitness = fitness_function(rank, population_size)
        if(elite_ind > elite_child):
            elite_ind = elite_child 
            solution  = copy.deepcopy(population[0])
            elite_cst = copy.deepcopy(cost[0][0])
        count = count + 1  
        print('Generation = ', count, ' Distance = ', elite_ind, ' f(x) = ', round(elite_cst, 2))
    if (graph == True):
        plot_tour_coordinates(coordinates, solution, n_depots = n_depots, route = route)
    solution_report = show_report(solution, distance_matrix, parameters, velocity, fixed_cost, variable_cost, route = route, time_window  = time_window)
    end = tm.time()
    print('Algorithm Time: ', round((end - start), 2), ' seconds')
    return solution_report, solution
   
   ############################################################################
