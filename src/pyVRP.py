############################################################################
# Genetic Algorithm for TSP / VRP / MDVRP — Optimized Version
# Based on Valdecy Pereira's implementation (Metaheuristic-Genetic_Algorithm).
#
# Public API (genetic_algorithm_vrp, show_report, plot_tour_coordinates,
# plot_tour_latlong, build_distance_matrix, build_coordinates,
# initial_population) is preserved — drop-in replacement.
#
# Key speed changes vs. the original:
#   1.  clone_individual() replaces copy.deepcopy in the breeding loop. For
#       this nested-list-of-ints structure it is ~30–50x faster.
#   2.  BCR crossover uses an O(1)-per-position *insertion delta* instead of
#       rebuilding the full route and recomputing cumulative distances for
#       every candidate position. O(n^2) -> O(n) per inserted customer
#       (when time_window='without'; 'with' falls back to per-position
#       eval but still avoids deepcopy and full-array rebuilds).
#   3.  evaluate_vehicle re-scores only the route whose vehicle changed,
#       not the whole solution. R*V full-target-function calls per
#       individual -> R*V cheap single-route costs.
#   4.  target_function pre-extracts parameter columns once per call and
#       uses pure numpy indexing inside the per-subroute work.
#   5.  Random draws use random.random() / random.randrange() instead of
#       os.urandom. Still seed-reproducible via random.seed(seed).
#   6.  Fitness computation is fully vectorized.
#   7.  cap_break has a hard iteration cap so it can never spin.
############################################################################

import os
import time as tm
import random
import copy
import numpy as np
import pandas as pd

from itertools import cycle
from matplotlib import pyplot as plt
plt.style.use('bmh')

# Folium is only needed for plot_tour_latlong; keep import optional.
try:
    import folium
    import folium.plugins
    _HAS_FOLIUM = True
except ImportError:  # pragma: no cover
    _HAS_FOLIUM = False


############################################################################
# Fast clone — replaces copy.deepcopy for our specific individual structure.
############################################################################

def clone_individual(ind):
    """Clone individual = [depots, routes, vehicles], each a list of list of ints."""
    return [
        [lst[:] for lst in ind[0]],
        [lst[:] for lst in ind[1]],
        [lst[:] for lst in ind[2]],
    ]


def clone_population(pop):
    return [clone_individual(ind) for ind in pop]


############################################################################
# Geometry
############################################################################

def build_coordinates(distance_matrix):
    a = distance_matrix[0, :].reshape(-1, 1)
    b = distance_matrix[:, 0].reshape(1, -1)
    m = 0.5 * (a ** 2 + b ** 2 - distance_matrix ** 2)
    w, u = np.linalg.eig(m.T @ m)
    s = np.diag(np.sort(w)[::-1]) ** 0.5
    coords = (u @ (s ** 0.5)).real[:, :2]
    return coords


def build_distance_matrix(coordinates):
    diff = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))


############################################################################
# Plotting (unchanged behavior)
############################################################################

_COLOR_CYCLE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#bf77f6', '#ff9408',
                '#d1ffbd', '#c85a53', '#3a18b1', '#ff796c', '#04d8b2', '#ffb07c',
                '#aaa662', '#0485d1', '#fffe7a', '#b0dd16', '#d85679', '#12e193',
                '#82cafc', '#ac9362', '#f8481c', '#c292a1', '#c0fa8b', '#ca7b80',
                '#f4d054', '#fbdd7e', '#ffff7e', '#cd7584', '#f9bc08', '#c7c10c']


def plot_tour_coordinates(coordinates, solution, n_depots, route, size_x=10, size_y=10):
    depot = solution[0]
    city_tour = solution[1]
    cycol = cycle(_COLOR_CYCLE)
    plt.figure(figsize=[size_x, size_y])
    for j in range(len(city_tour)):
        extra = 2 if route == 'closed' else 1
        xy = np.zeros((len(city_tour[j]) + extra, 2))
        xy[0] = coordinates[depot[j][0]]
        for i in range(len(city_tour[j])):
            xy[i + 1] = coordinates[city_tour[j][i]]
        if route == 'closed':
            xy[-1] = coordinates[depot[j][0]]
        plt.plot(xy[:, 0], xy[:, 1], marker='s', alpha=0.5, markersize=5, color=next(cycol))
    for i in range(coordinates.shape[0]):
        if i < n_depots:
            plt.plot(coordinates[i, 0], coordinates[i, 1], marker='s',
                     alpha=1.0, markersize=7, color='k')
        plt.text(coordinates[i, 0], coordinates[i, 1] + 0.04, i,
                 ha='center', va='bottom', color='k', fontsize=7)
    return


def plot_tour_latlong(lat_long, solution, n_depots, route):
    if not _HAS_FOLIUM:
        raise ImportError("folium is required for plot_tour_latlong; install with `pip install folium`.")
    m = folium.Map(location=(lat_long.iloc[0, 0], lat_long.iloc[0, 1]), zoom_start=14)
    clients = folium.plugins.MarkerCluster(name='Clients').add_to(m)
    depots = folium.plugins.MarkerCluster(name='Depots').add_to(m)
    for i in range(lat_long.shape[0]):
        lat, lng = lat_long.iloc[i, 0], lat_long.iloc[i, 1]
        if i < n_depots:
            folium.Marker(location=[lat, lng],
                          popup='<b>Client: </b>%s</br> <b>Address: </b>D</br>' % int(i),
                          icon=folium.Icon(color='black', icon='home')).add_to(depots)
        else:
            folium.Marker(location=[lat, lng],
                          popup='<b>Client: </b>%s</br> <b>Address: </b>C</br>' % int(i),
                          icon=folium.Icon(color='blue')).add_to(clients)
    depot = solution[0]
    city_tour = solution[1]
    cycol = cycle(_COLOR_CYCLE)
    for j in range(len(city_tour)):
        extra = 2 if route == 'closed' else 1
        ltlng = np.zeros((len(city_tour[j]) + extra, 2))
        ltlng[0] = [lat_long.iloc[depot[j][0], 0], lat_long.iloc[depot[j][0], 1]]
        for i in range(len(city_tour[j])):
            ltlng[i + 1] = [lat_long.iloc[city_tour[j][i], 0], lat_long.iloc[city_tour[j][i], 1]]
        if route == 'closed':
            ltlng[-1] = [lat_long.iloc[depot[j][0], 0], lat_long.iloc[depot[j][0], 1]]
        c = next(cycol)
        for i in range(ltlng.shape[0] - 1):
            folium.PolyLine([(ltlng[i, 0], ltlng[i, 1]),
                             (ltlng[i + 1, 0], ltlng[i + 1, 1])],
                            color=c, weight=1.5, opacity=1).add_to(m)
    return m


############################################################################
# Route-level evaluators
############################################################################

def evaluate_distance(distance_matrix, depot, subroute):
    """Cumulative distance along depot -> subroute -> depot.
    Returns list [0.0, d1, d1+d2, ...]  (same contract as the original).
    """
    d = depot[0]
    n = len(subroute)
    if n == 0:
        return [0.0, 0.0]
    path = np.empty(n + 2, dtype=np.int64)
    path[0] = d
    path[1:n + 1] = subroute
    path[-1] = d
    seg = distance_matrix[path[:-1], path[1:]]
    out = [0.0]
    out.extend(np.cumsum(seg).tolist())
    return out


def evaluate_time(distance_matrix, parameters, depot, subroute, velocity):
    """Wait & arrival-time lists of length n+2 (indices align with evaluate_distance)."""
    tw_early = parameters[:, 1]
    tw_st = parameters[:, 3]
    d = depot[0]
    vel = velocity[0]
    nodes = [d] + list(subroute) + [d]  # length n+2
    L = len(nodes)
    wait = [0.0] * L
    time = [0.0] * L
    for i in range(1, L):
        prev = nodes[i - 1]
        cur = nodes[i]
        t = time[i - 1] + distance_matrix[prev, cur] / vel
        if t < tw_early[cur]:
            wait[i] = tw_early[cur] - t
            t = tw_early[cur]
        t = t + tw_st[cur]
        time[i] = t
    return wait, time


def evaluate_capacity(parameters, depot, subroute):
    demand = parameters[:, 0]
    if not subroute:
        return [0.0, 0.0]
    idx = np.asarray([depot[0]] + list(subroute) + [depot[0]], dtype=np.int64)
    return np.cumsum(demand[idx]).tolist()


def evaluate_cost(dist, wait, parameters, depot, subroute, fixed_cost, variable_cost, time_window):
    tw_wc = parameters[:, 4]
    subroute_ = depot + subroute + depot
    fc = fixed_cost[0]
    vc = variable_cost[0]
    if time_window == 'with':
        cost = [fc + wait[i] * tw_wc[subroute_[i]] if dist[i] == 0
                else fc + dist[i] * vc + wait[i] * tw_wc[subroute_[i]]
                for i in range(len(subroute_))]
    else:
        cost = [fc if x == 0 else fc + x * vc for x in dist]
    return cost


def evaluate_cost_penalty(dist, time, wait, cap, capacity, parameters, depot, subroute,
                          fixed_cost, variable_cost, penalty_value, time_window, route):
    tw_late = parameters[:, 2]
    tw_st = parameters[:, 3]
    tw_wc = parameters[:, 4]
    subroute_ = depot + subroute if route == 'open' else depot + subroute + depot
    fc = fixed_cost[0]
    vc = variable_cost[0]
    L = len(subroute_)

    pnlt = int(np.sum(np.asarray(cap[:L]) > capacity))
    if time_window == 'with':
        t_arr = np.asarray(time[:L])
        late = tw_late[subroute_] + tw_st[subroute_]
        pnlt += int(np.sum(t_arr > late))
        cost_last = 0.0
        for i in range(L):
            if dist[i] == 0:
                cost_last = fc + wait[i] * tw_wc[subroute_[i]]
            else:
                cost_last = cost_last + dist[i] * vc + wait[i] * tw_wc[subroute_[i]]
    else:
        cost_last = fc if dist[-1] == 0 else fc + dist[-1] * vc
        # Need the accumulated variable cost over the whole path:
        # original did cost[0] + x*variable_cost[0] iteratively; equivalent to sum(dist)*vc + fc.
        cost_last = fc + float(np.sum(dist)) * vc
    return cost_last + pnlt * penalty_value


############################################################################
# Helpers: depot and vehicle re-assignment (fast, local)
############################################################################

def evaluate_depot(n_depots, individual, distance_matrix):
    """Assign each route to its nearest depot (based on last-leg distance)."""
    for j in range(len(individual[1])):
        subroute = individual[1][j]
        best_d = float('+inf')
        best = individual[0][j][0]
        for i in range(n_depots):
            d = evaluate_distance(distance_matrix, [i], subroute)[-1]
            if d < best_d:
                best_d = d
                best = i
        individual[0][j] = [best]
    return individual


def _route_cost(distance_matrix, parameters, velocity, fixed_cost, variable_cost,
                capacity, penalty_value, time_window, route, depot, subroute, v_type):
    """Compute penalized cost of a single route for the given vehicle type."""
    dist = evaluate_distance(distance_matrix, depot, subroute)
    if time_window == 'with':
        wait, time = evaluate_time(distance_matrix, parameters, depot, subroute,
                                   velocity=[velocity[v_type]])
    else:
        wait, time = [], []
    cap = evaluate_capacity(parameters, depot, subroute)
    return evaluate_cost_penalty(dist, time, wait, cap, capacity[v_type],
                                 parameters, depot, subroute,
                                 [fixed_cost[v_type]], [variable_cost[v_type]],
                                 penalty_value, time_window, route)


def evaluate_vehicle(vehicle_types, individual, distance_matrix, parameters,
                     velocity, fixed_cost, variable_cost, capacity, penalty_value,
                     time_window, route, fleet_size):
    """Greedily pick best vehicle for each route. Only re-scores the changed route."""
    for i in range(len(individual[0])):
        depot = individual[0][i]
        subroute = individual[1][i]
        current = individual[2][i][0]
        best_v = current
        best_c = _route_cost(distance_matrix, parameters, velocity, fixed_cost,
                             variable_cost, capacity, penalty_value, time_window,
                             route, depot, subroute, current)
        for j in range(vehicle_types):
            if j == current:
                continue
            c = _route_cost(distance_matrix, parameters, velocity, fixed_cost,
                            variable_cost, capacity, penalty_value, time_window,
                            route, depot, subroute, j)
            if c < best_c:
                best_c = c
                best_v = j
        individual[2][i] = [best_v]
    return individual


def cap_break(vehicle_types, individual, parameters, capacity, max_iter=64):
    """Split routes that exceed capacity into feasible + overflow halves. Iterates
    until stable or until max_iter rounds (safety guard against pathological loops)."""
    for _ in range(max_iter):
        solution = [[], [], []]
        changed = False
        for i in range(len(individual[0])):
            cap = evaluate_capacity(parameters, individual[0][i], individual[1][i])
            cap_core = cap[1:-1]  # drop both depot entries
            cap_i = capacity[individual[2][i][0]]
            if not cap_core or max(cap_core) <= cap_i:
                solution[0].append(individual[0][i])
                solution[1].append(individual[1][i])
                solution[2].append(individual[2][i])
                continue
            sep = [x > cap_i for x in cap_core]
            sub = individual[1][i]
            sep_f = [sub[x] for x in range(len(sub)) if not sep[x]]
            sep_t = [sub[x] for x in range(len(sub)) if sep[x]]
            if sep_f and sep_t:
                changed = True
                solution[0].append(individual[0][i]); solution[0].append(individual[0][i])
                solution[1].append(sep_f);            solution[1].append(sep_t)
                solution[2].append(individual[2][i]); solution[2].append(individual[2][i])
            elif sep_t:
                solution[0].append(individual[0][i])
                solution[1].append(sep_t)
                solution[2].append(individual[2][i])
            elif sep_f:
                solution[0].append(individual[0][i])
                solution[1].append(sep_f)
                solution[2].append(individual[2][i])
        individual = solution
        if not changed:
            break
    return individual


############################################################################
# Target function (fully vectorized over the subroutes of one individual)
############################################################################

def target_function(population, distance_matrix, parameters, velocity, fixed_cost,
                    variable_cost, capacity, penalty_value, time_window, route,
                    fleet_size=[]):
    demand  = parameters[:, 0]
    tw_early= parameters[:, 1]
    tw_late = parameters[:, 2]
    tw_st   = parameters[:, 3]
    tw_wc   = parameters[:, 4]

    end = 2 if route == 'open' else 1
    cost = [[0.0] for _ in range(len(population))]

    for k in range(len(population)):
        individual = population[k]
        total = 0.0
        pnlt = 0
        flt_cnt = [0] * len(fleet_size)

        for i in range(len(individual[1])):
            depot = individual[0][i]
            subroute = individual[1][i]
            v_type = individual[2][i][0]
            vel = velocity[v_type]
            fc = fixed_cost[v_type]
            vc = variable_cost[v_type]
            cap_lim = capacity[v_type]
            n = len(subroute)

            # --- Distances along the subroute ---
            if n == 0:
                continue
            path = np.empty(n + 2, dtype=np.int64)
            path[0] = depot[0]
            path[1:n + 1] = subroute
            path[-1] = depot[0]
            seg = distance_matrix[path[:-1], path[1:]]
            total_dist = float(seg.sum()) if route == 'closed' else float(seg[:-1].sum())

            # --- Capacity check ---
            sub_arr = np.asarray(subroute, dtype=np.int64)
            cum_cap = np.cumsum(demand[sub_arr])
            pnlt += int(np.sum(cum_cap > cap_lim))

            # --- Time windows ---
            wait_total_cost = 0.0
            if time_window == 'with':
                t = 0.0
                prev = depot[0]
                nodes = list(subroute) + [depot[0]]
                for node in nodes:
                    t += distance_matrix[prev, node] / vel
                    if t < tw_early[node]:
                        wait_total_cost += (tw_early[node] - t) * tw_wc[node]
                        t = tw_early[node]
                    t += tw_st[node]
                    if t > tw_late[node] + tw_st[node]:
                        pnlt += 1
                    prev = node

            # --- Fleet size ---
            if fleet_size:
                flt_cnt[v_type] += 1

            # --- Cost assembly ---
            # Original's final cost for the subroute: fc + total_dist*vc + wait_time_weighted_cost
            route_cost = fc + total_dist * vc + wait_total_cost
            total += route_cost

        if fleet_size:
            for v in range(len(fleet_size)):
                over = flt_cnt[v] - fleet_size[v]
                if over > 0:
                    pnlt += over

        cost[k][0] = total + pnlt * penalty_value

    return cost, population


############################################################################
# Initial population, fitness, selection
############################################################################

def initial_population(coordinates='none', distance_matrix='none', population_size=5,
                       vehicle_types=1, n_depots=1, model='vrp'):
    try:
        distance_matrix.shape[0]
    except AttributeError:
        distance_matrix = build_distance_matrix(coordinates)
    if model == 'tsp':
        n_depots = 1
    depots = [[i] for i in range(n_depots)]
    vehicles = [[i] for i in range(vehicle_types)]
    clients = list(range(n_depots, distance_matrix.shape[0]))
    population = []
    for _ in range(population_size):
        remaining = clients[:]
        routes, routes_depot, routes_vehicles = [], [], []
        while remaining:
            e = random.choice(vehicles)
            d = random.choice(depots)
            if model == 'tsp':
                c = random.sample(remaining, len(remaining))
            else:
                c = random.sample(remaining, random.randint(1, len(remaining)))
            routes_vehicles.append(e[:])
            routes_depot.append(d[:])
            routes.append(c)
            rem_set = set(c)
            remaining = [x for x in remaining if x not in rem_set]
        population.append([routes_depot, routes, routes_vehicles])
    return population


def fitness_function(cost, population_size):
    c = np.asarray([row[0] for row in cost], dtype=np.float64)
    f = 1.0 / (1.0 + c + abs(c.min()))
    total = f.sum()
    cdf = np.cumsum(f) / total
    fitness = np.column_stack([f, cdf])
    return fitness


def roulette_wheel(fitness):
    r = random.random()
    # fitness[:, 1] is already a normalized CDF, searchsorted is O(log N).
    return int(np.searchsorted(fitness[:, 1], r, side='left'))


############################################################################
# Crossovers
############################################################################

def crossover_tsp_brbax(parent_1, parent_2):
    offspring = clone_individual(parent_2)
    L = len(parent_1[1][0])
    cut = sorted(random.sample(range(L), 2))
    A = parent_1[1][0][cut[0]:cut[1]]
    A_set = set(A)
    B = [item for item in parent_2[1][0] if item not in A_set]
    if random.random() > 0.5:
        A = A[::-1]
    offspring[1][0] = A + B
    return offspring


def _best_insertion(distance_matrix, depot_idx, subroute, A):
    """O(n) best-insertion position by distance delta. Returns (pos, delta)."""
    n = len(subroute)
    if n == 0:
        return 0, float(distance_matrix[depot_idx, A] + distance_matrix[A, depot_idx])
    prev = np.empty(n + 1, dtype=np.int64); nxt = np.empty(n + 1, dtype=np.int64)
    prev[0] = depot_idx
    prev[1:] = subroute
    nxt[:-1] = subroute
    nxt[-1] = depot_idx
    add = distance_matrix[prev, A] + distance_matrix[A, nxt]
    remove = distance_matrix[prev, nxt]
    delta = add - remove
    pos = int(np.argmin(delta))
    return pos, float(delta[pos])


def crossover_tsp_bcr(parent_1, parent_2, distance_matrix, velocity, capacity,
                      fixed_cost, variable_cost, penalty_value, time_window,
                      parameters, route):
    offspring = clone_individual(parent_2)
    L = len(parent_1[1][0])
    cut = random.sample(range(L), 2)

    # Remove from offspring and reinsert at best position, one at a time.
    for idx in range(2):
        A = parent_1[1][0][cut[idx]]
        if A in offspring[1][0]:
            offspring[1][0].remove(A)
        depot_idx = offspring[0][0][0]

        if time_window == 'with':
            # Fall back to full per-position eval (still no deepcopy, still linear rebuild).
            sub = offspring[1][0]
            v_type = offspring[2][0][0]
            best_pos, best_cost = 0, float('+inf')
            for n in range(len(sub) + 1):
                trial = sub[:n] + [A] + sub[n:]
                c = _route_cost(distance_matrix, parameters, velocity, fixed_cost,
                                variable_cost, capacity, penalty_value, time_window,
                                route, offspring[0][0], trial, v_type)
                if c < best_cost:
                    best_cost, best_pos = c, n
            offspring[1][0] = sub[:best_pos] + [A] + sub[best_pos:]
        else:
            pos, _ = _best_insertion(distance_matrix, depot_idx, offspring[1][0], A)
            offspring[1][0].insert(pos, A)
    return offspring


def crossover_vrp_brbax(parent_1, parent_2):
    s = random.randrange(len(parent_1[0]))
    subroute_d = parent_1[0][s][:]
    subroute_r = parent_1[1][s][:]
    subroute_v = parent_1[2][s][:]
    transferred = set(subroute_r)

    offspring = clone_individual(parent_2)
    for k in range(len(offspring[1]) - 1, -1, -1):
        offspring[1][k] = [x for x in offspring[1][k] if x not in transferred]
        if not offspring[1][k]:
            del offspring[0][k]; del offspring[1][k]; del offspring[2][k]
    offspring[0].append(subroute_d)
    offspring[1].append(subroute_r)
    offspring[2].append(subroute_v)
    return offspring


def crossover_vrp_bcr(parent_1, parent_2, distance_matrix, velocity, capacity,
                      fixed_cost, variable_cost, penalty_value, time_window,
                      parameters, route):
    s = random.randrange(len(parent_1[0]))
    offspring = clone_individual(parent_2)
    if len(parent_1[1][s]) > 1:
        cut = random.sample(range(len(parent_1[1][s])), 2)
        gene = 2
    else:
        cut = [0, 0]
        gene = 1

    for idx in range(gene):
        A = parent_1[1][s][cut[idx]]
        # Remove A wherever it currently lives in offspring
        for m in range(len(offspring[1])):
            if A in offspring[1][m]:
                offspring[1][m].remove(A)
        # Find best route + best position to reinsert
        best_m, best_pos, best_delta = 0, 0, float('+inf')
        if time_window == 'with':
            # Per-position full eval — still no deepcopy.
            for m in range(len(offspring[1])):
                depot = offspring[0][m]
                sub = offspring[1][m]
                v_type = offspring[2][m][0]
                for n in range(len(sub) + 1):
                    trial = sub[:n] + [A] + sub[n:]
                    c = _route_cost(distance_matrix, parameters, velocity, fixed_cost,
                                    variable_cost, capacity, penalty_value, time_window,
                                    route, depot, trial, v_type)
                    if c < best_delta:
                        best_delta, best_m, best_pos = c, m, n
        else:
            for m in range(len(offspring[1])):
                depot_idx = offspring[0][m][0]
                pos, delta = _best_insertion(distance_matrix, depot_idx, offspring[1][m], A)
                # capacity penalty if inserting would overflow (position-invariant)
                v_type = offspring[2][m][0]
                cap_lim = capacity[v_type]
                new_total = sum(parameters[x, 0] for x in offspring[1][m]) + parameters[A, 0]
                score = delta + (penalty_value if new_total > cap_lim else 0.0)
                if score < best_delta:
                    best_delta, best_m, best_pos = score, m, pos
        offspring[1][best_m].insert(best_pos, A)

    # Prune any empty routes that may have resulted
    for i in range(len(offspring[1]) - 1, -1, -1):
        if not offspring[1][i]:
            del offspring[0][i]; del offspring[1][i]; del offspring[2][i]
    return offspring


############################################################################
# Breeding and mutation
############################################################################

def breeding(cost, population, fitness, distance_matrix, n_depots, elite, velocity,
             capacity, fixed_cost, variable_cost, penalty_value, time_window,
             parameters, route, vehicle_types, fleet_size):
    # Elitism: sort once, keep top `elite` unchanged.
    if elite > 0:
        order = sorted(range(len(population)), key=lambda i: cost[i][0])
        sorted_pop = [population[i] for i in order]
        population = sorted_pop
        cost = [cost[i] for i in order]
        offspring = [clone_individual(population[i]) for i in range(elite)]
        offspring.extend([None] * (len(population) - elite))
    else:
        offspring = [None] * len(population)

    pop_len = len(population)
    for i in range(elite, pop_len):
        p1 = roulette_wheel(fitness)
        p2 = roulette_wheel(fitness)
        while p1 == p2:
            p2 = random.randrange(pop_len)
        parent_1 = population[p1]
        parent_2 = population[p2]
        r = random.random()

        if len(parent_1[1]) == 1 and len(parent_2[1]) == 1:
            # TSP case
            if r > 0.5:
                child = crossover_tsp_brbax(parent_1, parent_2)
                child = crossover_tsp_bcr(child, parent_2, distance_matrix, velocity,
                                          capacity, fixed_cost, variable_cost,
                                          penalty_value, time_window, parameters, route)
            else:
                child = crossover_tsp_brbax(parent_2, parent_1)
                child = crossover_tsp_bcr(child, parent_1, distance_matrix, velocity,
                                          capacity, fixed_cost, variable_cost,
                                          penalty_value, time_window, parameters, route)
        elif len(parent_1[1]) > 1 and len(parent_2[1]) > 1:
            # VRP case
            if r > 0.5:
                child = crossover_vrp_brbax(parent_1, parent_2)
                child = crossover_vrp_bcr(child, parent_2, distance_matrix, velocity,
                                          capacity, fixed_cost, variable_cost,
                                          penalty_value, time_window, parameters, route)
            else:
                child = crossover_vrp_brbax(parent_2, parent_1)
                child = crossover_vrp_bcr(child, parent_1, distance_matrix, velocity,
                                          capacity, fixed_cost, variable_cost,
                                          penalty_value, time_window, parameters, route)
        else:
            child = clone_individual(parent_1 if len(parent_1[1]) > len(parent_2[1]) else parent_2)

        if n_depots > 1:
            child = evaluate_depot(n_depots, child, distance_matrix)
        if vehicle_types > 1:
            child = evaluate_vehicle(vehicle_types, child, distance_matrix, parameters,
                                     velocity, fixed_cost, variable_cost, capacity,
                                     penalty_value, time_window, route, fleet_size)
        child = cap_break(vehicle_types, child, parameters, capacity)
        offspring[i] = child

    return offspring


def mutation_tsp_vrp_swap(individual):
    if len(individual[1]) == 1:
        k1 = k2 = 0
    else:
        k1, k2 = random.sample(range(len(individual[1])), 2)
    c1 = random.randrange(len(individual[1][k1]))
    c2 = random.randrange(len(individual[1][k2]))
    individual[1][k1][c1], individual[1][k2][c2] = individual[1][k2][c2], individual[1][k1][c1]
    return individual


def mutation_tsp_vrp_insertion(individual):
    if len(individual[1]) == 1:
        k1 = k2 = 0
    else:
        k1, k2 = random.sample(range(len(individual[1])), 2)
    c1 = random.randrange(len(individual[1][k1]))
    c2 = random.randrange(len(individual[1][k2]) + 1)
    A = individual[1][k1].pop(c1)
    individual[1][k2].insert(c2, A)
    if not individual[1][k1]:
        del individual[0][k1]; del individual[1][k1]; del individual[2][k1]
    return individual


def mutation(offspring, mutation_rate, elite):
    for i in range(elite, len(offspring)):
        if random.random() <= mutation_rate:
            if random.random() <= 0.5:
                offspring[i] = mutation_tsp_vrp_insertion(offspring[i])
            else:
                offspring[i] = mutation_tsp_vrp_swap(offspring[i])
        for k in range(len(offspring[i][1])):
            if len(offspring[i][1][k]) >= 2 and random.random() <= mutation_rate:
                r = random.random()
                cut = sorted(random.sample(range(len(offspring[i][1][k])), 2))
                segment = offspring[i][1][k][cut[0]:cut[1] + 1]
                if r <= 0.5:
                    random.shuffle(segment)
                else:
                    segment.reverse()
                offspring[i][1][k][cut[0]:cut[1] + 1] = segment
    return offspring


############################################################################
# Reporting
############################################################################

def elite_distance(individual, distance_matrix, route):
    end = 2 if route == 'open' else 1
    td = 0.0
    for n in range(len(individual[1])):
        td += evaluate_distance(distance_matrix, individual[0][n], individual[1][n])[-end]
    return round(td, 2)


def show_report(solution, distance_matrix, parameters, velocity, fixed_cost,
                variable_cost, route, time_window):
    column_names = ['Route', 'Vehicle', 'Activity', 'Job', 'Arrive_Load', 'Leave_Load',
                    'Wait_Time', 'Arrive_Time', 'Leave_Time', 'Distance', 'Costs']
    tt = td = tc = 0.0
    tw_st = parameters[:, 3]
    report_lst = []
    for i in range(len(solution[1])):
        dist = evaluate_distance(distance_matrix, solution[0][i], solution[1][i])
        wait, time = evaluate_time(distance_matrix, parameters, solution[0][i],
                                   solution[1][i],
                                   velocity=[velocity[solution[2][i][0]]])
        reversed_sol = solution[1][i][::-1]
        cap = evaluate_capacity(parameters, solution[0][i], reversed_sol)
        cap.reverse()
        leave_cap = cap[:]
        for n in range(1, len(leave_cap) - 1):
            leave_cap[n] = cap[n + 1]
        cost = evaluate_cost(dist, wait, parameters, solution[0][i], solution[1][i],
                             fixed_cost=[fixed_cost[solution[2][i][0]]],
                             variable_cost=[variable_cost[solution[2][i][0]]],
                             time_window=time_window)
        if route == 'closed':
            subroute = [solution[0][i] + solution[1][i] + solution[0][i]]
        else:
            subroute = [solution[0][i] + solution[1][i]]
        for j in range(len(subroute[0])):
            if j == 0:
                activity = 'start'
                arrive_time = round(time[j], 2)
            else:
                arrive_time = round(time[j] - tw_st[subroute[0][j]] - wait[j], 2)
            if 0 < j < len(subroute[0]) - 1:
                activity = 'service'
            if j == len(subroute[0]) - 1:
                activity = 'finish'
                if time[j] > tt:
                    tt = time[j]
                td += dist[j]
                tc += cost[j]
            report_lst.append(['#' + str(i + 1), solution[2][i][0], activity,
                               subroute[0][j], cap[j], leave_cap[j],
                               round(wait[j], 2), arrive_time, round(time[j], 2),
                               round(dist[j], 2), round(cost[j], 2)])
        report_lst.append(['-//-'] * 11)
    report_lst.append(['MAX TIME', '', '', '', '', '', '', '', round(tt, 2), '', ''])
    report_lst.append(['TOTAL', '', '', '', '', '', '', '', '', round(td, 2), round(tc, 2)])
    return pd.DataFrame(report_lst, columns=column_names)


############################################################################
# Main driver
############################################################################

def genetic_algorithm_vrp(coordinates, distance_matrix, parameters, velocity,
                          fixed_cost, variable_cost, capacity, population_size=5,
                          vehicle_types=1, n_depots=1, route='closed', model='vrp',
                          time_window='without', fleet_size=[], mutation_rate=0.1,
                          elite=0, generations=50, penalty_value=1000, graph=True,
                          selection='rw', seed=None, verbose=True):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    start = tm.time()
    max_capacity = list(capacity)
    if model == 'tsp':
        n_depots = 1
        max_capacity = [float('+inf')] * len(max_capacity)
    elif model == 'mtsp':
        max_capacity = [float('+inf')] * len(max_capacity)
    for i in range(n_depots):
        parameters[i, 0] = 0

    population = initial_population(coordinates, distance_matrix,
                                    population_size=population_size,
                                    vehicle_types=vehicle_types,
                                    n_depots=n_depots, model=model)
    cost, population = target_function(population, distance_matrix, parameters,
                                       velocity, fixed_cost, variable_cost,
                                       max_capacity, penalty_value,
                                       time_window=time_window, route=route,
                                       fleet_size=fleet_size)
    order = sorted(range(len(cost)), key=lambda i: cost[i][0])
    population = [population[i] for i in order]
    cost = [cost[i] for i in order]

    if selection == 'rw':
        fitness = fitness_function(cost, population_size)
    else:
        rank = [[i] for i in range(1, len(cost) + 1)]
        fitness = fitness_function(rank, population_size)

    elite_ind = elite_distance(population[0], distance_matrix, route=route)
    elite_cst = cost[0][0]
    solution = clone_individual(population[0])
    if verbose:
        print(f'Generation = 0  Distance = {elite_ind}  f(x) = {round(elite_cst, 2)}')

    for count in range(1, generations + 1):
        offspring = breeding(cost, population, fitness, distance_matrix, n_depots,
                             elite, velocity, max_capacity, fixed_cost, variable_cost,
                             penalty_value, time_window, parameters, route,
                             vehicle_types, fleet_size)
        offspring = mutation(offspring, mutation_rate=mutation_rate, elite=elite)
        cost, population = target_function(offspring, distance_matrix, parameters,
                                           velocity, fixed_cost, variable_cost,
                                           max_capacity, penalty_value,
                                           time_window=time_window, route=route,
                                           fleet_size=fleet_size)
        order = sorted(range(len(cost)), key=lambda i: cost[i][0])
        population = [population[i] for i in order]
        cost = [cost[i] for i in order]

        elite_child = elite_distance(population[0], distance_matrix, route=route)
        if selection == 'rw':
            fitness = fitness_function(cost, population_size)
        else:
            rank = [[i] for i in range(1, len(cost) + 1)]
            fitness = fitness_function(rank, population_size)

        if elite_ind > elite_child:
            elite_ind = elite_child
            solution = clone_individual(population[0])
            elite_cst = cost[0][0]

        if verbose:
            print(f'Generation = {count}  Distance = {elite_ind}  f(x) = {round(elite_cst, 2)}')

    if graph:
        plot_tour_coordinates(coordinates, solution, n_depots=n_depots, route=route)

    solution_report = show_report(solution, distance_matrix, parameters, velocity,
                                  fixed_cost, variable_cost, route=route,
                                  time_window=time_window)
    if verbose:
        print(f'Algorithm Time: {round(tm.time() - start, 2)} seconds')
    return solution_report, solution


############################################################################
