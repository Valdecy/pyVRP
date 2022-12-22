# pyVRP
The pyVRP is python library that solves (using Genetic Algorithms): Capacitated VRP, Multiple Depot VRP, VRP with Time Windows, VRP with Homogeneous or Heterogeneous Fleet, VRP with Finite or Infinite Fleet, Open or Closed Routes, TSP, mTSP and various combination of these types.

Try it in **Colab**: 

- Capacitated VRP ([ Colab Demo ](https://colab.research.google.com/drive/1FUIa98I3uBVHpSNLfkFGdS_5WH1-0NTH?usp=sharing))
- Multiple Depot VRP ([ Colab Demo ](https://colab.research.google.com/drive/1QYm1g6Zy58cTvR1fWX571nhDDcgusE90?usp=sharing))
- VRP with Time Windows ([ Colab Demo ](https://colab.research.google.com/drive/1T6fByfiZ-q_3D6_DQYhS6bxPq9eRqDlf?usp=sharing))
- VRP with Heterogeneous Fleet ([ Colab Demo ](https://colab.research.google.com/drive/11d2ESpjdT9u8mLnDpsE1xwL16IHoVmFh?usp=sharing))
- VRP with Infinite Fleet ([ Colab Demo ](https://colab.research.google.com/drive/1tIN9uK7jEK1uyHzq7xxjmNS1Jm87t5eQ?usp=sharing))
- Open VRP ([ Colab Demo ](https://colab.research.google.com/drive/1KSlhRskcRjs5483nBdrxcjEoCwW-4Ns0?usp=sharing))
- TSP ([ Colab Demo ](https://colab.research.google.com/drive/1WzRKa7aUUe-hV9RFbQwQdPLd0VYNNPIu?usp=sharing))
- mTSP ([ Colab Demo ](https://colab.research.google.com/drive/1fLCSwpxLi62NJ5ru6uZLOdyBQ-L29piV?usp=sharing))

# Parameters Description

A) **Input**: As demonstrated in the Colab examples along with the coordinates (or distance matrix) for each client, a table with the following information for each one is needed: "Demand" (the demand for each client, the deposit has a demand = 0),  "TW_early" (if the client has a time window, this parameter indicates the earliest time that a vehicle can arrive. Only relevant for problems with Time Window), "TW_late" (if the client has a time window, this parameter indicates the latest time that a vehicle can arrive. Only relevant for problems with Time Window), "TW_service_time" (arriving at client how much time is needed to the vehicle unload the goods), "TW_wait_cost" (cost/time or penalty/time for a vehicle that spends time waiting to a client to be available. Only relevant for problems with Time Window )

B) **Model**: "n_depots" (an integer that indicates that the first n_depots rows of the distance_matrix or coordinates will be considered as depots), "time_window" ('with': indicates VRP problems with Time Windows, 'without': indicates VRP problems without Time Windows), "route"('open': open routes meaning that the vehicle does not need to return to a depot, 'closed': closed routes meaning that the vehicle needs to return to a depot), "model" ('tsp': Traveling Salesman Problem, 'mtsp': Multiple Traveling Salesman Problem, 'vrp': Vehicle Routing Problem), "graph"(True: the solution will be plotted at the end of the iterations, False: the solution will not be plotted at the end of the iterations)

C) **Vehicle**: "vehicle_types" (an integer that indicates the different types of vehicles that are available), "fixed_cost" (a list that indicates for each type of vehicle the cost spent for taking that vehicle), "variable_cost" (a list that indicates for each type of vehicle the cost spent for travelled distance), "capacity" (a list that indicates for each type of vehicle the total capacity), "velocity" (a list that indicates for each type of vehicle the average velocity. this value is used as a constant that divides the distance matrix), "fleet_size" (a list that indicates the quantity of each type of vehicle. an empty list means that the fleet is infinite)

D) **Genetic Algorithm**: "penalty_value" (penalty value for violating the problem constraints), "population_size" (an integer that represents the number of individuals in a population), "mutation_rate" (a continuous value between 0 (0%) and 1 (100%) that indicates the probability of mutation), "elite" (an integer representing the number of best individuals preserved from a generation to another), "generations" (an integer indicating the total number of iterations)

# TSP (Travelling Salesman Problem)
For Specialized Travelling Salesman Problems Algorithms try [pyCombinatorial](https://github.com/Valdecy/pyCombinatorial)

# Acknowledgement 
This section is dedicated to all the people that helped to improve or correct the code. Thank you very much!

* Estela Perez da Cruz Ulhoa Tenorio (10.AUGUST.2021) - Federal Fluminense University.
