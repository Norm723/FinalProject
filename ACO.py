import numpy as np
import DataSet
from Node import Node


class ACO:
    def __init__(self, dec_tree, data_set, nant=200, niter=500, rho=0.95, alpha=1, beta=10, seed=None):
        self.tree = dec_tree
        self.data_set = data_set
        self.Nant = nant
        self.Niter = niter
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.pheromone = data_set.getAllThresholds()
        self.local_state = np.random.RandomState(seed)

    def run(self):
        best_tree = None
        best_path = ("TBD", np.inf)
        for i in range(self.Niter):
            contradictions = np.zeros((self.Nant, self.dim))
            all_paths = self.constructColonyPaths(contradictions)
            self.depositPheronomes(all_paths, contradictions)
            best_placement = min(all_paths, key=lambda x: x[1])
            print(i+1, ": ", best_placement[1])
            if best_placement[1] < best_path[1]:
                best_path = best_placement
            self.pheromone *= self.rho  #evaporation
        return best_path

        """
        This method deposits pheromones on the edges.
        Importantly, unlike the lecture's version, this ACO selects only 1/4 of the top tours - and updates only their edges, 
        in a slightly different manner than presented in the lecture.
        """
    def depositPheronomes(self, all_paths, contradictions):
        #sorted_paths = sorted(all_paths, key=lambda x: x[1])
        #Nsel = int(self.Nant/4) # Proportion of updated paths
        currPath = 0
        #for path, fitVal in sorted_paths[:Nsel]:
        for path, fitVal in all_paths:
            for move in range(self.dim):
                self.pheromone[path[move]][move] += 1.0 / (score[currPath][move])**(self.dim/2) #by score
            currPath += 1

        """
        This method generates a single Hamiltonian tour per an ant, starting from node 'start'
        The output, 'path', is a list of edges, each represented by a pair of nodes.
        """
    def constructSolution(self, ant, contradictions):
        path = []
        for i in range(self.dim):
            path = self.nextMove(self.pheromone[:][i], path, ant, contradictions)
        return path, self.evalTour(path, contradictions[ant])
        """
        This method generates 'Nant' paths, for the entire colony, representing a single iteration.
        """
    def constructColonyPaths(self, contradictions):
        all_paths = []
        for i in range(self.Nant):
            path, value = self.constructSolution(i, contradictions)
            #constructing pairs: first is the tour, second is its length
            all_paths.append((path, value))
        return all_paths
    
        """
        This method probabilistically calculates the next move (node) given a neighboring 
        information per a single ant at a specified node.
        Importantly, 'pheromone' is a specific row out of the original matrix, representing the neighbors of the current node.
        Similarly, 'dist' is the row out of the original graph, associated with the neighbors of the current node.
        'visited' is a set of nodes - whose probability weights are constructed as zeros, to eliminate revisits.
        The random generation relies on norm_row, as a vector of probabilities, using the numpy function 'choice'
        """
    def nextMove(self, pheromone, path, ant,contradictions):
        colContr = self.getContradictions(path) #for column k, return pair(num contradictions, vector of with whom contradiction)
        row = pheromone ** self.alpha * ((1.0 / (colContr+1)) ** self.beta)
        norm_row = row / row.sum()
        dims = range(self.dim)
        move = self.local_state.choice(dims, 1, p=norm_row)[0]
        #changes to path and self.contradictions
        path.append(move)
        if colContr[move] != 0:
            contradictions[ant][len(path)-1] += 1
        for j in range((len(path)-1)): #j = column of path[j], path[j] = row of this element
            if path[j] == move:
                contradictions[ant][j] += 1
            if path[j] + j == len(path) - 1 + move:
                contradictions[ant][j] += 1
            if path[j] - j == move - (len(path) - 1):
                contradictions[ant][j] += 1
        return path

