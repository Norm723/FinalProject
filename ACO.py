import numpy as np

from Node import Node


class ACO:
    def __init__(self, dec_tree, nant=200, niter=500, rho=0.95, alpha=1, beta=10, seed=None):
        self.tree = dec_tree
        self.Nant = nant
        self.Niter = niter
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.pheromone = None
        self.local_state = np.random.RandomState(seed)

    def run(self):
        best_tree = None  # need to think what makes a tree to a better one.
        for i in range(self.Niter):
            break
            # self.__buildTree()
            # self.__depositPheromones()
            # self.pheromone *= self.rho  # evaporation


    def __depositPheromones(self):
        pass

