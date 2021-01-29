import numpy as np
import DataSet
from Node import Node
import DecisionsTree


class ACO:
    def __init__(self, dec_tree, data_set, nant=200, niter=500, rho=0.95, alpha=1, beta=1, seed=None):
        self.tree = dec_tree
        self.data_set = data_set
        self.Nant = nant
        self.Niter = niter
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.pheromone = data_set.getAllThresholds()
        self.local_state = np.random.RandomState(seed)

    def run(self):  # TODO: implement best tree function
        best_tree = None
        for i in range(self.Niter):
            trees = self.constructTrees()
            self.depositPheromones(trees)
            temp_best_tree = get_best_tree(trees)
            if best_tree < temp_best_tree:
                best_tree = temp_best_tree
            self.pheromone *= self.rho  # evaporation
        return best_tree

    def depositPheromones(self, trees):

        for tree in trees:
            for node in tree:
                #  TODO: implement iterative function there goes over all nodes in tree
                #  pseudo code
                self.pheromone[node.feature][node.threshold][1] += node.score
            #  self.pheromone[path[move]][move] += 1.0 / (score[currPath][move])**(self.dim/2) #by score

    def constructSolution(self):
        temp_tree = self.tree
        current_edges = list()
        current_edges.append(self.tree.root)
        for i in range(self.tree.max_depth):
            new_edges = list()
            for edgeNode in current_edges:
                temp_score = temp_tree.get_all_scores(edgeNode)[0]
                left, right = self.nextMove(edgeNode, temp_score)
                if left is not None and right is not None:
                    new_edges.append(left)
                    new_edges.append(right)
            current_edges = new_edges
        return temp_tree

    def constructTrees(self):
        all_trees = list()
        for i in range(self.Nant):
            tree = self.constructSolution()
            all_trees.append(tree)
        return all_trees

    def getMaxThresholdsLength(self):
        thresholds_length = list()
        for thresholds in range(len(self.pheromone)):
            thresholds_length.append(len(self.pheromone[thresholds]))
        return max(thresholds_length)

    def nextMove(self, node, scores):
        temp_pheromones = np.zeros(shape=(len(self.pheromone.shape), self.getMaxThresholdsLength()))
        threshold = node.data_set.getAllThresholds()

        for feature in range(len(scores)):
            for threshold in range(len(scores[feature])):
                temp_pheromones[feature][threshold] = (self.pheromone[feature][threshold][1] ** self.alpha) *\
                                                      (scores[feature][threshold] ** self.beta)

        linear_idx = np.random.choice(temp_pheromones.size, p=temp_pheromones.ravel() / float(temp_pheromones.sum()))

        feature_idx, threshold_idx = np.unravel_index(linear_idx, temp_pheromones.shape)
        # unravel/ravel doesn't work as expected
        #temp_pheromones /= temp_pheromones.sum()
        #feature, threshold = self.local_state.choice(shape=(temp_pheromones.shape[0], temp_pheromones.shape[1]), 1, p=temp_pheromones)[0]
        threshold_value = threshold[feature_idx][threshold_idx]
        left, right = self.tree.question(node, feature_idx.astype(int), threshold_value)
        if left is None or right is None:
            print("split got none")
        node.score = scores[feature][threshold]
        node.feature = feature
        node.threshold = threshold
        if left is None or right is None:
            return None, None
        node.leftNode = Node(left, node.depth + 1, 0)
        node.rightNode = Node(right, node.depth + 1, 0)
        return node.leftNode, node.rightNode

