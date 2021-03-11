import numpy as np
from Node import Node
import DecisionsTree
from multiprocessing import Process, Queue, Manager
import os
import threading

class ACO:
    def __init__(self, dec_tree, data_set, testing_data_set, nant=200, niter=500, rho=0.95, alpha=3, beta=1, seed=None):
        self.tree = dec_tree
        self.data_set = data_set
        self.testing_data_set = testing_data_set
        self.Nant = nant
        self.Niter = niter
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.pheromone = data_set.getAllThresholds()
        self.local_state = np.random.RandomState(seed)
        self.best_tree = None
        self.best_tree_score = 0

    def run(self):
        for i in range(self.Niter):
            print("iter: ", i)
            trees = self.constructTrees()
            temp_tree, temp_tree_score, tree_scores = self.getBestTree(trees)
            self.depositPheromones(tree_scores)
            if self.best_tree_score < temp_tree_score:
                self.best_tree = temp_tree
                self.best_tree_score = temp_tree_score
            self.evaporation()
            print(self.best_tree_score)
        return self.best_tree
    
    def getBestTree(self, trees):
        tree_scores = list()
        best_tree = None
        best_result = 0
        last_column = len(self.testing_data_set.data[0]) -1
        # for parallel
        # q = Queue()
        mgr = Manager()
        q = mgr.list()
        processes = [Process(target=tree.classifyParallel, args=(self.testing_data_set, tree, q)) for tree in trees]

        for p in processes:
            p.start()

        # results = [q.get() for _ in processes]

        for p in processes:
            p.join()
        
        for classify_results in q:
            tree = classify_results[0]
            percent_correct = classify_results[2]
            tree_scores.append((tree, percent_correct))
            if percent_correct > best_result:
                best_result = percent_correct
                best_tree = tree
        return best_tree, best_result, tree_scores
        # for non-parallel
        # for tree in trees:
        #     count = 0
        #     result = tree.classifyOrPredict(self.testing_data_set)
        #     for index in range(len(result)):
        #         if result[index] == self.testing_data_set.data[index][last_column]:
        #             count += 1
        #     percent_correct = 100*(count/len(result)) 
        #     print(percent_correct)
        #     if percent_correct > best_result:
        #         best_result = percent_correct
        #         best_tree = tree
        # return best_tree, best_result

    def evaporation(self):
        for feature in range(len(self.pheromone)):
            for threshold in range(len(self.pheromone[feature])):
                self.pheromone[feature][threshold][1] *= self.rho  # evaporation

    def depositPheromones(self, trees_and_scores):
        processes = [Process(target=self._depositPheromones, args=(tree_and_score[0].root, tree_and_score[1],)) for tree_and_score in trees_and_scores]

        for p in processes:
            p.start()

        for p in processes:
            p.join()
        
        # non parallel
        # for tree in trees:
        #     self._depositPheromones(tree.root)

    def _depositPheromones(self, node, tree_score):
        score = tree_score if self.tree.max else 1/(tree_score+1)
        feature = node.feature
        sem = threading.Semaphore()
        for index in range(len(self.pheromone[feature])):
            sem.acquire()
            if self.pheromone[feature][index][0] == node.threshold:
                self.pheromone[feature][index][1] += score
                break
            sem.release()
        if node.leftNode:
            self._depositPheromones(node.leftNode, tree_score)
            self._depositPheromones(node.rightNode, tree_score)

    # FOR PARRELEL 
    def constructSolution(self, q):
    # FOR NON-PARALLEL
    # def constructSolution(self):
        temp_tree = DecisionsTree.DecisionsTree(self.data_set, self.tree.scoring_func, self.tree.max_depth, self.tree.alpha, self.tree.min_data_points, self.tree.min_change)
        current_edges = list()
        current_edges.append(temp_tree.root)
        for _ in range(temp_tree.max_depth):
            new_edges = list()
            for edgeNode in current_edges:
                temp_score = temp_tree.get_all_scores(edgeNode)[0]
                left, right = self.nextMove(edgeNode, temp_score)
                if left is not None and right is not None:
                    new_edges.append(left)
                    new_edges.append(right)
            current_edges = new_edges
        # FOR PARRELEL    
        q.append(temp_tree)
        # FOR NON-PARALLEL
        # return temp_tree

    def constructTrees(self):
        # FOR PARRELEL
        # q = Queue()
        mgr = Manager()
        q = mgr.list()
        processes = [Process(target=self.constructSolution, args=(q,)) for _ in range(self.Nant)]

        for p in processes:
            p.start()

        # all_trees = [q.get() for _ in processes]

        for p in processes:
            p.join()
        
        return q
        # FOR NON-PARALLEL
        # all_trees = list()
        # for _ in range(self.Nant):
        #     tree = self.constructSolution()
        #     all_trees.append(tree)
        # return all_trees

    def getMaxThresholdsLength(self):
        thresholds_length = list()
        for thresholds in range(len(self.pheromone)):
            thresholds_length.append(len(self.pheromone[thresholds]))
        return max(thresholds_length)

    def nextMove(self, node, scores):
        if scores is None:
            return None, None
        temp_pheromones = np.zeros(shape=(len(self.pheromone), self.getMaxThresholdsLength()))
        thresholds = node.data_set.getAllThresholds()
        for feature in range(len(thresholds)):
            for thresh in range(len(thresholds[feature])):
                for pheromone_thresh in range(len(self.pheromone[feature])):
                    if thresholds[feature][thresh][0] == self.pheromone[feature][pheromone_thresh][0]:
                        thresholds[feature][thresh][1] = self.pheromone[feature][pheromone_thresh][1] 

        for feature in range(len(scores)):
            for threshold in range(len(scores[feature])):
                temp_pheromones[feature][threshold] = (thresholds[feature][threshold][1] ** self.alpha) *\
                                                      (scores[feature][threshold] ** self.beta)

        np.random.seed(os.getpid())
        linear_idx = np.random.choice(temp_pheromones.size, p=temp_pheromones.ravel() / float(temp_pheromones.sum()))

        feature_idx, threshold_idx = np.unravel_index(linear_idx, temp_pheromones.shape)
        # unravel/ravel doesn't work as expected
        #temp_pheromones /= temp_pheromones.sum()
        #feature, threshold = self.local_state.choice(shape=(temp_pheromones.shape[0], temp_pheromones.shape[1]), 1, p=temp_pheromones)[0]
        threshold_value = thresholds[feature_idx][threshold_idx][0]
        left, right = self.tree.question(node, feature_idx.astype(int), threshold_value)

        node.score = scores[feature_idx][threshold_idx]
        node.feature = feature_idx
        node.threshold = threshold_value
        if left is None or right is None:
            return None, None
        node.leftNode = Node(left, node.depth + 1, 0)
        node.rightNode = Node(right, node.depth + 1, 0)
        return node.leftNode, node.rightNode

