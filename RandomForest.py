import numpy as np
import DataSet
import DecisionsTree

class RandomForest:
    def __init__(self, data_file, num_trees=1, scoring_func=DecisionsTree.score_by_gini, max_depth=20, alpha=1, min_data_pts=5, min_change=0.001):
        self.data_file = data_file
        self.num_trees = num_trees
        self.trees = list()
        self.scoring_func = scoring_func
        self.max_depth = max_depth  # max depth to keep splitting until
        self.alpha = alpha  # min best score to split on
        self.min_data_points = min_data_pts  # min data points a node can contain
        self.min_change = min_change  # todo figure out or leave out
    
    # function for building n trees and storing them in a list
    def buildTrees(self):
        for i in range(self.num_trees):
            ds = DataSet.DataSet(self.data_file) 
            train, test = ds.splitIntoTrainingTest()
            tree = DecisionsTree.DecisionsTree(train, self.scoring_func, self.max_depth, self.alpha, self.min_data_points, self.min_change)
            tree.buildTree()
            self.trees.append(tree)

    # get mode of classifications for data point
    def __getClassification(self, predList, row):
        data_list = list()
        for i in range(self.num_trees): #self.num_trees is equal to the amount of predictions
            data_list.append(predList[i][row])
        return max(set(data_list), key=data_list.count)

    # classify function based on majority rule
    def classify(self, dataset):
        predList = list()
        for i in range(self.num_trees):
            pred = self.trees[i].classifyOrPredict(dataset)
            predList.append(pred)
        numRows = dataset.data.shape[0]
        predictions = np.zeros(numRows)
        for row in range(numRows):
            predictions[row] = self.__getClassification(predList, row) 
        return predictions


