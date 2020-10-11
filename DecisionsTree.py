import numpy as np

from DataSet import DataSet
from Node import Node


class DecisionsTree:
    def __init__(self, data_set, scoring_func, max_depth=20, alpha=1, min_data_points=5):
        self.root = Node(np.random.randint(0, data_set.data.shape[0] - 1), np.random.randint(0, 10), data_set)
        self.scoring_func = scoring_func
        self.max_depth = max_depth
        self.alpha = alpha
        self.min_data_points = min_data_points

    def __buildTree(self):
        i = 0
        leafs = [self.tree_root]
        while i < self.max_depth:
            new_leafs = leafs
            leafs = []
            for leaf in new_leafs:
                if leaf.data_set.data is not None:
                    f, t = leaf.data_set.question(leaf.feature, leaf.threshold)
                    if not t.data.any() or not f.data.any():
                        continue
                    if t.data.any():
                        feature = np.random.randint(0, t.data.shape[1] - 1)
                        threshold = np.random.randint(0, 10)
                        right_node = Node(feature, threshold, t)
                        leaf.SetRightNode(right_node)
                        leafs.append(right_node)
                    if f.data.any():
                        feature = np.random.randint(0, f.data.shape[1] - 1)
                        threshold = np.random.randint(0, 10)
                        left_node = Node(feature, threshold, f)
                        leaf.SetLeftNode(left_node)
                        leafs.append(left_node)
            i += 1
        self.tree_root.Print()

    def question(self, feature, threshold):
        if not isinstance(feature, int):
            raise Exception("feature index must be an integer value.")
        if feature < 0 or feature >= self.data.shape[1]:
            raise Exception('feature index must be between zero and {dim}'.format(dim=(self.data.shape[1] - 1)))
        false_array = []
        true_array =[]
        false_data = DataSet()
        true_data = DataSet()
        for row in self.data:
            if row[feature] < threshold:
                true_array.append(row)
            else:
                false_array.append(row)
        false_data.data = np.array(false_array)
        true_data.data = np.array(true_array)
        false_data.__sortByColumns()
        true_data.__sortByColumns()
        return false_data, true_data
