import numpy as np

from DataSet import DataSet
from Node import Node


class DecisionsTree:
    def __init__(self, data_set, scoring_func, max_depth=20, alpha=1, min_data_points=5):
        self.root = Node(data_set, 0)
        self.scoring_func = scoring_func
        self.max_depth = max_depth
        self.alpha = alpha
        self.min_data_points = min_data_points

    def __buildTree(self, node):
        if node.depth >= self.max_depth:
            return

        scores, left, right = self.get_all_scores(node)
        if left is not None:
            node.left = Node(left, node.depth + 1)
            node.right = Node(right, node.depth + 1)
            self.__buildTree(node.leftNode)
            self.__buildTree(node.rightNode)
        # i = 0
        # leafs = [self.tree_root]
        # while i < self.max_depth:
        #     new_leafs = leafs
        #     leafs = []
        #     for leaf in new_leafs:
        #         if leaf.data_set.data is not None:
        #             f, t = leaf.data_set.question(leaf.feature, leaf.threshold)
        #             if not t.data.any() or not f.data.any():
        #                 continue
        #             if t.data.any():
        #                 feature = np.random.randint(0, t.data.shape[1] - 1)
        #                 threshold = np.random.randint(0, 10)
        #                 right_node = Node(feature, threshold, t)
        #                 leaf.SetRightNode(right_node)
        #                 leafs.append(right_node)
        #             if f.data.any():
        #                 feature = np.random.randint(0, f.data.shape[1] - 1)
        #                 threshold = np.random.randint(0, 10)
        #                 left_node = Node(feature, threshold, f)
        #                 leaf.SetLeftNode(left_node)
        #                 leafs.append(left_node)
        #     i += 1
        # self.tree_root.Print()

    def question(self, feature, threshold):
        if not isinstance(feature, int):
            raise Exception("feature index must be an integer value.")
        if feature < 0 or feature >= self.data.shape[1]:
            raise Exception('feature index must be between zero and {dim}'.format(dim=(self.data.shape[1] - 1)))
        false_array = []
        true_array = []
        false_data = DataSet()
        true_data = DataSet()
        for row in self.data:
            if row[feature] < threshold:
                true_array.append(row)
            else:
                false_array.append(row)
        if false_data.data.shape[0] < self.min_data_points or true_data.data.shape[0] < self.min_data_points:
            return None, None
        false_data.data = np.array(false_array)
        true_data.data = np.array(true_array)
        return false_data, true_data

    def get_all_scores(self, node):
        best_score = self.alpha
        num_thresholds = node.data_set.shape[0]-1
        num_features = node.data_set.shape[1]-1
        temp_score = np.ones(num_thresholds, num_features)
        temp_left = None
        temp_right = None

        for i in num_features:
            thresholds = node.data_set.getThresholds[i]
            for j in num_thresholds:
                left, right = self.question(i, thresholds[j])
                temp_score[i, j] = self.scoring_func(left, right)  # todo write at least one scoring_func: entropy, gini
                if temp_score[i, j] < best_score:
                    best_score = temp_score[i, j]
                    temp_left = left
                    temp_right = right

        return temp_score, temp_left, temp_right

    def predict(self):
        pass

    def classify(self):
        pass
