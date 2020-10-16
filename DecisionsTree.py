import numpy as np

from DataSet import DataSet
from Node import Node
import math


def getY(data_set):
    ycol = np.shape(data_set)[1] - 1
    data_set = data_set.transpose()
    return data_set[ycol]


def getClassDivisions(data_set):
    temp = getY(data_set)
    return np.unique(temp, return_counts=True)


def informationGainImpurity(data_set):
    classes, counts = getClassDivisions(data_set)
    total = len(classes)
    score = 0
    for count in counts:
        score += (count / total) * math.log2(count / total)
    return -score


def giniImpurity(data_set):
    classes, counts = getClassDivisions(data_set)
    total = len(classes)
    score = 1
    for count in counts:
        score -= (count / total) ** 2
    return score


def weighted_average(left, right, function):
    left_score = function(left)
    right_score = function(right)
    num_left = np.shape(left)[0]
    num_right = np.shape(right)[0]
    total = num_left + num_right
    weighted_avg = left_score * (num_left / total) + right_score * (num_right / total)
    return weighted_avg, left_score, right_score


def informationGain(left, right):
    return weighted_average(left, right, informationGainImpurity)


def score_by_gini(left, right):
    return weighted_average(left, right, giniImpurity)


def prediction(data_set):
    return np.mean(data_set)


def sumSquared(data_set, predict):
    res = 0
    for point in data_set:
        res += (point - predict) ** 2
    return res


def rss(left, right):
    left = getY(left)
    right = getY(right)
    predLeft = prediction(left)
    predRight = prediction(right)
    rssL = sumSquared(left, predLeft)
    rssR = sumSquared(right, predRight)
    return rssL + rssR, rssL, rssR
    # todo can rss score be considered the node score?


class DecisionsTree:
    def __init__(self, data_set, scoring_func=score_by_gini, max_depth=20, alpha=1, min_data_points=5, min_change=0.1):
        self.root = Node(data_set, 0)
        self.scoring_func = scoring_func
        self.max_depth = max_depth  # max depth to keep splitting until
        self.alpha = alpha  # min best score to split on
        self.min_data_points = min_data_points  # min data points a node can contain
        self.min_change = min_change  # todo figure out or leave out

    def __buildTree(self, node):
        if node.depth >= self.max_depth:
            return

        scores, left, right, sl, sr = self.get_all_scores(node)
        if left is not None:
            node.left = Node(left, node.depth + 1, sl)
            node.right = Node(right, node.depth + 1, sr)
            self.__buildTree(node.leftNode)
            self.__buildTree(node.rightNode)

    def buildTree(self):
        self.__buildTree(self.root)

    # returns the splits on the data set
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

    # returns the score for every possible split on each feature and threshold, and the left and right datasets of
    # the best split
    def get_all_scores(self, node):
        best_score = self.alpha
        num_thresholds = node.data_set.shape[0] - 1
        num_features = node.data_set.shape[1] - 1
        temp_score = np.ones(num_thresholds, num_features)
        temp_left = None
        temp_right = None

        for i in num_features:
            thresholds = node.data_set.getThresholds[i]
            for j in num_thresholds:
                left, right = self.question(i, thresholds[j])
                temp_score[i, j], left_score, right_score = self.scoring_func(left,
                                                                              right)
                if self.scoring_func == informationGain:
                    temp_score[i, j] = node.score - temp_score[i, j]
                if left < self.min_data_points or right < self.min_data_points:
                    temp_score[i, j] = math.inf  # don't want it to be selected
                if temp_score[i, j] < best_score:
                    best_score = temp_score[i, j]
                    temp_left = left
                    temp_right = right
                    temp_lscore = left_score
                    temp_rscore = right_score
                if best_score > self.alpha or best_score >= node.score:
                    temp_left = None
                    temp_right = None

        return temp_score, temp_left, temp_right, temp_lscore, temp_rscore

    def predict(self):
        pass

    def classify(self):
        pass
