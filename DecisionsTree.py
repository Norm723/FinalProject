import numpy as np
import math
from DataSet import DataSet
from Node import Node


def getYValues(data_set):
    y_values_vec = data_set.data.shape[1] - 1
    data_set = data_set.data.transpose()
    return data_set[y_values_vec]


# returns the divisions, and the number of samples in each division
def getClassDivisions(data_set):
    y_values_vec = getYValues(data_set)
    return np.unique(y_values_vec, return_counts=True)


def entropy(data_set):
    classes, counts = getClassDivisions(data_set)
    num_samples = len(data_set.data)
    score = 0
    for count in counts:
        p = count / num_samples
        score += p * math.log2(p)
    return -score


def giniImpurity(data_set):
    classes, counts = getClassDivisions(data_set)
    total = len(data_set.data)
    score = 1
    for count in counts:
        score -= (count / total) ** 2
    return score


def weighted_average(left, right, function):
    left_score = function(left)
    right_score = function(right)
    num_left = left.data.shape[0]
    num_right = right.data.shape[0]
    total = num_left + num_right
    left_avg = (left_score * (num_left / total))
    right_avg = (right_score * (num_right / total))
    weighted_avg = left_avg + right_avg
    return weighted_avg, left_score, right_score


def informationGain(left, right):
    return weighted_average(left, right, entropy)


def score_by_gini(left, right):
    return weighted_average(left, right, giniImpurity)


def prediction(data_set):
    return np.mean(data_set)


def sumSquared(data_set):
    res = 0
    predict = prediction(data_set)
    for point in data_set:
        res += (point - predict) ** 2
    return res


def rss(left, right):
    left = getYValues(left)
    right = getYValues(right)
    rssL = sumSquared(left)
    rssR = sumSquared(right)
    return rssL + rssR, rssL, rssR
    # todo can rss score be considered the node score?


class DecisionsTree:
    def __init__(self, data_set, scoring_func=score_by_gini, max_depth=20, alpha=1, min_data_pts=5, min_change=0.001):
        self.root = Node(data_set, 0)
        self.scoring_func = scoring_func
        self.max_depth = max_depth  # max depth to keep splitting until
        self.alpha = alpha  # min best score to split on
        self.min_data_points = min_data_pts  # min data points a node can contain
        self.min_change = min_change  # todo figure out or leave out
        self.max = True if self.scoring_func == informationGain else False

    def __buildTree(self, node):
        if node.depth >= self.max_depth:
            return

        scores, left, right, sl, sr = self.get_all_scores(node)
        if left is not None:
            node.leftNode = Node(left, node.depth + 1, sl)
            node.rightNode = Node(right, node.depth + 1, sr)
            self.__buildTree(node.leftNode)
            self.__buildTree(node.rightNode)

    def buildTree(self):
        if self.scoring_func == informationGain:
            self.root.score = entropy(self.root.data_set)
        else:
            self.root.score = math.inf
        self.__buildTree(self.root)

    # returns the splits on the data set
    def question(self, node, feature, threshold):
        # if not isinstance(feature, int):
        #    raise Exception("feature index must be an integer value.")
        if feature < 0 or feature >= node.data_set.data.shape[1]:
            print(feature)
            raise Exception('feature index must be between zero and {dim}'.format(dim=(node.data_set.shape[1] - 1)))
        false_array = []
        true_array = []
        false_data = DataSet()
        true_data = DataSet()
        for row in node.data_set.data:
            if row[feature] < threshold:
                false_array.append(row)
            else:
                true_array.append(row)
        false_data.data = np.array(false_array)
        true_data.data = np.array(true_array)
        if false_data.data.shape[0] < self.min_data_points or true_data.data.shape[0] < self.min_data_points:
            return None, None
        return false_data, true_data

    # returns the score for every possible split on each feature and threshold, and the left and right datasets of
    # the best split
    def get_all_scores(self, node):
        num_features = node.data_set.data.shape[1] - 1
        best_score = self.alpha
        temp_scores = list()
        temp_left = None
        temp_right = None
        haveScore = False

        for i in range(num_features):
            # todo send Node
            thresholds = node.data_set.getThresholds(i)
            num_thresholds = thresholds.shape[0] - 1
            feature_scores = list()
            for j in range(num_thresholds):
                left, right = self.question(node, i, thresholds[j])
                if left is None or right is None:
                    continue
                score, left_score, right_score = self.scoring_func(left, right)
                if self.scoring_func == informationGain:
                    score = node.score - score
                if left.data.shape[0] < self.min_data_points or right.data.shape[0] < self.min_data_points:
                    score = -math.inf if self.max else math.inf   # don't want it to be selected

                if not self.max and score < best_score or self.max and score > best_score:
                    best_score = score
                    temp_left = left
                    temp_right = right
                    temp_lscore = left_score
                    temp_rscore = right_score
                    temp_thresh = thresholds[j]
                    temp_feat = i
                if score != 0:
                    haveScore = True
                feature_scores.append(score)
            temp_scores.append(feature_scores)
        if (not self.max and best_score > self.alpha) or (not self.max and best_score >= node.score) or (self.max and best_score <= self.alpha) or (self.max and best_score <= self.min_change):
            temp_left = None
            temp_right = None
            temp_lscore = 1
            temp_rscore = 1
            temp_thresh = 0
            temp_feat = 0
        node.threshold = temp_thresh
        node.feature = temp_feat
        if not haveScore:
            temp_scores = None
        return temp_scores, temp_left, temp_right, temp_lscore, temp_rscore

    def __getPrediction(self, row, node):
        if node.IsLeaf():
            return node.prediction()

        if row[node.feature] < node.threshold:
            prediction = self.__getPrediction(row, node.leftNode)
        else:
            prediction = self.__getPrediction(row, node.rightNode)
        return prediction

    def __getClassification(self, row, node):
        if node.IsLeaf():
            return node.classify()

        if row[node.feature] < node.threshold:
            classification = self.__getClassification(row, node.leftNode)
        else:
            classification = self.__getClassification(row, node.rightNode)
        return classification

    #get predictions for all rows in the dataset
    def predict(self, dataset):
        # todo for every datapoint in set move left right until reach leaf and then prediction value is mean of
        #  dataset in the leaf
        numRows = dataset.data.shape[0]
        #creat an array of length of number of rows of data to store the prediction
        predictions = np.zeros(numRows)
        for row in range(numRows):
            predictions[row] = self.__getPrediction(dataset.data[row], self.root) 
        return predictions

    def classify(self, dataset):
        # todo for every datapoint in set move left right until reach leaf and then prediction value is mode of
        #  dataset in the leaf
        numRows = dataset.data.shape[0]
        #creat an array of length of number of rows of data to store the prediction
        classifications = np.zeros(numRows)
        for row in range(numRows):
           classifications[row] = self.__getClassification(dataset.data[row], self.root) 
        return classifications

    def classifyOrPredict(self, dataset):
        if self.scoring_func == rss:
            return self.predict(dataset)
        else:
            return self.classify(dataset)

    def printTree(self):
        self.root.Print()
