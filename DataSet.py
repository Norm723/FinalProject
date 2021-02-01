import pandas as pd
import numpy as np


def getMeans(row_vector):
    length = (len(row_vector) - 1)
    means = np.zeros(length)
    for i in range(length):
        means[i] = np.mean(row_vector[i:i + 2])
    return means


class DataSet:
    def __init__(self, file_path=None):
        if file_path is not None:
            # check if path is good
            # automatically removes first row
            self.data = pd.read_csv(file_path)
            self.data = self.data.to_numpy()
            # randomly change order of rows
            self.data = np.random.permutation(self.data)
            if self.data.dtype == object:
                self.data = self.data.astype('float64')

    def getThresholds(self, feature):
        temp_data = self.data
        thresholds = list()
        for i in range(len(temp_data)):
            thresholds.append(temp_data[i][feature])
        thresholds = np.unique(thresholds)
        thresholds = np.sort(thresholds)
        return getMeans(thresholds)

    # returns all possible thresholds as a list of tuples with (value, 1)
    # 1 is starting pheromone value for ACO
    def getAllThresholds(self):
        num_features = self.data.shape[1] - 1
        threshes = list()
        for i in range(num_features):
            feature_thresholds = self.getThresholds(i)
            threshes_pher_for_feature = list()
            for j in range(feature_thresholds.size):
                thresholds_pheromones = list()
                thresholds_pheromones.append(feature_thresholds[j])
                thresholds_pheromones.append(1)
                threshes_pher_for_feature.append(thresholds_pheromones)
            threshes.append(threshes_pher_for_feature)
        return threshes

    def printData(self):
        print(self.data)
