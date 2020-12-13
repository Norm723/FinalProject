import pandas as pd
import numpy as np


def getMeans(temp):
    length = (len(temp)-1)
    temp_avg = np.zeros(length)
    for i in range(length):
        temp_avg[i] = np.mean(temp[i:i+2])
    return temp_avg


class DataSet:
    def __init__(self, file_path=None):
        if file_path is not None:
            # check if path is good
            # automatically removes first row
            self.data = pd.read_csv(file_path)
            self.data = self.data.to_numpy()
            self.data = np.random.permutation(self.data)
        else:
            exit

    def getThresholds(self, index):
        temp = self.data
        temp2 = list()
        for i in range(len(temp)):
            temp2.append(temp[i][index])
        temp2 = np.unique(temp2)
        temp2 = np.sort(temp2)
        return getMeans(temp2)

    # returns all possible thresholds as a list of tuples with (value, 0)
    # 0 is starting pheromone value for ACO
    def getAllThresholds(self):
        len = self.data.shape[1] - 1
        threshes = list()
        for i in range(len):
            temp = self.getThresholds(i)
            tempTouples = list()
            for j in range(temp.size):
                tempTouples.append((temp[j], 0))
            threshes.append(tempTouples)
        return threshes

    def printData(self):
        print(self.data)

    def getData(self):
        return self.data
