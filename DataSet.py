import pandas as pd
import numpy as np


def getMeans(temp):
    length = (len(temp)-1)
    temp_avg = np.zeros(length)
    for i in length:
        temp_avg[i] = np.mean(temp[i:i+1])
    return temp_avg


class DataSet:
    def __init__(self, file_path=None):
        if file_path is not None:
            # check if path is good
            self.data = pd.read_csv(file_path)
            self.data = self.data.to_numpy()
        else:
            exit

    def getThresholds(self, index):
        temp = np.apply_over_axes(np.sort, axes=0, a=self.data)
        temp = temp.transpose()
        temp = temp[index]
        temp = np.unique(temp)
        return getMeans(temp)

    def getAllThresholds(self):
        len = self.data.shape[1] - 1
        threshes = list()
        for i in range(len):
            threshes.append(self.getThresholds(i))
        return threshes

    def printData(self):
        print(self.data)
