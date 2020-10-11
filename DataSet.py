import pandas as pd
import numpy as np


class DataSet:
    def __init__(self, file_path=None):
        if file_path is not None:
            # check if path is good
            self.data = pd.read_csv(file_path)
            self.data = self.data.to_numpy()
        else:
            self.data = None

    def __getThresholds(self, index):
        pass
        # TODO: implement
        # temp_data = self.data.transpose()
        # temp_data = np.sort(temp_data[self.data.shape[0] - index])
        # check sort
        # calculate means
        # return unique(vec)
        # self.data = np.apply_over_axes(np.sort, axes=0, a=self.data)
        # [1 3 4 5] sorted column
        # [2 3.5 4.5] potential thresholds
        # build temp right & left node and check splitting score


    def __getMeans(self):
        pass
        # TODO: implement

    def printData(self):
        print(self.data)

    def getRandomFeature(self):
        return 0
        # return random index
