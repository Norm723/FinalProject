import pandas as pd
import numpy as np
import os
import DataSet


# giniImpurity(temp)

# temp = np.array([2,33,4,5,6,7,7,8,9,10])
# lengthTemp = len(temp)
# tempTouples = list()
# for j in range(lengthTemp):
#     tempTouples.append((temp[j], 0))

# print(tempTouples[2][0])

# temp = pd.read_csv("C:\\Users\\normt\\Desktop\\semester6\\bigData\\project\\Batting.csv")
# temp = temp.to_numpy()
# temp2 = list()
# for i in range(len(temp)):
#     temp2.append(temp[i][6])
# temp2 = np.unique(temp2)
# temp2 = np.sort(temp2)
# temp2 = DataSet.getMeans(temp2)
# print(temp2)

ds = DataSet.DataSet("C:\\Users\\normt\\Desktop\\testData.csv")
f = ds.getAllThresholds()
print(f[0][0][0])
