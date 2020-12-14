import pandas as pd
import numpy as np
import os
import DataSet
import DecisionsTree
from sklearn import datasets
import sklearn.model_selection as model_selection
import math


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

# iris = datasets.load_iris()
# ds = DataSet.DataSet('C:\\Users\\normt\\AppData\\Local\\Programs\\Python\\Python37-32\\lib\\site-packages\\sklearn\\datasets\\data\\boston_house_prices.csv')
ds = DataSet.DataSet('C:\\Users\\normt\\AppData\\Local\\Programs\\Python\\Python37-32\\lib\\site-packages\\sklearn\\datasets\\data\\iris.csv')
f = ds.getAllThresholds()
# [row/vector][pair in vector][0 = thresh, 1 = pheromones]
# print(f[0][33][0])
# print(ds.data[:, 4])
rows = ds.data.shape[0]
trainsize = math.floor(rows*0.66)
train = DataSet.DataSet()
train.data = ds.data[0:math.floor(rows*0.66)]
test = DataSet.DataSet()
test.data = ds.data[math.ceil(rows*0.66): rows]
# train, test, ty1,ty2 = model_selection.train_test_split(ds.data, test_size = 0.4, stratify = ds.data[4], random_state = 42)
tree = DecisionsTree.DecisionsTree(train, scoring_func=DecisionsTree.informationGain, alpha=0)
tree.buildTree()
# tree.printTree()
temp = tree.classify(test)
print(temp)
print(temp[:] != ds.data[math.ceil(rows*0.66): rows, 4])
print(np.sum(temp[:] != ds.data[math.ceil(rows*0.66): rows, 4]))
print((1-np.sum(temp[:] != ds.data[math.ceil(rows*0.66): rows, 4])/(rows-math.floor(rows*0.66)))*100)
