import ACO
import DataSet
import DecisionsTree
import math
from sklearn import datasets, metrics


def main():
    #  data_set = DataSet.DataSet("C:\\Users\\shayb\\OneDrive\\Desktop\\testData.csv")
    ds = DataSet.DataSet('wine_data.csv')
    rows = ds.data.shape[0]
    last = ds.data.shape[1] -1
    trainsize = math.floor(rows*0.66)
    train = DataSet.DataSet()
    train.data = ds.data[0:math.floor(rows*0.66)]
    test = DataSet.DataSet()
    test.data = ds.data[math.ceil(rows*0.66): rows]
    tree = DecisionsTree.DecisionsTree(train)
    ant_colony = ACO.ACO(tree, train, test, 3, 10)
    tree = ant_colony.run()
    results = tree.classifyOrPredict(test)
    print(results[:])
    results = tree.classifyOrPredict(train)
    print(results[:])
    print('')

if __name__ == "__main__":
    main()
