import ACO
import DataSet
import DecisionsTree
import math


def main():
    #  data_set = DataSet.DataSet("C:\\Users\\shayb\\OneDrive\\Desktop\\testData.csv")
    ds = DataSet.DataSet('iris.csv')
    rows = ds.data.shape[0]
    last = ds.data.shape[1] -1
    trainsize = math.floor(rows*0.66)
    train = DataSet.DataSet()
    train.data = ds.data[0:math.floor(rows*0.66)]
    test = DataSet.DataSet()
    test.data = ds.data[math.ceil(rows*0.66): rows]
    tree = DecisionsTree.DecisionsTree(train)
    ant_colony = ACO.ACO(tree, train, test, 15, 10)
    tree = ant_colony.run()


if __name__ == "__main__":
    main()
