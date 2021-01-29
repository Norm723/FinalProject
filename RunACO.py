import ACO
import DataSet
import DecisionsTree


def main():
    #  data_set = DataSet.DataSet("C:\\Users\\shayb\\OneDrive\\Desktop\\testData.csv")
    ds = DataSet.DataSet('iris.csv')
    tree = DecisionsTree.DecisionsTree(ds)
    ant_colony = ACO.ACO(tree, ds)
    tree = ant_colony.run()


if __name__ == "__main__":
    main()
