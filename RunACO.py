import ACO
import DataSet
import DecisionsTree
import math
import pickle
# from sklearn import datasets, metrics

def predictionAccuracy(predArray, testing_data_set):
    count = 0
    last_column = len(testing_data_set.data[0]) -1
    for index in range(len(predArray)-1):
        if predArray[index] == testing_data_set.data[index][last_column]:
            count += 1
    percent_correct = 100*(count/len(predArray))
    print('accuracy of tree: ', percent_correct)

def main():
    # data_set = DataSet.DataSet("C:\\Users\\shayb\\OneDrive\\Desktop\\testData.csv")
    # ds = DataSet.DataSet('wine_data.csv')
    # rows = ds.data.shape[0]
    # last = ds.data.shape[1] -1
    # trainsize = math.floor(rows*0.66)
    # train = DataSet.DataSet()
    # train.data = ds.data[0:math.floor(rows*0.66)]
    # test = DataSet.DataSet()
    # test.data = ds.data[math.ceil(rows*0.66): rows]
    # tree = DecisionsTree.DecisionsTree(train)
    # ant_colony = ACO.ACO(tree, train, test, 20, 20)
    # tree = ant_colony.run()
    # # results = tree.classifyOrPredict(test)
    # # print(results[:])
    # results = tree.classifyOrPredict(test)
    # predictionAccuracy(results, test)
    # print(results[:])
    # print('')

    # with open('ACOTREE.pickle', 'wb') as handleTREE:
    #     pickle.dump(tree, handleTREE, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('ACOTEST.pickle', 'wb') as handleTEST:
    #     pickle.dump(test, handleTEST, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('ACOTREE.pickle', 'rb') as handleTREE:
        tree = pickle.load(handleTREE)

    with open('ACOTEST.pickle', 'rb') as handleTEST:
        test = pickle.load(handleTEST)

    results = tree.classifyOrPredict(test)
    predictionAccuracy(results, test)
    print(results[:])
    print('')

if __name__ == "__main__":
    main()
