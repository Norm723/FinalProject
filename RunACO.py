import ACO
import DataSet
import DecisionsTree
import pickle
import RandomForest
# from sklearn import datasets, metrics

def predictionAccuracy(predArray, testing_data_set, typeOfRun):
    count = 0
    last_column = len(testing_data_set.data[0]) -1
    for index in range(len(predArray)):
        if predArray[index] == testing_data_set.data[index][last_column]:
            count += 1
    percent_correct = 100*(count/len(predArray))
    print(typeOfRun, 'accuracy of tree: ', percent_correct)

def main():
    ds = DataSet.DataSet('wine_data.csv')
    train, test = ds.splitIntoTrainingTest()
    
    # for aco
    tree = DecisionsTree.DecisionsTree(train)
    ant_colony = ACO.ACO(tree, train, test, 20, 20)
    tree = ant_colony.run()

    with open('ACOTREE.pickle', 'wb') as handleTREE:
        pickle.dump(tree, handleTREE, protocol=pickle.HIGHEST_PROTOCOL)

    with open('ACOTEST.pickle', 'wb') as handleTEST:
        pickle.dump(test, handleTEST, protocol=pickle.HIGHEST_PROTOCOL)
    
    # with open('ACOTREE.pickle', 'rb') as handleTREE:
    #     tree = pickle.load(handleTREE)

    # with open('ACOTEST.pickle', 'rb') as handleTEST:
    #     test = pickle.load(handleTEST)

    results = tree.classifyOrPredict(test)
    predictionAccuracy(results, test, 'ACO')
    print(results[:])
    print('')

    # for regular 
    train = DataSet.DataSet('satTest.csv')
    test = DataSet.DataSet('satTrain.csv')
    tree = DecisionsTree.DecisionsTree(train)
    tree.buildTree()
    temp = tree.classify(test)
    predictionAccuracy(temp, test, 'Regular')
    print(temp[:])
    print('')

    # for random forest
    rf = RandomForest.RandomForest('wine_data.csv', 20)
    rf.buildTrees()
    temp = rf.classify(test)
    predictionAccuracy(temp, test, 'random forest')
    print(temp[:])
    print('')

if __name__ == "__main__":
    main()
