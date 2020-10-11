from ACO import ACO
from DataSet import DataSet
from DecisionsTree import DecisionsTree
from io import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz, DecisionTreeClassifier
import pydotplus



def main():
    # clf = DecisionTreeClassifier()
    # dot_data = StringIO()
    # export_graphviz(clf, out_file=dot_data,
    #                 filled=True, rounded=True,
    #                 special_characters=True, feature_names=["A", "B", "C"], class_names=['0', '1'])
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_png('diabetes.png')
    # Image(graph.create_png())

    data_set = DataSet("C:\\Users\\shayb\\OneDrive\\Desktop\\testData.csv")
    ant_colony = ACO(data_set)

if __name__ == "__main__":
    main()
