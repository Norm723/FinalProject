import numpy as np


class Node:
    def __init__(self, data_set, depth, score=1):
        self.feature = 0
        self.threshold = 0
        self.leftNode = None
        self.rightNode = None
        self.depth = depth
        self.data_set = data_set
        self.score = score

    def Split(self):
        pass

    def SetLeftNode(self, left_node):
        self.leftNode = left_node

    def SetRightNode(self, right_node):
        self.rightNode = right_node

    def __IsLeaf(self):
        if self.leftNode is None and self.rightNode is None:
            return True
        else:
            return False

    def Print(self):
        print(self.data_set.data)
        print("Feature: {feature}".format(feature=self.feature))
        print("Threshold: {threshold}".format(threshold=self.threshold))
        print("\n")
        if self.__IsLeaf():
            return
        else:
            if self.leftNode is not None:
                self.leftNode.Print()
            if self.rightNode is not None:
                self.rightNode.Print()

