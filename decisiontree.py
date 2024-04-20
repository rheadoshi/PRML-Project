import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def calculate_entropy(self,col):
      data, counts = np.unique(col, return_counts=True)
      N = int(col.shape[0])

      entropy = 0.0

      for count in counts:
        p = count / N
        entropy += p * np.log2(p)

      return -entropy

    def information_gain(self,y, y_left, y_right):
      parent_entropy = self.calculate_entropy(y)
      left_weight = len(y_left) / len(y)
      right_weight = len(y_right) / len(y)
      child_entropy = left_weight * self.calculate_entropy(y_left) + right_weight * self.calculate_entropy(y_right)
      return parent_entropy - child_entropy

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(set(y)) == 1:
            # If max depth reached or pure node ==> leaf node
            return {'class': max(set(y), key=y.tolist().count), 'is_leaf': True}

        best_split_attribute, best_split_value = self._get_best_split(X, y)

        if best_split_attribute is None:
            # information gain = 0 ==>leaf node
            return {'class': max(set(y), key=y.tolist().count), 'is_leaf': True}

        left_indices = X[best_split_attribute] == best_split_value
        right_indices = ~left_indices

        if len(left_indices) == 0 or len(right_indices) == 0:
            # If a split=empty subset ==>leaf node
            return {'class': max(set(y), key=y.tolist().count), 'is_leaf': True}

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'attribute': best_split_attribute,
            'value': best_split_value,
            'left': left_subtree,
            'right': right_subtree,
            'is_leaf': False
        }

    def _get_best_split(self, X, y):
        best_information_gain = 0
        best_split_attribute = None
        best_split_value = None

        for feature in X.columns:
            categories = X[feature].unique()
            for category in categories:
                left_indices = X[feature] == category
                right_indices = ~left_indices

                y_left, y_right = y[left_indices], y[right_indices]
                current_information_gain = self.information_gain(y, y_left, y_right)

                if current_information_gain > best_information_gain:
                    best_information_gain = current_information_gain
                    best_split_attribute = feature
                    best_split_value = category

        return best_split_attribute, best_split_value

    def predict(self, X):
        if self.tree is None:
            raise ValueError("The decision tree has not been fitted yet.")

        return np.array([self._traverse_tree(x, self.tree) for _, x in X.iterrows()])

    def _traverse_tree(self, sample, node):
        while not node['is_leaf']:
            attribute = node['attribute']
            value = node['value']

            if sample[attribute] == value:
                node = node['left']
            else:
                node = node['right']

        return node['class']
