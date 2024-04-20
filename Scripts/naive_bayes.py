import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')
class NaiveBayes:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series")

        self.class_probabilities = self.calculate_class_probabilities(y)
        self.feature_probabilities = self.calculate_feature_probabilities(X, y)

    def calculate_class_probabilities(self, y):
        class_counts = {}
        total_samples = len(y)
        for label in y.unique():
            class_counts[label] = np.sum(y == label) / total_samples
        return class_counts

    def calculate_feature_probabilities(self, X, y):
        feature_probabilities = {}
        for label in y.unique():
            label_indices = y[y == label].index
            label_data = X.loc[label_indices]
            feature_probabilities[label] = {
                'mean': label_data.mean(),
                'std': label_data.std() + 1e-6  # Adding a small value to avoid division by zero
            }
        return feature_probabilities

    def calculate_likelihood(self, x, mean, std):
        return np.exp(-((x - mean) ** 2) / (2 * std ** 2)) / (np.sqrt(2 * np.pi) * std)

    def predict_proba(self, X):
        probabilities = []
        for x in X.values:
            class_probabilities = {}
            for label, class_prob in self.class_probabilities.items():
                feature_probs = self.feature_probabilities[label]
                likelihoods = self.calculate_likelihood(x, feature_probs['mean'], feature_probs['std'])
                class_probabilities[label] = np.prod(likelihoods) * class_prob
            probabilities.append(class_probabilities)
        return probabilities

    def predict(self, X):
        probas = self.predict_proba(X)
        predictions = []
        for proba in probas:
            predictions.append(max(proba, key=proba.get))
        return predictions