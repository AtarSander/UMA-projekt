from cart_tree import CartTree
import numpy as np


class RandomForest:
    def __init__(self, n_estimators, max_tree_depth, min_tree_samples_split):
        self.n_estimators = n_estimators
        self.max_depth = max_tree_depth
        self.min_samples_split = min_tree_samples_split

    def predict(self, sample):
        predictions = []
        for tree in self.forest:
            predictions.append(tree.predict(sample))
        return max(set(predictions), key=predictions.count)

    def build_forest(self, dataset, target):
        self.forest = []
        for _ in range(self.n_estimators):
            bootstrap_dataset = self.bootstraping(dataset)
            tree = CartTree(self.max_depth, self.min_samples_split)
            tree.build_tree(bootstrap_dataset, target)
            self.forest.append(tree)

    def bootstraping(self, dataset):
        n = len(dataset)
        bootstrap_dataset = dataset.sample(n=n, replace=True)
        return bootstrap_dataset
