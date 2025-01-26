# Autor: Aleksander Szymczyk, Andrzej Tokajuk
# Data utworzenia: 07.12.2025

from cart_tree import CartTree


class RandomForest:
    def __init__(self, n_estimators, max_depth, min_samples_split, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    def predict(self, sample):
        predictions = []
        for tree in self.forest:
            predictions.append(tree.predict(sample))
        return max(set(predictions), key=predictions.count)

    def build_forest(self, dataset, target):
        self.forest = []
        for _ in range(self.n_estimators):
            bootstrap_dataset = self.bootstraping(dataset)
            tree = CartTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.build_tree(bootstrap_dataset, target)
            self.forest.append(tree)

    def bootstraping(self, dataset):
        n = len(dataset)
        bootstrap_dataset = dataset.sample(n=n, replace=True)
        return bootstrap_dataset
