# Autor: Aleksander Szymczyk, Andrzej Tokajuk
# Data utworzenia: 26.01.2025

import random
from collections import Counter

import graphviz
import numpy as np
import pandas as pd

from tree_node import TreeNode


class CartTree:
    def __init__(self, max_depth, min_samples_split, max_features):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    def visualize_tree(self):
        dot = graphviz.Digraph()
        root = self.root
        dot.node(str(root))

        def add_nodes_edges(node):
            if node.left:
                dot.node(str(node.left))
                dot.edge(str(node), str(node.left))
                add_nodes_edges(node.left)
            if node.right:
                dot.node(str(node.right))
                dot.edge(str(node), str(node.right))
                add_nodes_edges(node.right)

        add_nodes_edges(root)
        dot.render('cart_tree', view=False, format='png')

    def predict(self, sample):
        return self.predict_process(sample, self.root)

    def predict_process(self, sample, node):
        if node.feature is None:
            return node.label
        if isinstance(node.condition, tuple):
            if sample[node.feature] in node.condition[0]:
                return self.predict_process(sample, node.left)
            return self.predict_process(sample, node.right)
        if sample[node.feature] < node.condition:
            return self.predict_process(sample, node.left)
        return self.predict_process(sample, node.right)

    def build_tree(self,dataset, target):
        self.root = self.build_process(dataset, 0, target)

    def build_process(self, dataset, current_depth, target):
        if (
                len(dataset) < self.min_samples_split
                or len(dataset[target].unique()) == 1
                or current_depth >= self.max_depth
        ):
            return self.make_leaf(dataset, target)

        split_condition = self.choose_best_split(dataset, target)
        if split_condition is None:
            return self.make_leaf(dataset, target)

        left_data, right_data = self.split_tree(dataset, split_condition)
        left_branch = self.build_process(left_data, current_depth + 1, target)
        right_branch = self.build_process(right_data, current_depth + 1, target)

        return TreeNode(split_condition, left_branch, right_branch)

    def make_leaf(self, dataset, target):
        majority_class = dataset[target].mode()[0]
        return TreeNode(None, None, None, majority_class)

    def split_tree(self, dataset, split_condition):
        if isinstance(split_condition[1], tuple):
            return self.split_tree_categorical(dataset, split_condition)
        else:
            return self.split_tree_numerical(dataset, split_condition)

    def split_tree_categorical(self, dataset, split_condition):
        feature = split_condition[0]
        group1 = split_condition[1][0]
        group2 = split_condition[1][1]
        left_dataset = dataset[dataset[feature].isin(group1)]
        right_dataset = dataset[dataset[feature].isin(group2)]
        return left_dataset, right_dataset

    def split_tree_numerical(self, dataset, split_condition):
        feature = split_condition[0]
        split_point = split_condition[1]
        left_dataset = dataset[dataset[feature] < split_point]
        right_dataset = dataset[dataset[feature] >= split_point]
        return left_dataset, right_dataset

    def choose_best_split(self, dataset, target):
        split_points = self.make_splits(dataset, target)
        if not split_points:
            return None
        tournament_pair = random.choices(split_points, k=2)
        gini_1 = self.calculate_gini(dataset, tournament_pair[0], target)
        gini_2 = self.calculate_gini(dataset, tournament_pair[1], target)
        return tournament_pair[0] if gini_1 < gini_2 else tournament_pair[1]

    def make_splits(self, dataset, target):
        split_points = []
        features = dataset.columns.drop(target)
        selected_features = self._select_features(features)
        for column in selected_features:
            if dataset[column].nunique() < 2:
                continue
            if pd.api.types.is_numeric_dtype(dataset[column]):
                splits = self.split_numerical(dataset, column)
            else:
                splits = self.split_categorical(dataset, column)
            if splits:
                split_points.extend(splits)
        return split_points

    def _select_features(self, features):
        n_features = len(features)
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features))
        else:
            max_features = self.max_features

        max_features = max(1, min(max_features, n_features))
        return np.random.choice(features, size=max_features, replace=False)

    def split_numerical(self, dataset, column):
        unique_values = dataset[column].unique()
        if len(unique_values) < 2:
            return []
        unique_values_sorted = sorted(unique_values)
        all_split_points = [
            (column, (a + b) / 2)
            for a, b in zip(unique_values_sorted, unique_values_sorted[1:])
        ]
        max_splits = 10
        if len(all_split_points) > max_splits:
            return random.sample(all_split_points, max_splits)
        return all_split_points

    def split_categorical(self, dataset, column):
        unique_values = list(dataset[column].unique())
        n = len(unique_values)
        if n < 2:
            return []
        shuffled = random.sample(unique_values, len(unique_values))
        k = random.randint(1, n-1)
        group1 = set(shuffled[:k])
        group2 = set(shuffled[k:])
        return [(column, (group1, group2))]

    def calculate_gini(self, dataset, split, target):
        if isinstance(split, tuple) and isinstance(split[1], tuple):
            feature = split[0]
            group1 = split[1][0]
            left_mask = dataset[feature].isin(group1)
            right_mask = ~left_mask
        else:
            feature = split[0]
            split_point = split[1]
            left_mask = dataset[feature] < split_point
            right_mask = ~left_mask

        left_targets = dataset.loc[left_mask, target]
        right_targets = dataset.loc[right_mask, target]
        left_counts = Counter(left_targets)
        right_counts = Counter(right_targets)
        N_left = sum(left_counts.values())
        N_right = sum(right_counts.values())

        gini_left = self.gini_split(left_counts.values(), N_left)
        gini_right = self.gini_split(right_counts.values(), N_right)
        G_split = (N_left / len(dataset)) * gini_left + (
            N_right / len(dataset)
        ) * gini_right

        return G_split

    def gini_split(self, counts, total):
        G = 1
        probabilities = [value / total for value in counts]
        return G - sum(p ** 2 for p in probabilities)
