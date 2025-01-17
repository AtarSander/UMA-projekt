from itertools import combinations
from tree_node import TreeNode
import pandas as pd
import random
import graphviz


class CartTree:
    def __init__(self, max_depth, min_samples_split):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

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
            or current_depth > self.max_depth
        ):
            return self.make_leaf(dataset, target)
        split_condition = self.choose_best_split(dataset, target)
        left_dataset, right_dataset = self.split_tree(dataset, split_condition)

        left_branch = self.build_process(left_dataset, current_depth + 1, target)
        right_branch = self.build_process(right_dataset, current_depth + 1, target)
        return TreeNode(split_condition, left_branch, right_branch)

    def make_leaf(self, dataset, target):
        majority_class = dataset[target].mode()[0]
        return TreeNode(None, None, None, majority_class)

    def split_tree(self, dataset, split_condition):
        if isinstance(split_condition, list):
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
        tournament_pair = random.choices(split_points, k=2)
        gini_index_1 = self.calculate_gini(dataset, tournament_pair[0], target)
        gini_index_2 = self.calculate_gini(dataset, tournament_pair[1], target)
        if gini_index_1 < gini_index_2:
            return tournament_pair[0]
        return tournament_pair[1]

    def make_splits(self, dataset, target):
        split_points = []
        for column in dataset:
            if column == target:
                continue
            data_type = dataset[column].dtype
            if pd.api.types.is_numeric_dtype(data_type):
                split_point = self.split_numerical(dataset, column)
                if len(split_point) > 1:
                    split_points.extend(split_point)
            else:
                split_point = self.split_categorical(dataset, column)
                if len(split_point) > 1:
                    split_points.append(split_point)
        return split_points

    def split_numerical(self, dataset, column):
        unique_values_sorted = sorted(list(dataset[column].unique()))
        split_points = [
            (column, (a + b) / 2)
            for a, b in zip(unique_values_sorted, unique_values_sorted[1:])
        ]
        return split_points

    def split_categorical(self, dataset, column):
        unique_values = list(dataset[column].unique())
        splits = [column]
        n = len(unique_values)
        for i in range(1, n // 2 + 1):
            for group1 in combinations(unique_values, i):
                group1 = set(group1)
                group2 = set(unique_values) - group1
                if (group2, group1) not in splits:
                    splits.append((group1, group2))

        return splits

    def calculate_gini(self, dataset, split, target):
        target_values_left = {value: 0 for value in dataset[target].unique()}
        target_values_right = dict(target_values_left)
        N_left = 0
        N_right = 0
        if isinstance(split, list):
            feature = split[0]
            group1 = split[1][0]
            group2 = split[1][1]
            for _, row in dataset.iterrows():
                for category in group1:
                    if row[feature] == category:
                        N_left += 1
                        target_values_left[row[target]] += 1
                for category in group2:
                    if row[feature] == category:
                        N_right += 1
                        target_values_right[row[target]] += 1
        else:
            feature = split[0]
            split_point = split[1]
            for _, row in dataset.iterrows():
                if row[feature] < split_point:
                    N_left += 1
                    target_values_left[row[target]] += 1
                else:
                    N_right += 1
                    target_values_right[row[target]] += 1

        gini_left = self.gini_split(target_values_left, N_left)
        gini_right = self.gini_split(target_values_right, N_right)
        G_split = (N_left / len(dataset)) * gini_left + (
            N_right / len(dataset)
        ) * gini_right

        return G_split

    def gini_split(self, probabilities, count):
        G = 1
        for value in probabilities.values():
            G -= (value / count) ** 2
        return G
