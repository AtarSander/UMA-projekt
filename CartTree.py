import pandas as pd
import random


class CartTree:
    def __init__(self, max_depth, min_samples_split):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def build_tree(self, dataset, current_depth, target):
        if (
            len(dataset.index) < self.min_samples_split
            or len(dataset[target].values()) == 1
            or current_depth > self.max_depth
        ):
            return self.make_leaf(dataset, target)
        split_condition, split_value = self.make_split(dataset, target)
        for value in dataset[split_condition]:
            pass

    def make_split(self, dataset, target):
        for column in dataset:
            data_type = dataset[column].dtype()
            split_points = []
            if pd.api.types.is_numeric_dtype(data_type):
                split_point = self.split_numerical(dataset, column)
            else:
                split_point = self.split_categorical(dataset, column)
            split_points.append(split_point)
        tournament_pair = random.choices(split_points, k=2)
        gini_index_1 = self.calculate_gini(dataset, tournament_pair[0], target)
        gini_index_2 = self.calculate_gini(dataset, tournament_pair[1], target)
        if gini_index_1 < gini_index_2:
            return tournament_pair[0]
        return tournament_pair[1]

    def split_numerical(self, dataset, column):
        unique_values_sorted = list(dataset[column].unique()).sort()
        split_point = [
            (column, (a + b) // 2)
            for a, b in zip(unique_values_sorted, unique_values_sorted[1:])
        ]
        return split_point

    def split_categorical(self, dataset, column):
        unique_values = list(dataset[column].unique())
        combinations = []
        splits = []

        def backtrack(value, other_values, split):
            if len(other_values) == 1:
                split.append(value)
                combinations.append(list(split))
                path = []
            path.append(value)
            for i in other_values:
                new_other_values = other_values.remove(i)
                backtrack(i, new_other_values, split)
            split.pop()

        backtrack(unique_values[-1], unique_values.pop())
        group1 = []
        for i in range(len(combinations)):
            for _ in combinations:
                group1.append(combinations[i])
                group2 = combinations - group1
            splits.append(group1 + group2)

        return splits

    def calculate_gini(self, dataset, split, target):
        C = len(dataset[target].unique())
        if isinstance(split, list):
            pass
        else:
            pass
