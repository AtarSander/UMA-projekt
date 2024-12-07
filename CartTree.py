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
        splits = [column]

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
        target_values_left = {value: 0 for value in dataset[target].unique()}
        target_values_right = dict(target_values_left)
        N_left = 0
        N_right = 0
        if isinstance(split, list):
            label = split[0]
            group1 = split[1]
            group2 = split[2]
            for row in dataset.iterrows():
                for category in group1:
                    if row[label] == category:
                        N_left += 1
                        target_values_left[row[target]] += 1
                for category in group2:
                    if row[label] == category:
                        N_right += 1
                        target_values_right[row[target]] += 1
        else:
            label = split[0]
            split_point = split[1]
            for row in dataset.iterrows():
                if row[label] > split_point:
                    N_left += 1
                    target_values_left[row[target]] += 1
                else:
                    N_right += 1
                    target_values_right[row[target]] += 1
        gini_left = self.gini_split(target_values_left)
        gini_right = self.gini_split(target_values_right)
        G_split = (N_left / len(dataset)) * gini_left + (
            N_right / len(dataset)
        ) * gini_right

        return G_split

    def gini_split(self, probabilities):
        G = 1
        for value in probabilities.values():
            G -= (value / len(probabilities)) ** 2
        return G
