# Autor: Aleksander Szymczyk, Andrzej Tokajuk
# Data utworzenia: 26.01.2025

class TreeNode:
    def __init__(self, split_condition, left, right, label=None):
        if split_condition is None:
            self.feature = None
            self.condition = None
        else:
            self.feature = split_condition[0]
            self.condition = split_condition[1]
        self.left = left
        self.right = right
        self.label = label

    def __str__(self):
        if self.feature is None:
            return f"({self.label})"
        return f"({self.feature}, {self.condition})"
