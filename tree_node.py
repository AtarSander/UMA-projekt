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