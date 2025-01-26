# Autor: Aleksander Szymczyk, Andrzej Tokajuk
# Data utworzenia: 26.01.2025

from cart_tree import CartTree
from random_forest import RandomForest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from baseline import Baseline



df = pd.read_csv("../datasets/mushroom/agaricus-lepiota.data", header=None)
df.columns = [
    "poisonous",
    "cap_shape",
    "cap_surface",
    "cap_color",
    "bruises",
    "odor",
    "gill_attachment",
    "gill_spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

X = df.drop("poisonous", axis=1)
y = df["poisonous"]

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
random_forest = RandomForest(
    n_estimators=10, max_depth=40, min_samples_split=30
)
train = pd.concat([train_x, train_y], axis=1)
random_forest.build_forest(train, "poisonous")
predictions = []
for index, row in test_x.iterrows():
    predictions.append(random_forest.predict(row))


accuracy = accuracy_score(test_y, predictions)
print(f"Accuracy: {accuracy:.2f}")

# cartree = CartTree(max_depth=40, min_samples_split=30)
# predictions = []
# cartree.build_tree(train, "poisonous")
# for index, row in test_x.iterrows():
#     predictions.append(cartree.predict(row))
# cartree.visualize_tree()
# accuracy = accuracy_score(test_y, predictions)
# print(f"Accuracy: {accuracy:.2f}")


baseline_rf = Baseline()

categorical_columns = X.columns.tolist()
baseline_rf.setup_preprocessor([], categorical_columns)

baseline_rf.fit(train_x, train_y, params={"n_estimators":10, "random_state":42})

y_pred = baseline_rf.predict(test_x)

accuracy = accuracy_score(test_y, y_pred)
print(f"Accuracy: {accuracy:.2f}")