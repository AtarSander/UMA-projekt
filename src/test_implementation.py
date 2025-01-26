# Autor: Aleksander Szymczyk, Andrzej Tokajuk
# Data utworzenia: 15.01.2025

from random_forest import RandomForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import time


def load_dataset(path):
    df = pd.read_csv(path, header=None)
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
    return X, y


def test_implementation(path, n_estimators):
    X, y = load_dataset(path)
    recall = []
    precision = []
    f1 = []
    accuracy_test = []
    accuracy_train = []
    times = []

    for n in n_estimators:
        print(f"n_estimators: {n}\n")
        for _ in range(25):
            start_train = time.time()
            train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
            random_forest = RandomForest(
                n_estimators=n, max_depth=20, min_samples_split=30
            )
            train = pd.concat([train_x, train_y], axis=1)
            random_forest.build_forest(train, "poisonous")
            train_time = time.time() - start_train
            predictions_test = []
            predictions_train = []
            for index, row in test_x.iterrows():
                predictions_test.append(random_forest.predict(row))

            for index, row in train_x.iterrows():
                predictions_train.append(random_forest.predict(row))

            accuracy_train.append(accuracy_score(train_y, predictions_train))
            accuracy_test.append(accuracy_score(test_y, predictions_test))
            f1.append(f1_score(test_y, predictions_test, average='weighted'))
            precision.append(precision_score(test_y, predictions_test, average='weighted'))
            recall.append(recall_score(test_y, predictions_test, average='weighted'))
            times.append(train_time)

        print(f"Accuracy train: {sum(accuracy_train) / len(accuracy_train):.2f}")
        print(f"Accuracy train std: {np.std(accuracy_train):.2f}")
        print(f"Accuracy test: {sum(accuracy_test) / len(accuracy_test):.2f}")
        print(f"Accuracy test std: {np.std(accuracy_test):.2f}")
        print(f"F1: {sum(f1) / len(f1):.2f}")
        print(f"F1 std: {np.std(f1):.2f}")
        print(f"Precision: {sum(precision) / len(precision):.2f}")
        print(f"Precission std: {np.std(precision):.2f}")
        print(f"Recall: {sum(recall) / len(recall):.2f}")
        print(f"Recall std: {np.std(recall):.2f}")
        print(f"Time: {sum(times) / len(times):.2f}")
        print(f"Time std: {np.std(times):.2f} \n")


if __name__ == "__main__":
    path = "datasets/mushroom/agaricus-lepiota.data"
    n_estimators = [10, 5, 1]
    test_implementation(path, n_estimators)

