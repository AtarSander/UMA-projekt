# Autor: Aleksander Szymczyk, Andrzej Tokajuk
# Data utworzenia: 26.01.2025

import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from baseline import Baseline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from random_forest import RandomForest

with open("datasets_config.json", "r") as f:
    DATASETS = json.load(f)["datasets"]

PARAM_GRID = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [5, 10, 20, 30],
    "min_samples_split": [2, 5, 10, 20],
    "max_features": ["sqrt", "log2", 0.3]
}

N_TRIALS = 25
OUTPUT_DIR = "results"

def preprocess_data(X, y):
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

    return X_processed, y

def load_data(dataset_config):
    params = dataset_config["csv_params"]

    read_csv_args = {
        "filepath_or_buffer": dataset_config["path"],
        "header": 0 if params["header"] else None,
        "names": params["names"]
    }

    if params.get("delim_whitespace", False):
        read_csv_args["delim_whitespace"] = True
    else:
        read_csv_args["delimiter"] = params.get("delimiter", ",")

    df = pd.read_csv(**read_csv_args)
    df.dropna(inplace=True)

    if dataset_config["target"] == "popular":
        df["popular"] = (df["shares"] >= 1400).astype(int)

    if "drop_columns" in dataset_config:
        df = df.drop(columns=dataset_config["drop_columns"], errors="ignore")

    X = df.drop(dataset_config["target"], axis=1)
    y = df[dataset_config["target"]]

    X_processed, y_processed = preprocess_data(X, y)

    return X_processed, y_processed

def evaluate_model(model, X_test, y_test, is_custom=True):
    if is_custom:
        preds = [model.predict(row) for _, row in X_test.iterrows()]
    else:
        preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average='weighted'),
        "precision": precision_score(y_test, preds, average='weighted'),
        "recall": recall_score(y_test, preds, average='weighted'),
    }

def run_trial(params, X_train, y_train, X_eval, y_eval, evaluate_baseline=False):
    start_train = time.time()
    custom_rf = RandomForest(**params)
    train_data = pd.concat([X_train, y_train], axis=1)
    custom_rf.build_forest(train_data, y_train.name)
    train_time = time.time() - start_train
    custom_metrics = evaluate_model(custom_rf, X_eval, y_eval)

    baseline_metrics = {}
    if evaluate_baseline:
        params = {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "min_samples_split": params["min_samples_split"],
            "max_features": params["max_features"],
            "random_state": 42,
            "n_jobs": -1
        }
        baseline_rf = Baseline()
        baseline_rf.fit(X_train, y_train, params)
        baseline_metrics = evaluate_model(baseline_rf, X_eval, y_eval, is_custom=False)

    return {
        "custom": {**custom_metrics, "train_time": train_time},
        "baseline": baseline_metrics
    }

def tune_parameters(dataset_name, X_train, y_train, X_val, y_val):
    best_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "max_features": "sqrt"
    }
    tuning_results = []

    for param in PARAM_GRID.keys():
        print(f"Strojenie parametru: {param}")
        values = PARAM_GRID[param]

        trials = []
        for value in values:
            trial_params = best_params.copy()
            trial_params[param] = value
            trials.append(trial_params)

        results = Parallel(n_jobs=-1, verbose=10, prefer="processes")(
            delayed(run_trial)(params, X_train, y_train, X_val, y_val)
            for params in trials
        )

        best_score = -1
        best_value = best_params[param]
        for value, result, params in zip(values, results, trials):
            current_score = result["custom"]["f1"]

            tuning_results.append({
                "dataset": dataset_name,
                "params": params,
                "f1": current_score,
                "accuracy": result["custom"]["accuracy"],
                "precision": result["custom"]["precision"],
                "recall": result["custom"]["recall"],
                "train_time": result["custom"]["train_time"],
            })

            if current_score > best_score:
                best_score = current_score
                best_value = value

        best_params[param] = best_value
        print(f"Najlepsza wartość dla {param}: {best_value} (F1: {best_score:.3f})")

    with open(f"{OUTPUT_DIR}/{dataset_name}_tuning.json", "w") as f:
        json.dump(tuning_results, f, indent=2)

    return best_params

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for dataset_name, config in DATASETS.items():
        X, y = load_data(config)
        X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

        best_params = tune_parameters(dataset_name, X_train, y_train, X_val, y_val)
        print(f"\nNajlepsze parametry: {best_params}")

        all_results = []

        trial_results = Parallel(n_jobs=-1, verbose=1)(
            delayed(run_trial)(best_params, X_train, y_train, X_test, y_test, evaluate_baseline=True)
            for _ in range(N_TRIALS)
        )

        for trial in trial_results:
            record = {
                "custom": trial["custom"],
                "baseline": trial["baseline"]
            }
            all_results.append(record)

        with open(f"{OUTPUT_DIR}/{dataset_name}_final.json", "w") as f:
            json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    print(f"Czas rozpoczęcia: {time.ctime()}")
    main()
    print(f"Czas zakończenia: {time.ctime()}")
