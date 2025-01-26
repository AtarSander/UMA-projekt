# Autor: Aleksander Szymczyk, Andrzej Tokajuk
# Data utworzenia: 26.01.2025

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

DATASETS = [
    "rain_in_australia",
    "yeast",
    "online_news_popularity"
]

PARAM_GRID_ORDER = [
    ("n_estimators", [50, 100, 150, 200]),
    ("max_depth", [5, 10, 20, 30]),
    ("min_samples_split", [2, 5, 10, 20]),
    ("max_features", ["sqrt", "log2", "0.3"])
]

for dataset in DATASETS:
    tuning_file = f"results/{dataset}_tuning.json"

    with open(tuning_file, "r") as f:
        tuning_data = json.load(f)


    df = pd.json_normalize(tuning_data, sep='.')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    fig.suptitle(f"Strojenie hiperparametrów -- {dataset}", fontsize=14)

    start_idx = 0
    for i, (param_name, param_values) in enumerate(PARAM_GRID_ORDER):
        length = len(param_values)

        subset = df.iloc[start_idx : start_idx + length].copy()
        start_idx += length

        param_as_str = [str(v) for v in param_values]
        subset[param_name] = param_as_str

        ax = axes[i]
        ax.scatter(subset[param_name], subset["test_data.f1"], c='b')

        ax.set_title(f'Strojenie {param_name}')
        ax.set_xlabel(param_name)
        ax.set_ylabel('F1-score')
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"plots/{dataset}_tuning_params.png", bbox_inches='tight')
    plt.close(fig)
