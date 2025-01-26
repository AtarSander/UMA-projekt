# Autor: Aleksander Szymczyk, Andrzej Tokajuk
# Data utworzenia: 25.01.2025

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

DATASETS = [
    "rain_in_australia",
    "yeast",
    "online_news_popularity"
]

METRICS = ["accuracy", "f1", "precision", "recall"]

for dataset in DATASETS:
    final_file = f"results/{dataset}_final.json"

    with open(final_file, "r") as f:
        final_data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    fig.suptitle(f"Porównanie metryk: custom vs baseline - {dataset}", fontsize=14)

    for ax, metric in zip(axes, METRICS):
        custom_vals = []
        baseline_vals = []
        for trial in final_data:
            custom_vals.append(trial["custom"].get(metric, np.nan))
            baseline_vals.append(trial["baseline"].get(metric, np.nan))

        data_for_boxplot = [custom_vals, baseline_vals]

        sns.boxplot(
            data=data_for_boxplot,
            ax=ax,
            width=0.4,
            palette="Set2"
        )

        ax.set_xticklabels(["Custom RF", "Baseline RF"])
        ax.set_title(metric.capitalize())
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"plots/{dataset}_model_comparison.png", bbox_inches='tight')
    plt.close(fig)

    custom_conf_matrices = []
    for trial in final_data:
        if "confusion_matrix" in trial["custom"]:
            cm = trial["custom"]["confusion_matrix"]
            custom_conf_matrices.append(cm)

    if len(custom_conf_matrices) > 0:
        mean_conf_matrix = np.mean(custom_conf_matrices, axis=0)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            mean_conf_matrix,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            cbar=True,
            ax=ax
        )
        ax.set_title(f"Średnia macierz pomyłek dla własnej implementacji - {dataset}")

        plt.tight_layout()
        plt.savefig(f"plots/{dataset}_confusion_matrix.png", bbox_inches='tight')
        plt.close(fig)
