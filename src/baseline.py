# Autor: Aleksander Szymczyk, Andrzej Tokajuk
# Data utworzenia: 26.01.2025

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Baseline:
    def setup_preprocessor(self, numerical_columns, categorical_columns):
        categorical_transformer = OneHotEncoder()
        numerical_transformer = StandardScaler()
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_columns),
                ('cat', categorical_transformer, categorical_columns)
            ]
        )

    def fit(self, X, y, params):
        self.model = Pipeline(
            steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', RandomForestClassifier(**params))
            ]
        )
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
