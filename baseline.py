from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

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

    def fit(self, X, y, n_estimators):
        self.model = Pipeline(
        steps=[
        ('preprocessor', self.preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=n_estimators))
            ]
        )
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

