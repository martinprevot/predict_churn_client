"""Modèle baseline régression : Régression Linéaire."""

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def build_linear_regression_pipeline(preprocessor):
    """Pipeline Ridge Regression avec gestion de la multicolinéarité."""
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0, random_state=42))
    ])
