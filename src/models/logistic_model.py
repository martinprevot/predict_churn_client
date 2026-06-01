"""Modèle baseline : Régression Logistique."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_logistic_pipeline(preprocessor):
    """Pipeline Régression Logistique avec gestion du déséquilibre."""
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        ))
    ])
