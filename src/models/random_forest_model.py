"""Modèle Random Forest avec optimisation par RandomizedSearchCV."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def build_rf_pipeline(preprocessor):
    """
    Pipeline Random Forest avec RandomizedSearchCV.
    Optimise sur ROC-AUC avec cross-validation 5-fold stratifiée.
    """
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

    param_distributions = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [5, 10, 15, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2']
    }

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])

    return RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=20, cv=5, scoring='roc_auc',
        n_jobs=-1, random_state=42, verbose=1
    )
