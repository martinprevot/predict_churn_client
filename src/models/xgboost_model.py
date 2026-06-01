"""Modèle XGBoost avec optimisation par RandomizedSearchCV."""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb


def build_xgb_pipeline(preprocessor, scale_pos_weight=1.0):
    """
    Pipeline XGBoost avec RandomizedSearchCV.

    Args:
        scale_pos_weight: ratio n_négatifs / n_positifs
    """
    xgb_clf = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )

    param_distributions = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__subsample': [0.7, 0.8, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 1.0],
        'classifier__min_child_weight': [1, 3, 5]
    }

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb_clf)
    ])

    return RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=20, cv=5, scoring='roc_auc',
        n_jobs=-1, random_state=42, verbose=1
    )
