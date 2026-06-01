"""Modèle Random Forest Regressor avec optimisation par RandomizedSearchCV."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def build_rf_regressor_pipeline(preprocessor):
    """
    Pipeline Random Forest Regressor avec RandomizedSearchCV.
    Optimise sur R² avec cross-validation 5-fold.
    """
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_distributions = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [5, 10, 15, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__max_features': ['sqrt', 'log2']
    }

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', rf)
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    return search
