"""
Script d'entraînement — Régression (estimation CLV / total_revenue).
Lance : python src/train_regression.py
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    preprocess_data, build_preprocessor_regression, load_and_split,
    get_feature_names
)
from src.models.linear_regression_model import build_linear_regression_pipeline
from src.models.rf_regressor_model import build_rf_regressor_pipeline
from src.models.mlp_regressor_model import train_mlp_regressor
from src.evaluate import (
    compute_regression_metrics, plot_regression_metrics_comparison,
    plot_residuals, plot_predictions_vs_actual, plot_mlp_learning_curves,
    plot_feature_importance, generate_shap_plots, save_metrics_csv
)

DATA_PATH = 'data/raw/customer_churn_business_dataset.csv'
MODELS_DIR = 'models'


def train_regression():
    print("=" * 65)
    print("ENTRAÎNEMENT — RÉGRESSION (Estimation CLV / total_revenue)")
    print("=" * 65)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)

    # --- Preprocessing ---
    print("\n[1/4] Chargement et preprocessing...")
    X_train, X_test, y_train, y_test, _ = load_and_split(DATA_PATH, task='regression')
    preprocessor = build_preprocessor_regression()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    joblib.dump(preprocessor, f'{MODELS_DIR}/preprocessor_regression.joblib')

    feature_names = get_feature_names(preprocessor)
    print(f"  Train: {X_train_proc.shape} | Test: {X_test_proc.shape}")
    print(f"  CLV moyen train : {y_train.mean():.1f} ± {y_train.std():.1f}")

    all_metrics = []
    models_results = []

    # --- Modèle 1 : Ridge Regression (baseline) ---
    print("\n[2/4] Ridge Regression (baseline)...")
    t0 = time.time()
    ridge = build_linear_regression_pipeline(build_preprocessor_regression())
    ridge.fit(X_train, y_train)
    t_ridge = time.time() - t0

    y_pred_ridge = ridge.predict(X_test)
    m_ridge = compute_regression_metrics('Ridge Regression', y_test, y_pred_ridge, t_ridge)
    all_metrics.append(m_ridge)
    models_results.append({'name': 'Ridge Regression', 'y_true': y_test, 'y_pred': y_pred_ridge})
    joblib.dump(ridge, f'{MODELS_DIR}/ridge_regression_model.joblib')
    print(f"  MAE={m_ridge['mae']:.2f} | RMSE={m_ridge['rmse']:.2f} | R²={m_ridge['r2']:.4f} ({t_ridge:.1f}s)")

    # --- Modèle 2 : Random Forest Regressor ---
    print("\n[3/4] Random Forest Regressor (RandomizedSearchCV cv=5)...")
    t0 = time.time()
    rf_search = build_rf_regressor_pipeline(build_preprocessor_regression())
    rf_search.fit(X_train, y_train)
    t_rf = time.time() - t0
    best_rf = rf_search.best_estimator_

    y_pred_rf = best_rf.predict(X_test)
    m_rf = compute_regression_metrics('Random Forest Regressor', y_test, y_pred_rf, t_rf)
    all_metrics.append(m_rf)
    models_results.append({'name': 'Random Forest Regressor', 'y_true': y_test, 'y_pred': y_pred_rf})
    joblib.dump(best_rf, f'{MODELS_DIR}/rf_regressor_model.joblib')
    print(f"  Best params: {rf_search.best_params_}")
    print(f"  MAE={m_rf['mae']:.2f} | RMSE={m_rf['rmse']:.2f} | R²={m_rf['r2']:.4f} ({t_rf:.1f}s)")

    # --- Modèle 3 : MLP Regressor (Deep Learning) ---
    print("\n[4/4] MLP Regressor Deep Learning...")
    t0 = time.time()
    mlp_model, mlp_history = train_mlp_regressor(
        X_train_proc, y_train.values, epochs=100, batch_size=256
    )
    t_mlp = time.time() - t0

    y_pred_mlp = mlp_model.predict(X_test_proc).flatten()
    m_mlp = compute_regression_metrics('MLP Regressor', y_test, y_pred_mlp, t_mlp)
    all_metrics.append(m_mlp)
    models_results.append({'name': 'MLP Regressor', 'y_true': y_test, 'y_pred': y_pred_mlp})
    mlp_model.save(f'{MODELS_DIR}/mlp_regressor_model.h5')
    print(f"  MAE={m_mlp['mae']:.2f} | RMSE={m_mlp['rmse']:.2f} | R²={m_mlp['r2']:.4f} ({t_mlp:.1f}s)")

    # --- Meilleur modèle régression ---
    best_m = max(all_metrics, key=lambda m: m['r2'])
    model_map = {'Ridge Regression': ridge, 'Random Forest Regressor': best_rf}
    best_model = model_map.get(best_m['model_name'], best_rf)
    joblib.dump(best_model, f'{MODELS_DIR}/best_regression_model.joblib')
    print(f"\nMeilleur modèle : {best_m['model_name']} (R²={best_m['r2']:.4f})")

    # --- Visualisations ---
    print("\nGénération des visualisations...")
    plot_regression_metrics_comparison(all_metrics)
    plot_predictions_vs_actual(models_results)
    plot_mlp_learning_curves(mlp_history, task='regression')

    # Résidus du meilleur modèle
    best_preds = model_map.get(best_m['model_name'], best_rf).predict(X_test)
    plot_residuals(y_test, best_preds, model_name=best_m['model_name'])

    # Feature importance RF
    rf_reg = best_rf.named_steps['regressor']
    plot_feature_importance(rf_reg, feature_names, model_name='RF Regressor')
    generate_shap_plots(rf_reg, X_test_proc, feature_names, suffix='_regression')

    # --- Résultats ---
    df_metrics = save_metrics_csv(all_metrics, 'reports/regression_model_comparison.csv')
    print("\n" + "=" * 65)
    print("RÉSULTATS FINAUX — RÉGRESSION")
    print("=" * 65)
    cols = ['model_name', 'mae', 'rmse', 'r2', 'mape', 'train_time_s']
    print(df_metrics[cols].to_string(index=False))

    return all_metrics


if __name__ == '__main__':
    train_regression()
