"""
Script principal d'entraînement — Classification (churn prediction).
Lance : python src/train.py
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    preprocess_data, build_preprocessor, load_and_split,
    get_feature_names, compute_scale_pos_weight
)
from src.models.logistic_model import build_logistic_pipeline
from src.models.random_forest_model import build_rf_pipeline
from src.models.xgboost_model import build_xgb_pipeline
from src.models.mlp_model import train_mlp
from src.evaluate import (
    compute_classification_metrics, plot_roc_curves, plot_precision_recall_curves,
    plot_confusion_matrix, plot_metrics_comparison, plot_mlp_learning_curves,
    plot_feature_importance, generate_shap_plots, save_metrics_csv
)

DATA_PATH = 'data/raw/customer_churn_business_dataset.csv'
MODELS_DIR = 'models'


def train_classification():
    print("=" * 65)
    print("ENTRAÎNEMENT — CLASSIFICATION (Prédiction Churn)")
    print("=" * 65)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)

    # --- Preprocessing ---
    print("\n[1/5] Chargement et preprocessing...")
    X_train, X_test, y_train, y_test, _ = load_and_split(DATA_PATH, task='classification')
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    joblib.dump(preprocessor, f'{MODELS_DIR}/preprocessor.joblib')

    feature_names = get_feature_names(preprocessor)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    print(f"  Train: {X_train_proc.shape} | Test: {X_test_proc.shape}")
    print(f"  Taux de churn train : {y_train.mean():.3f}")
    print(f"  scale_pos_weight    : {scale_pos_weight:.2f}")

    all_metrics = []
    models_results = []

    # --- Modèle 1 : Régression Logistique ---
    print("\n[2/5] Régression Logistique (baseline)...")
    t0 = time.time()
    lr = build_logistic_pipeline(build_preprocessor())
    lr.fit(X_train, y_train)
    t_lr = time.time() - t0

    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]
    m_lr = compute_classification_metrics('Logistic Regression', y_test, y_pred_lr, y_proba_lr, t_lr)
    all_metrics.append(m_lr)
    models_results.append({'name': 'Logistic Regression', 'y_true': y_test, 'y_proba': y_proba_lr})
    joblib.dump(lr, f'{MODELS_DIR}/logistic_model.joblib')
    print(f"  ROC-AUC={m_lr['roc_auc']:.3f} | F1={m_lr['f1_score']:.3f} | Recall={m_lr['recall']:.3f} ({t_lr:.1f}s)")

    # --- Modèle 2 : Random Forest ---
    print("\n[3/5] Random Forest (RandomizedSearchCV cv=5)...")
    t0 = time.time()
    rf_search = build_rf_pipeline(build_preprocessor())
    rf_search.fit(X_train, y_train)
    t_rf = time.time() - t0
    best_rf = rf_search.best_estimator_

    y_pred_rf = best_rf.predict(X_test)
    y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
    m_rf = compute_classification_metrics('Random Forest', y_test, y_pred_rf, y_proba_rf, t_rf)
    all_metrics.append(m_rf)
    models_results.append({'name': 'Random Forest', 'y_true': y_test, 'y_proba': y_proba_rf})
    joblib.dump(best_rf, f'{MODELS_DIR}/random_forest_model.joblib')
    print(f"  Best params: {rf_search.best_params_}")
    print(f"  ROC-AUC={m_rf['roc_auc']:.3f} | F1={m_rf['f1_score']:.3f} | Recall={m_rf['recall']:.3f} ({t_rf:.1f}s)")

    # --- Modèle 3 : XGBoost ---
    print("\n[4/5] XGBoost (RandomizedSearchCV cv=5)...")
    t0 = time.time()
    xgb_search = build_xgb_pipeline(build_preprocessor(), scale_pos_weight=scale_pos_weight)
    xgb_search.fit(X_train, y_train)
    t_xgb = time.time() - t0
    best_xgb = xgb_search.best_estimator_

    y_pred_xgb = best_xgb.predict(X_test)
    y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
    m_xgb = compute_classification_metrics('XGBoost', y_test, y_pred_xgb, y_proba_xgb, t_xgb)
    all_metrics.append(m_xgb)
    models_results.append({'name': 'XGBoost', 'y_true': y_test, 'y_proba': y_proba_xgb})
    joblib.dump(best_xgb, f'{MODELS_DIR}/xgboost_model.joblib')
    print(f"  Best params: {xgb_search.best_params_}")
    print(f"  ROC-AUC={m_xgb['roc_auc']:.3f} | F1={m_xgb['f1_score']:.3f} | Recall={m_xgb['recall']:.3f} ({t_xgb:.1f}s)")

    # --- Modèle 4 : MLP ---
    print("\n[5/5] MLP Deep Learning...")
    t0 = time.time()
    mlp_model, mlp_history = train_mlp(X_train_proc, y_train.values, epochs=100, batch_size=256)
    t_mlp = time.time() - t0

    y_proba_mlp = mlp_model.predict(X_test_proc).flatten()
    y_pred_mlp = (y_proba_mlp >= 0.5).astype(int)
    m_mlp = compute_classification_metrics('MLP', y_test, y_pred_mlp, y_proba_mlp, t_mlp)
    all_metrics.append(m_mlp)
    models_results.append({'name': 'MLP', 'y_true': y_test, 'y_proba': y_proba_mlp})
    mlp_model.save(f'{MODELS_DIR}/mlp_model.h5')
    print(f"  ROC-AUC={m_mlp['roc_auc']:.3f} | F1={m_mlp['f1_score']:.3f} | Recall={m_mlp['recall']:.3f} ({t_mlp:.1f}s)")

    # --- Meilleur modèle ---
    best_m = max(all_metrics, key=lambda m: m['roc_auc'])
    model_map = {'Logistic Regression': lr, 'Random Forest': best_rf, 'XGBoost': best_xgb}
    best_model = model_map.get(best_m['model_name'], best_xgb)
    joblib.dump(best_model, f'{MODELS_DIR}/best_model.joblib')
    print(f"\nMeilleur modèle : {best_m['model_name']} (ROC-AUC={best_m['roc_auc']:.3f})")

    # --- Visualisations ---
    print("\nGénération des visualisations...")
    plot_roc_curves(models_results)
    plot_precision_recall_curves(models_results)
    plot_confusion_matrix(y_test, y_pred_xgb, model_name='XGBoost')
    plot_metrics_comparison(all_metrics)
    plot_mlp_learning_curves(mlp_history, task='classification')

    xgb_clf = best_xgb.named_steps['classifier']
    plot_feature_importance(xgb_clf, feature_names, model_name='XGBoost')
    generate_shap_plots(xgb_clf, X_test_proc, feature_names)

    # --- Résultats ---
    df_metrics = save_metrics_csv(all_metrics, 'reports/model_comparison.csv')
    print("\n" + "=" * 65)
    print("RÉSULTATS FINAUX — CLASSIFICATION")
    print("=" * 65)
    cols = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'train_time_s']
    print(df_metrics[cols].to_string(index=False))

    return all_metrics


if __name__ == '__main__':
    train_classification()
