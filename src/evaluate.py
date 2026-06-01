"""
Évaluation et visualisation des modèles (classification + régression).
Génère toutes les figures dans reports/figures/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend non-interactif — pas de fenêtre GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.inspection import permutation_importance

FIGURES_DIR = 'reports/figures'


# =============================================================
# MÉTRIQUES
# =============================================================

def compute_classification_metrics(model_name, y_true, y_pred, y_proba, train_time_s=0.0):
    """Métriques de classification."""
    return {
        'model_name': model_name,
        'task': 'classification',
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'train_time_s': train_time_s
    }


# Alias pour compatibilité
compute_metrics = compute_classification_metrics


def compute_regression_metrics(model_name, y_true, y_pred, train_time_s=0.0):
    """Métriques de régression : MAE, RMSE, R²."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

    return {
        'model_name': model_name,
        'task': 'regression',
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'train_time_s': train_time_s
    }


# =============================================================
# VISUALISATIONS — CLASSIFICATION
# =============================================================

def plot_roc_curves(models_results, save=True):
    """Courbes ROC superposées pour tous les modèles."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')

    for res in models_results:
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_proba'])
        auc = roc_auc_score(res['y_true'], res['y_proba'])
        ax.plot(fpr, tpr, label=f"{res['name']} (AUC = {auc:.3f})")

    ax.set_xlabel('Taux de faux positifs')
    ax.set_ylabel('Taux de vrais positifs')
    ax.set_title('Courbes ROC — Comparaison des modèles')
    ax.legend()
    ax.grid(alpha=0.3)

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(f'{FIGURES_DIR}/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_precision_recall_curves(models_results, save=True):
    """Courbes Precision-Recall pour tous les modèles."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for res in models_results:
        prec, rec, _ = precision_recall_curve(res['y_true'], res['y_proba'])
        ax.plot(rec, prec, label=res['name'])

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Courbes Precision-Recall — Comparaison des modèles')
    ax.legend()
    ax.grid(alpha=0.3)

    if save:
        fig.savefig(f'{FIGURES_DIR}/precision_recall_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name='Modèle final', save=True):
    """Heatmap de la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn'],
        ax=ax
    )
    ax.set_ylabel('Réel')
    ax.set_xlabel('Prédit')
    ax.set_title(f'Matrice de confusion — {model_name}')

    if save:
        fig.savefig(
            f'{FIGURES_DIR}/confusion_matrix_{model_name.lower().replace(" ", "_")}.png',
            dpi=150, bbox_inches='tight'
        )
    plt.close()
    return fig


def plot_metrics_comparison(metrics_list, save=True):
    """Heatmap comparatif des métriques de classification."""
    df = pd.DataFrame(metrics_list).set_index('model_name')
    cols = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    df_plot = df[cols]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df_plot, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title('Comparaison des métriques par modèle (Classification)')

    if save:
        fig.savefig(f'{FIGURES_DIR}/metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_mlp_learning_curves(history, task='classification', save=True):
    """Courbes d'apprentissage du MLP (loss + métrique principale)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    metric_key = 'auc' if task == 'classification' else 'mae'
    metric_label = 'AUC' if task == 'classification' else 'MAE'

    # Loss
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Courbe de loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Métrique principale
    if metric_key in history.history:
        axes[1].plot(history.history[metric_key], label='Train')
        axes[1].plot(history.history[f'val_{metric_key}'], label='Validation')
        axes[1].set_ylabel(metric_label)
    axes[1].set_xlabel('Epoch')
    axes[1].set_title(f'Courbe {metric_label}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    suffix = '' if task == 'classification' else '_regression'
    if save:
        fig.savefig(f'{FIGURES_DIR}/mlp_learning_curves{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    return fig


# =============================================================
# VISUALISATIONS — RÉGRESSION
# =============================================================

def plot_regression_metrics_comparison(metrics_list, save=True):
    """Tableau comparatif des métriques de régression."""
    df = pd.DataFrame(metrics_list).set_index('model_name')
    cols = ['mae', 'rmse', 'r2']
    df_plot = df[cols]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df_plot, annot=True, fmt='.2f', cmap='YlGn', ax=ax)
    ax.set_title('Comparaison des métriques par modèle (Régression CLV)')

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(f'{FIGURES_DIR}/regression_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_residuals(y_true, y_pred, model_name='Modèle', save=True):
    """
    Analyse des résidus : scatter + distribution.
    Permet de détecter biais, hétéroscédasticité, valeurs aberrantes.
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Résidus vs Valeurs prédites
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Valeurs prédites')
    axes[0].set_ylabel('Résidus')
    axes[0].set_title('Résidus vs Prédictions')
    axes[0].grid(alpha=0.3)

    # 2. Distribution des résidus
    axes[1].hist(residuals, bins=50, edgecolor='black', color='steelblue')
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_xlabel('Résidu')
    axes[1].set_ylabel('Fréquence')
    axes[1].set_title('Distribution des résidus')
    axes[1].grid(alpha=0.3)

    # 3. Valeurs réelles vs prédites
    axes[2].scatter(y_true, y_pred, alpha=0.3, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', label='Prédiction parfaite')
    axes[2].set_xlabel('Valeurs réelles')
    axes[2].set_ylabel('Valeurs prédites')
    axes[2].set_title(f'Réel vs Prédit — {model_name}')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle(f'Analyse des résidus — {model_name}', fontsize=13)
    plt.tight_layout()

    if save:
        fig.savefig(
            f'{FIGURES_DIR}/residuals_{model_name.lower().replace(" ", "_")}.png',
            dpi=150, bbox_inches='tight'
        )
    plt.close()
    return fig


def plot_predictions_vs_actual(models_results_reg, save=True):
    """Scatter réel vs prédit superposé pour tous les modèles de régression."""
    fig, axes = plt.subplots(1, len(models_results_reg), figsize=(5 * len(models_results_reg), 4))
    if len(models_results_reg) == 1:
        axes = [axes]

    for ax, res in zip(axes, models_results_reg):
        ax.scatter(res['y_true'], res['y_pred'], alpha=0.3, s=8)
        min_v = min(res['y_true'].min(), res['y_pred'].min())
        max_v = max(res['y_true'].max(), res['y_pred'].max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--')
        r2 = r2_score(res['y_true'], res['y_pred'])
        ax.set_title(f"{res['name']} (R²={r2:.3f})")
        ax.set_xlabel('Réel')
        ax.set_ylabel('Prédit')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(f'{FIGURES_DIR}/regression_predictions_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close()
    return fig


# =============================================================
# FEATURE IMPORTANCE & SHAP
# =============================================================

def plot_feature_importance(model, feature_names, model_name='Random Forest', top_n=15, save=True):
    """Barplot horizontal des feature importances natives."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance — {model_name} (Top {top_n})')

    if save:
        fig.savefig(
            f'{FIGURES_DIR}/feature_importance_{model_name.lower().replace(" ", "_")}.png',
            dpi=150, bbox_inches='tight'
        )
    plt.close()
    return fig


def plot_permutation_importance(model, X_test, y_test, feature_names, top_n=15, save=True):
    """Barplot de la permutation importance avec intervalles de confiance."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    indices = np.argsort(result.importances_mean)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        range(len(indices)),
        result.importances_mean[indices],
        xerr=result.importances_std[indices],
        align='center'
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Diminution de performance moyenne')
    ax.set_title(f'Permutation Importance (Top {top_n})')

    if save:
        fig.savefig(f'{FIGURES_DIR}/permutation_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def generate_shap_plots(model, X_test, feature_names, suffix='', save=True):
    """Génère les plots SHAP : summary + waterfall."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Summary plot
        fig1 = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        if save:
            os.makedirs(FIGURES_DIR, exist_ok=True)
            fig1.savefig(f'{FIGURES_DIR}/shap_summary{suffix}.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Waterfall (premier échantillon)
        shap_exp = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_test[0],
            feature_names=feature_names
        )
        fig2 = plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_exp, show=False)
        if save:
            fig2.savefig(f'{FIGURES_DIR}/shap_waterfall{suffix}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Plots SHAP générés dans {FIGURES_DIR}/")
        return shap_values, explainer

    except Exception as e:
        print(f"Erreur SHAP : {e}")
        return None, None


# =============================================================
# SAUVEGARDE
# =============================================================

def save_metrics_csv(metrics_list, path='reports/model_comparison.csv'):
    """Sauvegarde le tableau comparatif en CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(metrics_list)
    df.to_csv(path, index=False)
    print(f"Métriques sauvegardées → {path}")
    return df
