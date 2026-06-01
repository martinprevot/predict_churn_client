"""
Preprocessing pipeline pour la prédiction de churn et l'estimation CLV.
Pas de data leakage : fit uniquement sur le train set.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# --- Définition des features ---

NUMERICAL_FEATURES = [
    'age', 'tenure_months', 'monthly_logins', 'weekly_active_days',
    'avg_session_time', 'features_used', 'usage_growth_rate',
    'last_login_days_ago', 'monthly_fee', 'payment_failures',
    'support_tickets', 'avg_resolution_time', 'csat_score',
    'escalations', 'email_open_rate', 'marketing_click_rate',
    'nps_score', 'referral_count'
]

CATEGORICAL_FEATURES = [
    'gender', 'country', 'city', 'customer_segment', 'signup_channel',
    'contract_type', 'payment_method', 'discount_applied',
    'price_increase_last_3m', 'complaint_type', 'survey_response'
]

TARGET_CLASSIFICATION = 'churn'
TARGET_REGRESSION = 'total_revenue'   # CLV estimation


def build_preprocessor():
    """Construit le ColumnTransformer (non fitté)."""
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, NUMERICAL_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


def load_data(data_path='data/raw/customer_churn_business_dataset.csv'):
    """Charge et nettoie le dataset."""
    df = pd.read_csv(data_path)
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
    return df


def load_and_split(data_path='data/raw/customer_churn_business_dataset.csv',
                   task='classification', test_size=0.2, random_state=42):
    """
    Charge le dataset, sépare features/cible, split train/test stratifié.

    Args:
        task: 'classification' (churn) ou 'regression' (total_revenue / CLV)
    Returns:
        X_train, X_test, y_train, y_test, df
    """
    df = load_data(data_path)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    if task == 'classification':
        target = TARGET_CLASSIFICATION
        X = df[feature_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
    else:  # regression
        target = TARGET_REGRESSION
        # Pour la régression CLV, on enlève total_revenue des features
        num_feats = [f for f in NUMERICAL_FEATURES if f != TARGET_REGRESSION]
        X = df[num_feats + CATEGORICAL_FEATURES]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    return X_train, X_test, y_train, y_test, df


def build_preprocessor_regression():
    """Preprocessor pour la régression (sans total_revenue dans les features)."""
    num_feats = [f for f in NUMERICAL_FEATURES if f != TARGET_REGRESSION]

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, num_feats),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
    ])
    return preprocessor


def preprocess_data(data_path='data/raw/customer_churn_business_dataset.csv',
                    task='classification', save=True):
    """
    Pipeline complet : chargement → split → fit preprocessor sur train → transform.
    """
    X_train, X_test, y_train, y_test, _ = load_and_split(data_path, task=task)

    preprocessor = (build_preprocessor() if task == 'classification'
                    else build_preprocessor_regression())

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    if save:
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        suffix = '' if task == 'classification' else '_regression'
        joblib.dump(preprocessor, f'models/preprocessor{suffix}.joblib')

        train_df = X_train.copy()
        target_col = TARGET_CLASSIFICATION if task == 'classification' else TARGET_REGRESSION
        train_df[target_col] = y_train.values
        test_df = X_test.copy()
        test_df[target_col] = y_test.values
        train_df.to_csv(f'data/processed/train{suffix}.csv', index=False)
        test_df.to_csv(f'data/processed/test{suffix}.csv', index=False)

        print(f"[{task}] Preprocessor sauvegardé | Train: {X_train_proc.shape} | Test: {X_test_proc.shape}")

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor


def get_feature_names(preprocessor):
    """Récupère les noms de features après transformation."""
    # Détecter les features numériques utilisées
    num_transformer = preprocessor.named_transformers_['num']
    # Récupérer les colonnes du ColumnTransformer
    for name, _, cols in preprocessor.transformers_:
        if name == 'num':
            num_names = list(cols)
        elif name == 'cat':
            cat_names = list(
                preprocessor.named_transformers_['cat']
                .named_steps['encoder']
                .get_feature_names_out(cols)
            )
    return num_names + cat_names


def compute_scale_pos_weight(y_train):
    """Calcule scale_pos_weight pour XGBoost."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return n_neg / n_pos


if __name__ == '__main__':
    X_tr, X_te, y_tr, y_te, prep = preprocess_data(task='classification')
    print(f"Taux de churn train: {y_tr.mean():.3f} | scale_pos_weight: {compute_scale_pos_weight(y_tr):.2f}")

    X_tr_r, X_te_r, y_tr_r, y_te_r, prep_r = preprocess_data(task='regression')
    print(f"CLV (total_revenue) — mean train: {y_tr_r.mean():.1f} | std: {y_tr_r.std():.1f}")
