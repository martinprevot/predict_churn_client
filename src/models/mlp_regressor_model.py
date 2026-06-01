"""Modèle MLP Deep Learning pour la régression (estimation CLV / total_revenue)."""

import numpy as np


def build_mlp_regressor(n_features):
    """
    Construit et compile le MLP pour la régression.

    Architecture : 128 → BN → Dropout → 64 → BN → Dropout → 32 → 1 (linear)
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=(n_features,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # sortie continue pour la régression
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


def get_callbacks_regressor():
    """Callbacks pour la régression : EarlyStopping sur val_loss, ReduceLR."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        mode='min'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    return [early_stop, reduce_lr]


def train_mlp_regressor(X_train, y_train, X_val=None, y_val=None,
                        epochs=100, batch_size=256):
    """
    Entraîne le MLP pour la régression (estimation CLV).

    Returns:
        model, history
    """
    from sklearn.model_selection import train_test_split

    n_features = X_train.shape[1]
    model = build_mlp_regressor(n_features)
    callbacks = get_callbacks_regressor()

    if X_val is None:
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42
        )
    else:
        X_train_fit, y_train_fit = X_train, y_train

    history = model.fit(
        X_train_fit, y_train_fit,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history
