"""Modèle MLP Deep Learning avec TensorFlow/Keras."""

import numpy as np


def build_mlp_model(n_features):
    """
    Construit et compile le réseau de neurones.

    Architecture : 128 → BN → Dropout → 64 → BN → Dropout → 32 → 1 (sigmoid)
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.metrics import AUC

    model = Sequential([
        Input(shape=(n_features,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )

    return model


def get_callbacks():
    """Retourne les callbacks EarlyStopping et ReduceLROnPlateau."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    early_stop = EarlyStopping(
        monitor='val_auc',
        patience=10,
        restore_best_weights=True,
        mode='max'
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    return [early_stop, reduce_lr]


def compute_class_weights(y_train):
    """Calcule les poids de classes pour compenser le déséquilibre."""
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))


def train_mlp(X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=256):
    """
    Entraîne le MLP.

    Returns:
        model, history
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    n_features = X_train.shape[1]
    model = build_mlp_model(n_features)
    callbacks = get_callbacks()
    class_weights = compute_class_weights(y_train)

    if X_val is None:
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
        )
    else:
        X_train_fit, y_train_fit = X_train, y_train

    history = model.fit(
        X_train_fit, y_train_fit,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    return model, history
