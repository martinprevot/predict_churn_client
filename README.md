# Churn Prediction System

Système intelligent de prédiction de churn client pour environnement SaaS/Télécom.

## Dataset

Télécharger `customer_churn.csv` depuis Kaggle :
https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset

Placer le fichier dans `data/raw/customer_churn.csv`.

## Installation

```bash
pip install -r requirements.txt
```

## Entraînement des modèles

```bash
python src/train.py
```

## Dashboard

```bash
streamlit run dashboard/app.py
```

## API

```bash
cd api && uvicorn main:app --reload --port 8000
# Documentation : http://localhost:8000/docs
```

## Structure du projet

```
projet_churn/
├── data/
│   ├── raw/                    ← customer_churn.csv
│   └── processed/              ← train.csv, test.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── models/
│       ├── logistic_model.py
│       ├── random_forest_model.py
│       ├── xgboost_model.py
│       └── mlp_model.py
├── models/                     ← modèles sérialisés
├── dashboard/app.py            ← Streamlit (4 pages)
├── api/
│   ├── main.py                 ← FastAPI
│   └── schemas.py
└── reports/figures/            ← graphiques exportés
```

## Modèles entraînés

| Modèle | ROC-AUC | F1-Score |
|---|---|---|
| Régression Logistique | - | - |
| Random Forest | - | - |
| XGBoost | - | - |
| MLP (Deep Learning) | - | - |

> Remplir après entraînement avec les résultats réels.

## Ordre d'exécution

1. Placer `customer_churn.csv` dans `data/raw/`
2. `python src/train.py` — entraîne les 4 modèles
3. `streamlit run dashboard/app.py` — lance le dashboard
4. `uvicorn api.main:app --reload` — lance l'API
