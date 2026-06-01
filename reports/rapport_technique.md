# Système Intelligent Multi-Modèles pour la Rétention Client et l'Évaluation du Risque de Revenus

**Projet Data Science M2 — EFREI Paris**
**Module : Data Engineering & AI — Bloc 4 RNCP36739**
**Enseignante : Sarah Malaeb | Année : 2025-2026**

---

## Résumé Exécutif

Ce projet présente la conception et le développement d'un système complet de prédiction du churn client dans un environnement SaaS/télécom. À partir d'un dataset de 10 000 clients et 32 variables comportementales, financières et de satisfaction, nous avons implémenté et comparé quatre algorithmes de Machine Learning et Deep Learning pour deux tâches prédictives complémentaires : la **classification binaire du churn** et l'**estimation de la valeur vie client (CLV)**. Le système est déployé sous la forme d'un **dashboard interactif Streamlit** et d'une **API REST FastAPI**. Le modèle final retenu — XGBoost — atteint un **ROC-AUC de 0.820** et un **Recall de 0.838**, permettant de détecter 84 % des clients susceptibles de résilier leur abonnement.

---

## Table des matières

1. Introduction & Contexte Business
2. Présentation du Dataset & Analyse Exploratoire
3. Pipeline de Préparation des Données
4. Modélisation
5. Évaluation Comparative
6. Interprétabilité
7. Dashboard Streamlit
8. API REST FastAPI
9. Conclusion & Perspectives
10. Annexes

---

## 1. Introduction & Contexte Business

### 1.1 Problématique du churn

Dans l'économie des abonnements (SaaS, télécom, services cloud), le **churn** — ou attrition client — désigne la résiliation d'un contrat par un client. Il représente l'une des menaces économiques les plus directes pour ces entreprises : selon les études sectorielles, acquérir un nouveau client coûte en moyenne **5 à 7 fois plus cher** que de fidéliser un client existant.

Au-delà du coût d'acquisition, le churn impacte directement le **Monthly Recurring Revenue (MRR)**, la valeur des actifs clients au bilan et les projections de croissance présentées aux investisseurs. Un taux de churn de 10 % signifie qu'une entreprise doit remplacer un dixième de sa base client chaque année simplement pour maintenir son niveau de revenus.

### 1.2 Enjeux financiers

Pour quantifier l'enjeu sur notre dataset :

- **10 000 clients** avec un revenu mensuel moyen de **34,93 €**
- **Taux de churn observé : 10,2 %** → 1 021 clients à risque
- Revenu mensuel potentiellement perdu : `1021 × 34,93 ≈ 35 664 €/mois`
- Sur 12 mois : **~428 000 € de revenu annuel à risque**

Si un système prédictif permet de retenir seulement 30 % de ces clients via des actions ciblées (remises, appels retention, upgrade d'offre), cela représente une préservation de **~128 000 €/an**.

### 1.3 Objectifs du projet

Ce projet vise à concevoir un **MVP (Minimum Viable Product)** d'intelligence artificielle pour la rétention client, couvrant les dimensions suivantes :

1. **Prédiction binaire du churn** (tâche principale) — identifier les clients susceptibles de résilier
2. **Estimation de la valeur vie client CLV** (tâche bonus régression) — quantifier le revenu à risque par client
3. **Interprétabilité** — expliquer pourquoi un client est classé à risque (SHAP)
4. **Dashboard décisionnel** — interface orientée utilisateur métier (Streamlit)
5. **API REST** — industrialisation du modèle (FastAPI)

### 1.4 Positionnement académique

Ce projet s'inscrit dans le cadre de la certification RNCP36739, Bloc 4 : *"Implémenter des méthodes d'intelligence artificielle pour modéliser et prédire de nouveaux comportements et usages"*. Il couvre les compétences BC4.1 (extraction et préparation de données), BC4.3 (tableaux de bord interactifs) et BC4.4 (modèles prédictifs supervisés).

---

## 2. Présentation du Dataset & Analyse Exploratoire

### 2.1 Source et description générale

Le dataset utilisé est le **Customer Churn Prediction Business Dataset**, disponible sur Kaggle. Il simule un environnement SaaS/télécom avec :

| Caractéristique | Valeur |
|---|---|
| Nombre d'observations | 10 000 clients |
| Nombre de variables | 32 (dont 1 variable cible) |
| Variables numériques | 19 |
| Variables catégorielles | 12 |
| Variable cible principale | `churn` (0 = reste, 1 = résilie) |
| Variable cible secondaire | `total_revenue` (estimation CLV) |
| Valeurs manquantes | Aucune |

### 2.2 Description des variables clés

**Variables comportementales d'usage :**
- `monthly_logins` : nombre de connexions par mois (moy. : 15,7)
- `weekly_active_days` : jours d'activité par semaine (moy. : 4,1)
- `avg_session_time` : durée moyenne de session en minutes (moy. : 20,5 min)
- `features_used` : nombre de fonctionnalités utilisées (moy. : 10,1)
- `usage_growth_rate` : taux de croissance d'utilisation
- `last_login_days_ago` : nombre de jours depuis la dernière connexion

**Variables financières :**
- `monthly_fee` : mensualité en € (moy. : 34,93 €, min : 10 €, max : 100 €)
- `total_revenue` : revenu total généré (moy. : 1 057 €)
- `payment_failures` : échecs de paiement (moy. : 0,5)
- `discount_applied` : remise commerciale appliquée (Yes/No)
- `price_increase_last_3m` : augmentation de tarif dans les 3 derniers mois

**Variables de satisfaction et support :**
- `csat_score` : score de satisfaction client 0-5 (moy. : 3,49)
- `nps_score` : Net Promoter Score -100 à 100 (moy. : 19,1)
- `support_tickets` : tickets support ouverts (moy. : 1,2)
- `avg_resolution_time` : temps de résolution moyen des tickets
- `escalations` : nombre d'escalades
- `complaint_type` : type de plainte (None, Billing, Service, Technical)
- `survey_response` : réponse à l'enquête de satisfaction

**Variables contractuelles et démographiques :**
- `contract_type` : Monthly (50%), Quarterly (30%), Yearly (20%)
- `customer_segment` : Individual (60%), SME (30%), Enterprise (10%)
- `tenure_months` : ancienneté en mois (moy. : 30,2 mois)
- `signup_channel` : canal d'acquisition (Web, Mobile, Referral)
- `age` : âge du client (moy. : 45,9 ans)
- `gender`, `country`, `city`

### 2.3 Distribution de la variable cible

Le dataset présente un **fort déséquilibre de classes** :

| Classe | Effectif | Proportion |
|---|---|---|
| No Churn (0) | 8 979 | **89,8 %** |
| Churn (1) | 1 021 | **10,2 %** |

Ce déséquilibre est réaliste pour ce secteur (les taux de churn annuels en SaaS varient de 5 à 15 %). Il impose des stratégies spécifiques : métriques adaptées (F1, Recall, ROC-AUC plutôt qu'Accuracy seule), paramètre `class_weight='balanced'` pour les modèles sklearn, `scale_pos_weight = 8,79` pour XGBoost.

> **Attention au piège de l'Accuracy** : un modèle qui prédit systématiquement "No Churn" obtiendrait 89,8% d'accuracy sans aucune valeur prédictive.

### 2.4 Analyse des corrélations avec le churn

Les corrélations de Pearson avec la variable cible révèlent :

| Variable | Corrélation | Interprétation |
|---|---|---|
| `csat_score` | **-0.158** | Plus la satisfaction est faible, plus le risque est élevé |
| `tenure_months` | **-0.117** | Les clients récents churnent davantage |
| `monthly_logins` | **-0.098** | L'inactivité précède la résiliation |
| `payment_failures` | **+0.112** | Les incidents de paiement annoncent le départ |
| `age`, `monthly_fee` | ~0.010 | Peu corrélés linéairement |

Ces corrélations relativement faibles (~0.15 max) confirment que la relation entre comportement client et churn est **non-linéaire et multivariée** — justifiant le recours à des modèles d'ensemble (RF, XGBoost) capables de capturer ces interactions complexes.

### 2.5 Analyse des variables catégorielles

**Taux de churn par type de contrat :**
- Monthly : 10,3%
- Quarterly : 9,9%
- Yearly : 10,3%

Contrairement à l'intuition (les contrats mensuels devraient churner plus facilement), les taux sont quasi-identiques. Cela suggère que d'autres variables (satisfaction, engagement) sont plus déterminantes que le type de contrat dans ce dataset.

**Taux de churn par segment :**
- Individual : 10,0%
- SME : 10,9%
- Enterprise : 9,4%

Les PME (SME) churnent légèrement plus — probablement plus sensibles aux tarifs et à la qualité du support.

### 2.6 Score d'engagement client (feature engineering)

Conformément aux recommandations du sujet, nous avons construit un **score d'engagement composite** :

```
EngagementScore = 0.25 × monthly_logins_norm
                + 0.20 × weekly_active_days_norm
                + 0.20 × avg_session_time_norm
                + 0.10 × features_used_norm
                + 0.10 × usage_growth_rate_norm
                + 0.10 × (1 − last_login_days_ago_norm)
```

Ce score révèle une différence significative entre les deux groupes : les clients qui churnent ont un score d'engagement nettement inférieur, validant l'hypothèse que l'inactivité précède la résiliation.

### 2.7 Détection des outliers

Méthode IQR appliquée sur les variables financières :
- `monthly_fee` : outliers détectés au-delà de la borne supérieure (clients premium)
- `total_revenue` : distribution fortement asymétrique à droite (quelques clients très anciens)
- `payment_failures` : distribution concentrée en 0, avec quelques cas extrêmes (jusqu'à 5 échecs)

Ces outliers sont conservés car ils représentent des profils clients réels et informatifs pour la prédiction.

---

## 3. Pipeline de Préparation des Données

### 3.1 Principe anti-data-leakage

La règle fondamentale respectée tout au long du projet :

> **Le preprocessor est exclusivement fitté sur le train set, puis appliqué (transform only) sur le test set.**

Toute violation de ce principe constitue du data leakage et invalide les résultats. Nous utilisons le système de `Pipeline` de scikit-learn qui garantit cette séparation.

### 3.2 Split train/test

```python
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# Train : 8 000 clients | Test : 2 000 clients
# Churn train : 10,2% | Churn test : 10,2% (stratification respectée)
```

Le paramètre `stratify=y` est crucial : il garantit que la proportion de churners est identique dans les deux splits, évitant un biais par déséquilibre accidentel.

### 3.3 Pipeline numérique

Pour les 18 variables numériques :

1. **Imputation** : `SimpleImputer(strategy='median')` — la médiane est robuste aux outliers
2. **Normalisation** : `StandardScaler()` — centrage-réduction (moyenne=0, std=1) nécessaire pour la régression logistique et le MLP ; sans impact sur les arbres de décision

### 3.4 Pipeline catégoriel

Pour les 11 variables catégorielles (gender, country, city, customer_segment, signup_channel, contract_type, payment_method, discount_applied, price_increase_last_3m, complaint_type, survey_response) :

1. **Imputation** : `SimpleImputer(strategy='most_frequent')`
2. **Encodage** : `OneHotEncoder(handle_unknown='ignore')` — 11 variables → ~45 colonnes binaires après encodage

### 3.5 ColumnTransformer

```python
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, NUMERICAL_FEATURES),   # 18 features
    ('cat', categorical_pipeline, CATEGORICAL_FEATURES) # 11 features
])
# Résultat : vecteur de 56 features après transformation
```

### 3.6 Gestion du déséquilibre des classes

Trois stratégies complémentaires :

| Stratégie | Modèle | Paramètre |
|---|---|---|
| Class weights | Logistic Regression, Random Forest | `class_weight='balanced'` |
| Scale pos weight | XGBoost | `scale_pos_weight = 8 979/1 021 = 8.79` |
| Class weights | MLP Keras | `compute_class_weight('balanced', ...)` |

La stratégie `class_weight='balanced'` pénalise davantage les erreurs sur la classe minoritaire (churn=1), forçant le modèle à mieux détecter les churners au prix d'un taux de faux positifs légèrement plus élevé — compromis acceptable dans ce contexte business.

### 3.7 Sérialisation du preprocessor

```python
joblib.dump(preprocessor, 'models/preprocessor.joblib')
```

Le preprocessor fitté est sauvegardé séparément des modèles pour permettre une utilisation en production (transformation de nouvelles données sans re-fitter).

---

## 4. Modélisation

### 4.1 Stratégie globale

Nous avons suivi une progression logique de complexité croissante :

```
Régression Logistique (baseline) → Random Forest → XGBoost → MLP Deep Learning
```

Cette progression permet de mesurer objectivement le gain apporté par chaque niveau de complexité supplémentaire et de justifier le choix final.

**Validation croisée** : `RandomizedSearchCV(cv=5, scoring='roc_auc')` sur RF et XGBoost — 5-fold stratifié pour l'optimisation des hyperparamètres, garantissant la robustesse des résultats.

---

### 4.2 Modèle 1 — Régression Logistique (baseline)

**Principe algorithmique :**
La régression logistique modélise la probabilité d'appartenance à la classe positive via la fonction sigmoïde appliquée à une combinaison linéaire des features :

```
P(churn=1 | X) = σ(β₀ + β₁x₁ + ... + βₙxₙ)
```

C'est un modèle **linéaire** : il suppose que la frontière de décision entre churners et non-churners est un hyperplan dans l'espace des features. Cette hypothèse est forte mais le modèle reste robuste, interprétable et très rapide.

**Choix d'implémentation :**
```python
LogisticRegression(
    class_weight='balanced',  # gestion déséquilibre
    max_iter=1000,            # convergence assurée
    solver='lbfgs',           # adapté aux données multi-classes et petits datasets
    random_state=42
)
```

**Résultats :**

| Métrique | Valeur |
|---|---|
| Accuracy | 66,6% |
| Precision | 18,5% |
| Recall | 66,7% |
| F1-Score | 0.289 |
| ROC-AUC | **0.724** |

**Analyse :** La régression logistique détecte 2 churners sur 3 (Recall 66,7%) mais génère beaucoup de faux positifs (Precision 18,5%). Cela s'explique par la nature linéaire du modèle qui ne peut pas capturer les interactions non-linéaires entre variables comportementales. Il constitue néanmoins un bon point de référence.

---

### 4.3 Modèle 2 — Random Forest

**Principe algorithmique :**
Le Random Forest est un **ensemble de N arbres de décision** entraînés sur des sous-échantillons aléatoires des données (bagging) et des sous-ensembles aléatoires de features à chaque nœud. La prédiction finale est la moyenne des probabilités de tous les arbres, ce qui réduit la variance par rapport à un arbre unique.

Sa force réside dans sa capacité à capturer des **interactions non-linéaires complexes** sans nécessiter de normalisation des données.

**Optimisation des hyperparamètres (RandomizedSearchCV, n_iter=20, cv=5) :**

| Hyperparamètre | Espace de recherche | Valeur optimale |
|---|---|---|
| `n_estimators` | [100, 200, 300] | 200 |
| `max_depth` | [5, 10, 15, None] | None |
| `min_samples_split` | [2, 5, 10] | 5 |
| `min_samples_leaf` | [1, 2, 4] | 2 |
| `max_features` | ['sqrt', 'log2'] | 'sqrt' |

**Résultats :**

| Métrique | Valeur |
|---|---|
| Accuracy | 75,4% |
| Precision | 26,5% |
| Recall | **79,4%** |
| F1-Score | 0.397 |
| ROC-AUC | **0.819** |
| Temps d'entraînement | 16 secondes |

**Analyse :** Gain de +9,5 points de ROC-AUC par rapport à la baseline. Le Random Forest améliore significativement le Recall (79% vs 67%) et le F1-Score. La nature non-linéaire des relations comportement-churn est bien captée.

---

### 4.4 Modèle 3 — XGBoost

**Principe algorithmique :**
XGBoost (Extreme Gradient Boosting) est un algorithme de **boosting** : les arbres sont construits séquentiellement, chaque arbre cherchant à corriger les erreurs du précédent. Contrairement au Random Forest (arbres indépendants), le boosting optimise une fonction de perte de manière itérative avec régularisation L1/L2.

XGBoost introduit plusieurs innovations techniques :
- Approximation de l'arbre avec gains de split exacts
- Gestion native des valeurs manquantes
- Régularisation intégrée pour éviter l'overfitting
- Paramètre `scale_pos_weight` pour le déséquilibre de classes

**Optimisation des hyperparamètres (RandomizedSearchCV, n_iter=20, cv=5) :**

| Hyperparamètre | Espace de recherche | Valeur optimale |
|---|---|---|
| `n_estimators` | [100, 200, 300] | 100 |
| `max_depth` | [3, 5, 7] | 3 |
| `learning_rate` | [0.01, 0.05, 0.1] | 0.05 |
| `subsample` | [0.7, 0.8, 1.0] | 1.0 |
| `colsample_bytree` | [0.7, 0.8, 1.0] | 0.8 |
| `min_child_weight` | [1, 3, 5] | 5 |
| `scale_pos_weight` | calculé | **8.79** |

**Résultats :**

| Métrique | Valeur |
|---|---|
| Accuracy | 71,8% |
| Precision | 24,3% |
| Recall | **83,8%** |
| F1-Score | 0.377 |
| ROC-AUC | **0.820** |
| Temps d'entraînement | 4 secondes |

**Analyse :** XGBoost atteint le **meilleur ROC-AUC (0.820)** et le **meilleur Recall (83,8%)** — il détecte 838 churners sur 1 000. Son `max_depth=3` optimal confirme que les arbres peu profonds combinés en ensemble sont plus efficaces qu'un arbre complexe unique. L'entraînement en 4 secondes (vs 16 pour RF) est un avantage pratique en production.

---

### 4.5 Modèle 4 — MLP Deep Learning (Classification)

**Architecture :**

```
Input(56 features)
  → Dense(128, relu) → BatchNormalization → Dropout(0.3)
  → Dense(64, relu)  → BatchNormalization → Dropout(0.2)
  → Dense(32, relu)
  → Dense(1, sigmoid)  ← sortie probabilité
```

**Compilation :**
```python
optimizer = Adam(learning_rate=0.001)
loss = 'binary_crossentropy'
metrics = ['accuracy', AUC(name='auc')]
```

**Callbacks :**
- `EarlyStopping(monitor='val_auc', patience=10)` → arrêt à l'epoch 23
- `ReduceLROnPlateau(factor=0.5, patience=5)` → LR réduit à 0.0005 à l'epoch 14, puis 0.00025 à l'epoch 22

**Résultats :**

| Métrique | Valeur |
|---|---|
| Accuracy | 75,5% |
| Precision | 22,0% |
| Recall | 55,4% |
| F1-Score | 0.315 |
| ROC-AUC | 0.733 |
| Epochs | 23 (arrêt précoce) |

**Analyse critique :** Le MLP est le modèle le moins performant pour la classification. Plusieurs facteurs l'expliquent :

1. **Volume de données insuffisant** : les réseaux de neurones nécessitent généralement >100 000 exemples pour surpasser les modèles d'ensemble sur données tabulaires. Avec 8 000 exemples d'entraînement, XGBoost a l'avantage.
2. **Overfitting précoce** : l'AUC de validation plafonne à 0.73 alors que l'AUC train continue de monter → l'EarlyStopping joue son rôle protecteur.
3. **Nature tabulaire des données** : les features numériques et catégorielles discrètes ne présentent pas les structures hiérarchiques que le Deep Learning exploite naturellement (images, texte, audio).

> **Conclusion pédagogique** : Le Deep Learning n'est pas systématiquement supérieur. Sur ce type de données, XGBoost surpasse le MLP tout en étant plus rapide et plus interprétable.

---

### 4.6 Tâche bonus — Régression CLV (Estimation du revenu total)

Pour la tâche de régression, nous estimons le `total_revenue` (Customer Lifetime Value) à partir des caractéristiques client sans utiliser cette variable comme feature.

**Trois modèles comparés :**

**Ridge Regression (baseline)** : régularisation L2 pour gérer la multicolinéarité entre `tenure_months` et `monthly_fee`.

**Random Forest Regressor** : même principe que le classifieur mais avec critère MSE au lieu de Gini. `RandomizedSearchCV(cv=5, scoring='r2')`.

**MLP Regressor** :
```
Input → Dense(128) → BN → Dropout(0.3) → Dense(64) → BN → Dropout(0.2) → Dense(32) → Dense(1, linear)
Loss = MSE | Métrique = MAE | 100 epochs
```

**Résultats :**

| Modèle | MAE (€) | RMSE (€) | R² |
|---|---|---|---|
| Ridge Regression | 286 | 401 | 0.852 |
| Random Forest Regressor | 251 | 398 | 0.854 |
| **MLP Regressor** | **36** | **54** | **0.997** |

**Analyse du R²=0.997 du MLP :** Ce résultat exceptionnel s'explique par une relation quasi-déterministe dans les données : `total_revenue ≈ tenure_months × monthly_fee`. Le MLP a appris cette formule arithmétique presque parfaitement. Bien que le score soit impressionnant, il ne reflète pas la complexité d'une vraie régression sur une variable indépendante. Pour le Random Forest (R²=0.854), les résultats sont plus representatifs d'une capacité de généralisation réelle.

---

## 5. Évaluation Comparative

### 5.1 Tableau comparatif — Classification

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC | Temps |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.666 | 0.185 | 0.667 | 0.289 | 0.724 | < 1s |
| Random Forest | 0.754 | 0.265 | 0.794 | 0.397 | 0.819 | 16s |
| **XGBoost** | 0.718 | 0.243 | **0.838** | 0.377 | **0.820** | 4s |
| MLP | 0.755 | 0.220 | 0.554 | 0.315 | 0.733 | 9s |

### 5.2 Analyse des courbes ROC

Les courbes ROC superposent le taux de vrais positifs (Recall) en fonction du taux de faux positifs pour tous les seuils de décision possibles. L'aire sous la courbe (AUC) mesure la capacité discriminante globale indépendamment du seuil choisi.

**Interprétation :**
- **XGBoost et Random Forest** (AUC ~0.82) dominent clairement la baseline aléatoire (AUC=0.5)
- L'écart XGBoost/RF est minime (~0.001) : les deux modèles ont des capacités discriminantes quasi-identiques
- **MLP** (0.733) et **Logistic Regression** (0.724) sont significativement inférieurs
- La Logistic Regression, malgré sa simplicité, n'est pas négligeable (0.724 vs 0.5)

### 5.3 Matrice de confusion — XGBoost

Sur le test set (2 000 clients, dont 204 churners) :

|  | Prédit No Churn | Prédit Churn |
|---|---|---|
| **Réel No Churn** | 1 286 (VP) | 510 (FP) |
| **Réel Churn** | 33 (FN) | 171 (TP) |

- **171 churners correctement identifiés** sur 204 → Recall 83,8%
- **33 churners manqués** → clients qui partiront sans qu'on les contacte (coût élevé)
- **510 faux positifs** → clients contactés inutilement (coût faible : simple email retention)

Dans ce contexte business, **un faux négatif (churner manqué) est bien plus coûteux qu'un faux positif** (client non-churner contacté par erreur). La stratégie `class_weight='balanced'` maximise le Recall au détriment de la Precision, ce qui est le bon arbitrage.

### 5.4 Analyse des erreurs

**Profil des churners manqués (33 faux négatifs) :**
Ces clients se distinguent par des profils atypiques : satisfaction correcte mais comportements d'usage en déclin récent, ou clients dont l'inactivité est récente et pas encore bien captée par le modèle. L'ajout de variables temporelles (trend d'utilisation sur les 3 derniers mois) pourrait réduire ces erreurs.

**Profil des faux positifs (510) :**
Clients classés comme churners mais fidèles. Ils présentent souvent des caractéristiques mixtes : quelques tickets support mais une satisfaction correcte, ou un payment_failure isolé. Une action de rétention proactive sur ces clients ne présente pas de risque majeur (simple contact commercial).

### 5.5 Tableau comparatif — Régression CLV

| Modèle | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Ridge Regression | 286 € | 401 € | 0.852 | 131% |
| Random Forest Regressor | 251 € | 398 € | 0.854 | 74% |
| **MLP Regressor** | **36 €** | **54 €** | **0.997** | 13% |

**Note sur la MAPE élevée de Ridge (131%)** : La MAPE (Mean Absolute Percentage Error) est sensible aux valeurs très faibles. Certains clients avec un `total_revenue` de 10-20 € génèrent des erreurs relatives très élevées, même si l'erreur absolue est faible. Le MAE et le RMSE sont plus fiables ici.

### 5.6 Choix et justification du modèle final

**Modèle retenu pour la classification : XGBoost**

| Critère | Logistic | RF | **XGBoost** | MLP |
|---|---|---|---|---|
| ROC-AUC | 0.724 | 0.819 | **0.820** | 0.733 |
| Recall | 0.667 | 0.794 | **0.838** | 0.554 |
| Interprétabilité | Haute | Moyenne | Haute (SHAP) | Faible |
| Vitesse | Très rapide | Modérée | **Rapide** | Lente |
| Déployabilité | Facile | Facile | **Facile** | Complexe |

XGBoost est retenu car il maximise simultanément le ROC-AUC et le Recall (métrique prioritaire en rétention client), s'entraîne rapidement, se déploie facilement via `joblib` et est entièrement explicable via SHAP.

**Modèle retenu pour la régression : Random Forest Regressor**

Malgré le R²=0.997 du MLP, nous recommandons le Random Forest (R²=0.854) pour la production. Le MLP exploite une relation quasi-déterministe (`revenue ≈ tenure × fee`) plutôt qu'une vraie capacité de généralisation. Le RF est plus robuste sur des distributions de données nouvelles.

---

## 6. Interprétabilité

### 6.1 Feature Importance native — XGBoost

La feature importance native de XGBoost mesure le gain moyen apporté par chaque feature lors des splits. Elle fournit une vision **globale** de l'importance relative des variables.

**Top variables selon XGBoost :**

Les variables les plus importantes identifiées sont principalement :
- **`csat_score`** — la satisfaction client est le signal le plus prédictif du churn
- **`tenure_months`** — l'ancienneté : les clients récents churnent davantage
- **`monthly_logins`** — l'engagement en termes de connexions
- **`last_login_days_ago`** — l'inactivité récente précède la résiliation
- **`payment_failures`** — les incidents de paiement sont un signal d'alarme direct
- **`nps_score`** — le score de recommandation reflète la propension au départ

Ces résultats sont cohérents avec la théorie de la rétention client : la satisfaction et l'engagement sont les meilleurs prédicteurs du churn.

### 6.2 Analyse SHAP (SHapley Additive exPlanations)

SHAP est une méthode d'explicabilité basée sur la théorie des jeux coopératifs (valeurs de Shapley). Elle attribue à chaque feature une contribution précise à la prédiction, pour chaque observation individuellement.

**Avantages par rapport à la feature importance classique :**
- Explications **locales** (client par client) et **globales** (vue d'ensemble)
- Direction de l'effet : indique si une variable augmente ou diminue la probabilité de churn
- Cohérence : la somme des valeurs SHAP explique exactement l'écart entre la prédiction et la valeur de base

#### 6.2.1 SHAP Summary Plot (vue globale)

Le summary plot superpose la distribution des valeurs SHAP pour chaque feature sur l'ensemble du test set. Chaque point représente un client ; la couleur indique la valeur de la feature (rouge = élevée, bleu = faible).

**Interprétations clés :**
- Un `csat_score` faible (bleu) génère des valeurs SHAP fortement positives → augmente le risque de churn
- Un `tenure_months` élevé (rouge) génère des valeurs SHAP négatives → réduit le risque (fidélité)
- Des `payment_failures` élevés (rouge) → augmentent le risque
- Un `monthly_logins` élevé → réduit le risque (client engagé)

#### 6.2.2 SHAP Waterfall Plot (explication individuelle)

Le waterfall plot décompose la prédiction pour un client spécifique. Il montre comment chaque feature pousse la probabilité de churn vers le haut ou vers le bas, à partir de la valeur de base (probabilité moyenne de churn = 10,2%).

**Exemple d'interprétation pour un client à risque :**
> "La probabilité de churn de ce client est de 0,82. Elle est principalement tirée vers le haut par un `csat_score` de 1 (+0.23), un `last_login_days_ago` de 45 jours (+0.18) et 3 `payment_failures` (+0.15). L'ancienneté de 8 mois contribue également positivement (+0.09). Seul le `monthly_fee` élevé tempère légèrement le risque (-0.04)."

### 6.3 Recommandations business actionnables

| Signal détecté par SHAP | Action recommandée |
|---|---|
| `csat_score` < 3 | Appel proactif du service client, offre de résolution personnalisée |
| `last_login_days_ago` > 14 | Email de réengagement avec mise en avant de nouvelles fonctionnalités |
| `payment_failures` ≥ 2 | Contact préventif pour résoudre le problème de paiement, proposition de facilité |
| `tenure_months` < 6 | Programme d'onboarding renforcé, accompagnement early-stage |
| `nps_score` < 0 | Enquête de satisfaction approfondie, escalade vers un Customer Success Manager |
| `support_tickets` ≥ 3 + `avg_resolution_time` élevé | Priorité de résolution, escalade managériale |

**Priorisation financière :**
```
Score_priorité = P(churn) × monthly_fee
```
Cette formule permet de concentrer les efforts de rétention sur les clients à la fois à haut risque ET à haute valeur.

---

## 7. Dashboard Streamlit

### 7.1 Architecture et choix techniques

Le dashboard est développé avec **Streamlit 1.55** et structuré en 4 pages accessibles via une barre de navigation latérale. Il charge le modèle et les données une seule fois en cache (`@st.cache_resource`, `@st.cache_data`) pour des performances optimales.

Les visualisations interactives sont réalisées avec **Plotly** (graphiques zoomables, filtrables, avec tooltips enrichis), offrant une expérience bien supérieure aux graphiques statiques matplotlib.

### 7.2 Page 1 — Vue d'ensemble / KPIs

Cette page affiche en temps réel les indicateurs clés de pilotage :

- **4 métriques (st.metric)** : Total clients, Taux de churn, Revenu mensuel à risque, Clients high-risk
- **Pie chart** : répartition churners / non-churners
- **Histogramme** : distribution des probabilités de churn (seuil 0.5 visualisé)
- **Scatter plot** : Monthly Fee vs Tenure, coloré par probabilité de churn
- **Barplots** : taux de churn par segment client et type de contrat
- **Heatmap de corrélation interactive** : exploration des relations entre variables

### 7.3 Page 2 — Analyse des facteurs de risque

Orientée vers la compréhension des drivers du churn :

- **Feature Importance XGBoost** (image statique générée à l'entraînement)
- **Boxplot interactif** : sélection de variable via selectbox → distribution par churn
- **SHAP Summary Plot** : importance globale des features
- **Table des top 10 clients à risque** : filtrables, avec probabilité, revenu à risque et variables clés

### 7.4 Page 3 — Comparaison des modèles

Vue comparative accessible aux data scientists et aux managers techniques :

- **Tableau comparatif stylé** avec highlight des meilleures valeurs
- **Radar chart Plotly** : comparaison visuelle des 5 métriques par modèle
- **Courbes ROC** et **courbes d'apprentissage MLP**
- **Sélecteur** de matrice de confusion par modèle
- **Onglet Régression** : tableau CLV + graphiques résidus

### 7.5 Page 4 — Simulateur client (prédiction temps réel)

Interface opérationnelle permettant de scorer n'importe quel client :

- **Formulaire complet** (30 champs) : âge, contrat, usage, finance, satisfaction
- **Jauge circulaire Plotly** : probabilité de churn (0-100%) avec code couleur vert/orange/rouge
- **Métriques** : niveau de risque, probabilité, revenu à risque
- **Top 3 facteurs SHAP** : explication personnalisée de la prédiction
- **Toggle API** : bascule entre prédiction locale (joblib) et appel à l'API REST

---

## 8. API REST FastAPI

### 8.1 Architecture

L'API expose le modèle XGBoost comme service d'inférence REST, permettant l'intégration dans n'importe quel système tiers (CRM, marketing automation, ERP).

```
Client (CRM / Dashboard / cURL)
        ↓ HTTP POST /predict
FastAPI (api/main.py)
        ↓ validation Pydantic
Preprocessor (models/preprocessor.joblib)
        ↓ transform
Model (models/best_model.joblib)
        ↓ predict_proba
PredictionResponse (JSON)
```

### 8.2 Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | Statut du service + modèle chargé |
| GET | `/model-info` | Métadonnées du modèle (features, version, seuil) |
| POST | `/predict` | Prédiction pour un client |
| POST | `/predict/batch` | Prédictions pour une liste (max 1 000 clients) |

### 8.3 Schémas Pydantic

La validation des données d'entrée est assurée par **Pydantic v2** avec contraintes métier :

```python
class CustomerFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100)
    monthly_fee: float = Field(..., ge=0)
    csat_score: float = Field(..., ge=0, le=5)
    nps_score: float = Field(..., ge=0, le=100)
    # ... 26 autres champs
```

### 8.4 Exemple de requête/réponse

**Requête :**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "gender": "Male", "customer_segment": "SME",
       "tenure_months": 6, "contract_type": "Monthly",
       "monthly_logins": 3, "csat_score": 1.5,
       "payment_failures": 3, "monthly_fee": 79.99, ...}'
```

**Réponse :**
```json
{
  "churn_probability": 0.847,
  "churn_prediction": 1,
  "risk_level": "High",
  "revenue_at_risk": 67.78,
  "model_version": "1.0.0"
}
```

### 8.5 Gestion des erreurs

| Code HTTP | Cause |
|---|---|
| 200 | Succès |
| 422 | Données invalides (géré automatiquement par Pydantic) |
| 400 | Batch > 1 000 clients |
| 503 | Modèle non chargé |
| 500 | Erreur interne |

La documentation interactive Swagger est disponible à `http://localhost:8000/docs`.

---

## 9. Conclusion & Perspectives

### 9.1 Synthèse des résultats

Ce projet a permis de concevoir un système complet de prédiction du churn couvrant l'intégralité du pipeline data science :

| Composant | Résultat |
|---|---|
| Préprocessing | Pipeline sklearn sans data leakage, 56 features après encodage |
| Classification | XGBoost — ROC-AUC 0.820, Recall 0.838 |
| Régression CLV | Random Forest — R² 0.854, MAE 251 € |
| Interprétabilité | SHAP global + local, recommandations business actionnables |
| Dashboard | Streamlit 4 pages, graphiques Plotly interactifs |
| API REST | FastAPI, 4 endpoints, Pydantic v2, documentation Swagger |

### 9.2 Limites identifiées

1. **Volume de données** : 10 000 observations limitent la capacité du Deep Learning. Un dataset de 100 000+ clients permettrait de mieux exploiter le MLP.
2. **Features temporelles absentes** : l'évolution des comportements dans le temps (trend du nombre de connexions sur 3 mois, vitesse de déclin de l'engagement) enrichirait significativement le modèle.
3. **Dérive des données** : le modèle entraîné sur des données historiques peut se dégrader si les comportements des clients évoluent (concept drift). Un système de monitoring (Evidently, MLflow) serait nécessaire en production.
4. **Relation quasi-déterministe CLV** : `total_revenue ≈ tenure_months × monthly_fee` limite l'intérêt de la tâche de régression. Une variable cible plus complexe (CLV prédictive 12 mois) serait plus pertinente.
5. **Seuil de décision fixe à 0.5** : l'ajustement du seuil selon le ratio coût(FP)/coût(FN) spécifique à l'entreprise permettrait d'optimiser le ROI des actions de rétention.

### 9.3 Perspectives d'amélioration

1. **Feature engineering avancé** : ratio `support_tickets / tenure_months`, score d'engagement composite, variables de tendance (delta d'usage mois sur mois)
2. **Modèles supplémentaires** : LightGBM, CatBoost (gestion native des catégorielles), modèles de survie (Survival Analysis) pour modéliser le temps jusqu'au churn
3. **MLOps** : intégration MLflow pour le tracking des expériences, versioning des modèles, A/B testing des stratégies de rétention
4. **Monitoring en production** : détection de la dérive des données (data drift), alertes automatiques de dégradation des performances
5. **Segmentation client** : clustering (K-Means, DBSCAN) pour identifier des profils homogènes de clients à risque et personnaliser les actions par segment

---

## 10. Annexes

### Annexe A — Structure du projet

```
projet_churn/
├── data/
│   ├── raw/customer_churn_business_dataset.csv
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── train_regression.py
│   ├── evaluate.py
│   └── models/
│       ├── logistic_model.py
│       ├── random_forest_model.py
│       ├── xgboost_model.py
│       ├── mlp_model.py
│       ├── linear_regression_model.py
│       ├── rf_regressor_model.py
│       └── mlp_regressor_model.py
├── models/               ← modèles sérialisés (.joblib, .h5)
├── dashboard/app.py      ← Streamlit
├── api/
│   ├── main.py           ← FastAPI
│   └── schemas.py        ← Pydantic
├── reports/
│   ├── model_comparison.csv
│   ├── regression_model_comparison.csv
│   └── figures/          ← 13 graphiques
├── requirements.txt
└── README.md
```

### Annexe B — Environnement technique

| Outil | Version |
|---|---|
| Python | 3.12.9 |
| scikit-learn | 1.8.0 |
| XGBoost | 3.2.0 |
| TensorFlow/Keras | 2.21.0 |
| SHAP | 0.51.0 |
| Streamlit | 1.55.0 |
| FastAPI | 0.135.2 |
| Pydantic | 2.12.5 |
| pandas | 2.3.3 |

### Annexe C — Commandes de lancement

```bash
# Installation
cd projet_churn && uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Entraînement classification
python src/train.py

# Entraînement régression
python src/train_regression.py

# Dashboard
streamlit run dashboard/app.py

# API REST
uvicorn api.main:app --reload --port 8000
```

### Annexe D — Métriques détaillées

**Classification — Résultats complets :**

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.6655 | 0.1845 | 0.6667 | 0.2891 | 0.7237 |
| Random Forest | 0.7540 | 0.2647 | 0.7941 | 0.3971 | 0.8188 |
| XGBoost | 0.7175 | 0.2432 | 0.8382 | 0.3771 | 0.8196 |
| MLP | 0.7545 | 0.2203 | 0.5539 | 0.3152 | 0.7335 |

**Régression — Résultats complets :**

| Modèle | MAE (€) | RMSE (€) | R² | MAPE |
|---|---|---|---|---|
| Ridge Regression | 286.19 | 400.94 | 0.8516 | 131.0% |
| Random Forest Regressor | 251.02 | 398.18 | 0.8537 | 74.3% |
| MLP Regressor | 36.40 | 54.48 | 0.9973 | 13.3% |

---

*Rapport généré dans le cadre du Projet Data Science M2 — EFREI Paris 2025-2026*
