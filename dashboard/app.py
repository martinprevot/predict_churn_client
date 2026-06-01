"""
Dashboard Streamlit — Système de prédiction de churn client.
4 pages : Vue d'ensemble | Facteurs de risque | Comparaison modèles | Simulateur
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

# Racine du projet = dossier parent de dashboard/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Chargement des ressources ---

@st.cache_resource
def load_model():
    path = os.path.join(ROOT, 'models', 'best_model.joblib')
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data
def load_data():
    path = os.path.join(ROOT, 'data', 'raw', 'customer_churn_business_dataset.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_metrics():
    path = os.path.join(ROOT, 'reports', 'model_comparison.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_regression_metrics():
    path = os.path.join(ROOT, 'reports', 'regression_model_comparison.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def fig_path(name):
    return os.path.join(ROOT, 'reports', 'figures', name)


def model_path(name):
    return os.path.join(ROOT, 'models', name)


model = load_model()
df = load_data()
metrics_df = load_metrics()

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

# Calcul des probabilités si modèle disponible
if df is not None and model is not None:
    try:
        features = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        df['churn_proba'] = model.predict_proba(features)[:, 1]
        df['risk_level'] = pd.cut(
            df['churn_proba'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        df['revenue_at_risk'] = df['monthly_fee'] * df['churn_proba']
    except Exception as e:
        st.sidebar.warning(f"Erreur calcul probas : {e}")

# --- Navigation ---
page = st.sidebar.selectbox(
    "Navigation",
    ["Vue d'ensemble", "Facteurs de risque", "Comparaison des modèles", "Simulateur client"]
)

# ===========================================================
# PAGE 1 — VUE D'ENSEMBLE
# ===========================================================
if page == "Vue d'ensemble":
    st.title("Vue d'ensemble — KPIs Churn")

    if df is None:
        st.warning("Données non disponibles. Placez le CSV dans `data/raw/`.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total clients", f"{len(df):,}")
    with col2:
        st.metric("Taux de churn", f"{df['churn'].mean()*100:.1f}%")
    with col3:
        if 'revenue_at_risk' in df.columns:
            st.metric("Revenu mensuel à risque", f"{df['revenue_at_risk'].sum():,.0f} €")
    with col4:
        if 'churn_proba' in df.columns:
            st.metric("Clients high-risk (>70%)", f"{(df['churn_proba'] > 0.7).sum():,}")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        counts = df['churn'].value_counts().reset_index()
        counts.columns = ['churn', 'count']
        counts['label'] = counts['churn'].map({0: 'No Churn', 1: 'Churn'})
        fig_pie = px.pie(counts, values='count', names='label',
                         title='Distribution Churn / No-Churn',
                         color_discrete_map={'Churn': '#EF553B', 'No Churn': '#636EFA'})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        if 'churn_proba' in df.columns:
            fig_hist = px.histogram(df, x='churn_proba', nbins=50,
                                    title='Distribution des probabilités de churn',
                                    color_discrete_sequence=['#636EFA'])
            fig_hist.add_vline(x=0.5, line_dash='dash', line_color='red', annotation_text='Seuil 0.5')
            st.plotly_chart(fig_hist, use_container_width=True)

    fig_scatter = px.scatter(
        df.sample(min(2000, len(df)), random_state=42),
        x='monthly_fee', y='tenure_months',
        color='churn_proba' if 'churn_proba' in df.columns else 'churn',
        title='Monthly Fee vs Tenure (coloré par probabilité de churn)',
        color_continuous_scale='RdYlGn_r', opacity=0.6
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        seg_churn = df.groupby('customer_segment')['churn'].mean().reset_index()
        fig_seg = px.bar(seg_churn, x='customer_segment', y='churn',
                         title='Taux de churn par segment client', color='customer_segment')
        st.plotly_chart(fig_seg, use_container_width=True)
    with col_d:
        ct_churn = df.groupby('contract_type')['churn'].mean().reset_index()
        fig_ct = px.bar(ct_churn, x='contract_type', y='churn',
                        title='Taux de churn par type de contrat', color='contract_type')
        st.plotly_chart(fig_ct, use_container_width=True)

    num_corr_cols = ['age', 'tenure_months', 'monthly_fee', 'payment_failures',
                     'support_tickets', 'csat_score', 'nps_score', 'churn']
    corr = df[num_corr_cols].corr()
    fig_heat = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu',
                         title='Heatmap de corrélation interactive', zmin=-1, zmax=1)
    st.plotly_chart(fig_heat, use_container_width=True)


# ===========================================================
# PAGE 2 — FACTEURS DE RISQUE
# ===========================================================
elif page == "Facteurs de risque":
    st.title("Analyse des facteurs de risque")

    if df is None:
        st.warning("Données non disponibles.")
        st.stop()

    col_a, col_b = st.columns(2)

    with col_a:
        p = fig_path('feature_importance_xgboost.png')
        if os.path.exists(p):
            st.image(p, caption='Feature Importance — XGBoost', use_column_width=True)
        else:
            st.info("Entraîner les modèles pour voir la feature importance.")

    with col_b:
        selected_var = st.selectbox("Variable à analyser vs churn", NUMERICAL_FEATURES)
        fig_box = px.box(df, x='churn', y=selected_var, color='churn',
                         labels={'churn': 'Churn (0=Non, 1=Oui)'},
                         title=f'Distribution de {selected_var} par churn',
                         color_discrete_map={0: '#636EFA', 1: '#EF553B'})
        st.plotly_chart(fig_box, use_container_width=True)

    p = fig_path('shap_summary.png')
    if os.path.exists(p):
        st.subheader("SHAP — Importance globale des variables")
        st.image(p, use_column_width=True)

    if 'churn_proba' in df.columns:
        st.subheader("Top 10 clients les plus à risque")
        top_cols = ['churn_proba', 'monthly_fee', 'tenure_months', 'contract_type',
                    'customer_segment', 'payment_failures', 'support_tickets', 'revenue_at_risk']
        top_risky = (
            df.nlargest(10, 'churn_proba')[top_cols]
            .style.format({'churn_proba': '{:.1%}', 'revenue_at_risk': '{:.2f} €',
                           'monthly_fee': '{:.2f} €'})
            .background_gradient(subset=['churn_proba'], cmap='Reds')
        )
        st.dataframe(top_risky, use_container_width=True)


# ===========================================================
# PAGE 3 — COMPARAISON DES MODÈLES
# ===========================================================
elif page == "Comparaison des modèles":
    st.title("Comparaison des modèles")

    tab_clf, tab_reg = st.tabs(["Classification (Churn)", "Régression (CLV)"])

    with tab_clf:
        if metrics_df is None:
            st.info("Lancez `python src/train.py` pour générer les métriques.")
        else:
            st.subheader("Tableau comparatif — Classification")
            display_cols = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'train_time_s']
            st.dataframe(
                metrics_df[display_cols].style.format({
                    'accuracy': '{:.3f}', 'precision': '{:.3f}', 'recall': '{:.3f}',
                    'f1_score': '{:.3f}', 'roc_auc': '{:.3f}', 'train_time_s': '{:.1f}s'
                }).highlight_max(subset=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
                                 color='lightgreen'),
                use_container_width=True
            )

            categories = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            fig_radar = go.Figure()
            for _, row in metrics_df.iterrows():
                vals = [row[c] for c in categories] + [row[categories[0]]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=categories + [categories[0]],
                    fill='toself', name=row['model_name']
                ))
            fig_radar.update_layout(
                title='Radar des métriques',
                polar=dict(radialaxis=dict(visible=True, range=[0.5, 1.0]))
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            p = fig_path('roc_curves.png')
            if os.path.exists(p):
                st.image(p, caption='Courbes ROC', use_column_width=True)
        with col_b:
            p = fig_path('mlp_learning_curves.png')
            if os.path.exists(p):
                st.image(p, caption="Courbes MLP (Classification)", use_column_width=True)

        cm_model = st.selectbox("Matrice de confusion :", ['XGBoost', 'Random Forest', 'Logistic Regression'])
        p = fig_path(f'confusion_matrix_{cm_model.lower().replace(" ", "_")}.png')
        if os.path.exists(p):
            st.image(p, caption=f'Matrice de confusion — {cm_model}', use_column_width=True)

    with tab_reg:
        reg_metrics = load_regression_metrics()
        if reg_metrics is None:
            st.info("Lancez `python src/train_regression.py` pour générer les métriques.")
        else:
            st.subheader("Tableau comparatif — Régression CLV")
            st.dataframe(
                reg_metrics[['model_name', 'mae', 'rmse', 'r2', 'mape', 'train_time_s']]
                .style.format({'mae': '{:.2f}', 'rmse': '{:.2f}', 'r2': '{:.4f}',
                               'mape': '{:.2f}%', 'train_time_s': '{:.1f}s'})
                .highlight_max(subset=['r2'], color='lightgreen')
                .highlight_min(subset=['mae', 'rmse'], color='lightgreen'),
                use_container_width=True
            )

        for img_name, caption in [
            ('regression_metrics_comparison.png', 'Heatmap des métriques'),
            ('regression_predictions_vs_actual.png', 'Réel vs Prédit'),
            ('mlp_learning_curves_regression.png', 'Courbes MLP (Régression)')
        ]:
            p = fig_path(img_name)
            if os.path.exists(p):
                st.image(p, caption=caption, use_column_width=True)


# ===========================================================
# PAGE 4 — SIMULATEUR CLIENT
# ===========================================================
elif page == "Simulateur client":
    st.title("Simulateur client — Prédiction temps réel")

    st.sidebar.markdown("---")
    use_api = st.sidebar.toggle("Appeler l'API REST", value=False,
                                 help="Envoie la requête à FastAPI (doit être lancée séparément)")
    if use_api:
        api_url = st.sidebar.text_input("URL de l'API", value="http://localhost:8000")

    if not use_api and model is None:
        st.warning("Modèle non disponible. Lancez `python src/train.py` ou activez l'appel API.")
        st.stop()

    with st.form("client_form"):
        st.subheader("Profil client")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Démographie & Contrat**")
            age = st.slider("Age", 18, 80, 35)
            gender = st.selectbox("Genre", ['Male', 'Female'])
            customer_segment = st.selectbox("Segment", ['Individual', 'SME', 'Enterprise'])
            contract_type = st.selectbox("Type de contrat", ['Monthly', 'Quarterly', 'Yearly'])
            tenure_months = st.slider("Ancienneté (mois)", 0, 60, 12)
            signup_channel = st.selectbox("Canal d'acquisition", ['Web', 'Mobile', 'Referral'])

        with col2:
            st.markdown("**Usage & Engagement**")
            monthly_logins = st.slider("Connexions/mois", 0, 100, 15)
            weekly_active_days = st.slider("Jours actifs/semaine", 0, 7, 4)
            avg_session_time = st.number_input("Durée session moy. (min)", 0.0, 300.0, 25.0)
            features_used = st.slider("Features utilisées", 0, 20, 5)
            usage_growth_rate = st.slider("Taux de croissance usage", -1.0, 1.0, 0.05, step=0.01)
            last_login_days_ago = st.slider("Dernière connexion (jours)", 0, 90, 5)
            referral_count = st.slider("Parrainages", 0, 20, 1)

        with col3:
            st.markdown("**Finance & Support**")
            monthly_fee = st.number_input("Mensualité (€)", 0.0, 500.0, 79.99)
            total_revenue = st.number_input("Revenu total (€)", 0.0, 50000.0, float(monthly_fee * tenure_months))
            payment_method = st.selectbox("Moyen de paiement", ['Card', 'PayPal', 'Bank Transfer'])
            payment_failures = st.slider("Échecs de paiement", 0, 15, 1)
            discount_applied = st.selectbox("Remise appliquée", ['No', 'Yes'])
            price_increase_last_3m = st.selectbox("Augmentation prix 3 mois", ['No', 'Yes'])
            support_tickets = st.slider("Tickets support", 0, 30, 2)
            avg_resolution_time = st.number_input("Temps résolution moy. (h)", 0.0, 72.0, 10.0)
            complaint_type = st.selectbox("Type de plainte", ['None', 'Billing', 'Service', 'Technical'])
            csat_score = st.slider("Score CSAT", 0.0, 5.0, 3.5, step=0.1)
            escalations = st.slider("Escalades", 0, 10, 0)
            email_open_rate = st.slider("Taux ouverture emails", 0.0, 1.0, 0.5, step=0.01)
            marketing_click_rate = st.slider("Taux clics marketing", 0.0, 1.0, 0.2, step=0.01)
            nps_score = st.slider("NPS Score", 0.0, 100.0, 40.0)
            survey_response = st.selectbox("Réponse enquête", ['Satisfied', 'Neutral', 'Unsatisfied'])

        submitted = st.form_submit_button("Prédire le risque de churn", use_container_width=True)

    if submitted:
        customer_payload = {
            "age": age, "gender": gender, "country": "France", "city": "Paris",
            "customer_segment": customer_segment, "tenure_months": tenure_months,
            "signup_channel": signup_channel, "contract_type": contract_type,
            "monthly_logins": monthly_logins, "weekly_active_days": weekly_active_days,
            "avg_session_time": avg_session_time, "features_used": features_used,
            "usage_growth_rate": usage_growth_rate, "last_login_days_ago": last_login_days_ago,
            "monthly_fee": monthly_fee, "total_revenue": total_revenue,
            "payment_method": payment_method, "payment_failures": payment_failures,
            "discount_applied": discount_applied, "price_increase_last_3m": price_increase_last_3m,
            "support_tickets": support_tickets, "avg_resolution_time": avg_resolution_time,
            "complaint_type": complaint_type, "csat_score": csat_score,
            "escalations": escalations, "email_open_rate": email_open_rate,
            "marketing_click_rate": marketing_click_rate, "nps_score": nps_score,
            "survey_response": survey_response, "referral_count": referral_count
        }

        proba = None
        error_msg = None

        if use_api:
            try:
                import requests
                response = requests.post(f"{api_url}/predict", json=customer_payload, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    proba = result['churn_probability']
                else:
                    error_msg = f"API erreur {response.status_code}: {response.text}"
            except Exception as e:
                error_msg = f"Impossible de contacter l'API : {e}"
        else:
            try:
                input_df = pd.DataFrame([customer_payload])
                proba = float(model.predict_proba(input_df)[0, 1])
            except Exception as e:
                error_msg = f"Erreur modèle : {e}"

        if error_msg:
            st.error(error_msg)
        elif proba is not None:
            revenue_at_risk = monthly_fee * proba

            if proba < 0.3:
                risk_level, color = "Faible", "green"
            elif proba < 0.7:
                risk_level, color = "Moyen", "orange"
            else:
                risk_level, color = "Élevé", "red"

            st.divider()
            col_gauge, col_info = st.columns([1, 1])

            with col_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilité de churn (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 30], 'color': '#90EE90'},
                            {'range': [30, 70], 'color': '#FFD700'},
                            {'range': [70, 100], 'color': '#FF6B6B'}
                        ],
                        'threshold': {'line': {'color': 'black', 'width': 4}, 'value': 50}
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_info:
                st.metric("Niveau de risque", risk_level)
                st.metric("Probabilité de churn", f"{proba:.1%}")
                st.metric("Revenu mensuel à risque", f"{revenue_at_risk:.2f} €")
                st.caption("via API" if use_api else "modèle local")
                if proba > 0.5:
                    st.error("Contacter ce client en priorité !")
                else:
                    st.success("Client stable — surveiller régulièrement.")

            # SHAP local
            try:
                import shap
                p = model_path('xgboost_model.joblib')
                if os.path.exists(p):
                    xgb_pipeline = joblib.load(p)
                    xgb_clf = xgb_pipeline.named_steps['classifier']
                    xgb_prep = xgb_pipeline.named_steps['preprocessor']
                    input_proc = xgb_prep.transform(pd.DataFrame([customer_payload]))

                    explainer = shap.TreeExplainer(xgb_clf)
                    shap_vals = explainer.shap_values(input_proc)[0]

                    from src.preprocessing import get_feature_names
                    feat_names = get_feature_names(xgb_prep)

                    shap_df = pd.DataFrame({'feature': feat_names, 'shap': shap_vals})
                    shap_df['abs_shap'] = shap_df['shap'].abs()
                    top3 = shap_df.nlargest(3, 'abs_shap')

                    st.subheader("Top 3 facteurs de risque (SHAP)")
                    for _, row in top3.iterrows():
                        direction = "↑ augmente" if row['shap'] > 0 else "↓ réduit"
                        st.write(f"- **{row['feature']}** : {direction} le risque (SHAP = {row['shap']:.3f})")
            except Exception:
                pass
