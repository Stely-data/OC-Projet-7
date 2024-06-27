import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.graph_objects as go

# Charger le pipeline de feature engineering
pipeline = joblib.load('pipeline.pkl')

# Charger le seuil optimal
threshold = joblib.load('optimal_threshold.pkl')

# URL de l'API Flask
api_url = 'http://127.0.0.1:8000/predict/'

# Charger le fichier de données
data_file = 'application_test.csv'
data_df = pd.read_csv(data_file)


# Fonction pour envoyer les données à l'API et obtenir une prédiction
def obtenir_prediction(data):
    response = requests.post(api_url, json=data)
    return response.json()


# Interface Streamlit
st.title('Dashboard de Scoring de Crédit')

st.header('Analyse du Dossier')

# Sélectionner ou entrer l'ID du prêt
options_id_pret = data_df['SK_ID_CURR'].astype(str).tolist()
id_pret = st.selectbox('Sélectionnez un numéro de client', [''] + options_id_pret)

# Initialiser les variables pour les features en utilisant les colonnes du DataFrame
if id_pret:
    id_pret = int(id_pret)
    donnees_pret = data_df[data_df['SK_ID_CURR'] == id_pret]
else:
    id_pret = st.text_input('Ou entrez un numéro de client manuellement')
    if id_pret:
        donnees_pret = data_df[data_df['SK_ID_CURR'] == int(id_pret)]
    else:
        donnees_pret = pd.DataFrame()

# Afficher les informations du prêt
if not donnees_pret.empty:
    st.write("Informations sur le client")
    st.write(donnees_pret)
    valeurs_features = {col: str(donnees_pret.iloc[0][col]) for col in donnees_pret.columns if col != 'SK_ID_CURR'}
else:
    st.write("Numéro de client non trouvé. Veuillez entrer un numéro de client existant.")
    st.stop()

# Prédiction
if st.button('Prédire'):
    df = pd.DataFrame([valeurs_features])
    donnees_transformees = pipeline.transform(df)
    prediction = obtenir_prediction(donnees_transformees.to_dict(orient='records')[0])

    score = prediction['probability']
    decision = "Crédit accepté" if prediction['prediction'] == 1 else "Crédit rejeté"

    # Afficher les résultats
    st.subheader('Résultats de la Prédiction')
    st.write(f"Score: {score:.3f}")
    st.write(f"Décision: {decision}")

    # Jauge du score avec seuil
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Score"},
        gauge={
            'axis': {'range': [0, 1]},
            'steps': [
                {'range': [0, threshold * 0.8], 'color': "green"},
                {'range': [threshold * 0.8, threshold], 'color': "yellow"},
                {'range': [threshold, 1], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold}}))

    st.plotly_chart(fig)

    # Afficher l'importance des features
    if st.checkbox('Afficher l\'importance des features'):
        st.write('Importance des Features:')
        for feature, importance in prediction['feature_importance'].items():
            st.write(f"{feature}: {importance:.3f}")

if __name__ == "__main__":
    st.run()
