from flask import Flask, request, jsonify
import joblib
import pandas as pd
import shap


app = Flask(__name__)

# Charger le modèle et le seuil
model = joblib.load('best_model.pkl')
optimal_threshold = joblib.load('optimal_threshold.pkl')


@app.route("/")
def read_root():
    return jsonify({"message": "Credit Scoring API"})


@app.route("/predict/", methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])

    # Vérifier si le DataFrame a plus d'une ligne
    if df.shape[0] != 1:
        return jsonify({'error': 'Le DataFrame doit contenir exactement une seule ligne.'}), 400

    # Prédiction
    proba = model.predict_proba(df)[:, 1]
    prediction = (proba >= optimal_threshold).astype(int)

    # Calculer l'importance des features locales avec SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    # Obtenir l'importance des features sous forme de dictionnaire
    feature_importance = dict(zip(df.columns, shap_values[0][0]))

    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(proba[0]),
        'feature_importance': feature_importance  # Importance des features pour la prédiction donnée
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
