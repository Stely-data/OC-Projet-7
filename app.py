import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from models import (
    PipelineWithDriftDetection, ThresholdClassifier,
    FeatureEngineeringPipelineWrapper, clean_column_names,
    replace_infinite_values, prepare_pip_data
)

# Meilleurs paramètres pour le modèle CatBoost
best_params = {
    'depth': 6,
    'iterations': 1000,
    'l2_leaf_reg': 2,
    'learning_rate': 0.03
}

# Initialisation du modèle CatBoost avec les meilleurs paramètres
catboost_model = CatBoostClassifier(
    verbose=0,
    thread_count=-1,
    task_type="GPU",
    **best_params
)

app = Flask(__name__)

# Chemin pour sauvegarder le fichier joblib
joblib_file = 'pipeline_credit_scoring_with_drift_detection.joblib'

# Fonction pour re-sauvegarder le modèle sur Heroku
def save_model_on_heroku():
    # Chargez les données d'entraînement
    application_train = pd.read_csv('data/Sources/application_train.csv')

    # Initialisez et entraînez le modèle ici
    pipeline = Pipeline([
        ('preprocessor', FunctionTransformer(prepare_pip_data, validate=False)),
        ('classifier', ThresholdClassifier(catboost_model, threshold=0.49))
    ])

    pipeline_with_drift = PipelineWithDriftDetection(pipeline, application_train)
    pipeline_with_drift.fit(application_train)

    # Sauvegardez le modèle
    joblib.dump(pipeline_with_drift, joblib_file)

# Essayez de charger le fichier joblib existant
try:
    pipeline = joblib.load(joblib_file)
except KeyError:
    save_model_on_heroku()
    pipeline = joblib.load(joblib_file)


@app.route("/")
def read_root():
    return jsonify({"message": "Credit Scoring API"})


@app.route("/predict/", methods=['POST'])
def predict():
    data = request.get_json(force=True)
    client_id = data.get('client_id')

    if client_id is None:
        return jsonify({"error": "client_id is required"}), 400

    # Charger les données du client à partir de l'ID
    client_data = get_client_data(client_id)

    if client_data.empty:
        return jsonify({"error": "client_id not found"}), 404

    # Appliquer le preprocessing
    client_data_transformed = pipeline.pipeline.named_steps['preprocessor'].transform(client_data)

    # Prédire le score et la probabilité
    score = pipeline.pipeline.named_steps['classifier'].base_classifier.predict_proba(client_data_transformed)[:, 1][0]
    prediction = pipeline.pipeline.named_steps['classifier'].base_classifier.predict(client_data_transformed)[0]

    # Calculer la feature importance locale (SHAP)
    explainer = shap.Explainer(pipeline.pipeline.named_steps['classifier'].base_classifier)
    shap_values = explainer(client_data_transformed)
    feature_importance = shap_values.values[0].tolist()
    feature_names = client_data_transformed.columns.tolist()
    #feature_importance_dict = dict(zip(feature_names, feature_importance))
    feature_importance_dict = {str(k): float(v) for k, v in zip(feature_names, feature_importance)}

    return jsonify({
        "client_id": int(client_id),
        "score": float(score),
        "prediction": int(prediction),
        "feature_importance": feature_importance_dict
    })


def get_client_data(client_id):
    # Charger les données du client en utilisant l'ID
    df = pd.read_csv('data/Sources/application_test.csv')
    client_data = df[df['SK_ID_CURR'] == client_id]
    return client_data


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8000)
