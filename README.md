# OC-projet-7 : Projet de Scoring de Crédit

## Objectif du Projet

L'objectif de ce projet est de développer un outil de scoring de crédit pour la société "Prêt à dépenser", une entreprise financière offrant des crédits à la consommation aux personnes ayant peu ou pas d'historique de prêt. Cet outil de scoring calcule la probabilité qu'un client rembourse son crédit et classifie la demande en crédit accordé ou refusé. Le projet utilise diverses sources de données pour élaborer un algorithme de classification fiable et transparent.

## Structure du Projet

Le projet est organisé en plusieurs dossiers et fichiers pour structurer le code, les données, et les résultats des analyses. Voici un aperçu de la structure des dossiers :

### Racine du Projet

- `app.py` : Script principal de l'API Flask pour servir le modèle de prédiction.
- `check_data_drift.py` : Script pour vérifier la dérive des données (data drift) et déterminer si un réentraînement du modèle est nécessaire.
- `feature_pipeline.py` : Script de pipeline pour l'ingénierie des features et la préparation des données.
- `streamlit_app.py` : Script de l'application Streamlit pour le tableau de bord de scoring de crédit.
- `Procfile` : Fichier de configuration pour le déploiement de l'API Flask sur Heroku.
- `README.md` : Ce fichier.
- `requirements.txt` : Liste des dépendances Python nécessaires au projet.
- `pipeline.pkl` : Pipeline de préparation des données sauvegardé.
- `best_model.pkl` : Modèle de prédiction entraîné et sauvegardé.
- `optimal_threshold.pkl` : Seuil optimal pour la classification sauvegardé.

### Dossier `data`

Ce dossier contient les données utilisées pour l'entraînement et la validation du modèle.

- `data/Sources/` : Contient toutes les sources de données brutes.
  - `application_train.csv` : Données d'entraînement.
  - `application_test.csv` : Données de test.
  - `bureau.csv` : Données supplémentaires sur les bureaux de crédit.
  - `bureau_balance.csv` : Équilibres des bureaux de crédit.
  - `credit_card_balance.csv` : Équilibres des cartes de crédit.
  - `installments_payments.csv` : Données sur les paiements par versements.
  - `POS_CASH_balance.csv` : Équilibres POS CASH.
  - `previous_application.csv` : Données sur les demandes de crédit précédentes.
  - `HomeCredit_columns_description.csv` : Description des colonnes des données Home Credit.
  - `data_drift_report.html` : Rapport de dérive des données généré par le script `check_data_drift.py`.

### Dossier `data/Sources/param`

Ce sous-dossier contient les paramètres et les objets nécessaires pour le pipeline de données.

- `label_encoders.pkl` : Encoders de labels sauvegardés.
- `train_columns.pkl` : Colonnes de l'ensemble de données d'entraînement sauvegardées.

### Fichiers Jupyter Notebooks

- `Deveau_Estelle_1_notebook_EDA_052024.ipynb` : Notebook pour l'analyse exploratoire des données (EDA).
- `Deveau_Estelle_2_notebook_Modelisation_052024.ipynb` : Notebook pour la modélisation, y compris l'entraînement du modèle, la validation et l'optimisation des hyperparamètres.

## Instructions pour l'Exécution

### 1. Déploiement de l'API Flask

1. Installez les dépendances :

   pip install -r requirements.txt


2. Déployez l'API sur Heroku :

   heroku create project7-creditscoring

   git push heroku Rework

### 2. Exécution de l'Application Streamlit

1. Installez les dépendances :

   pip install -r requirements.txt


2. Lancez l'application Streamlit :

   streamlit run streamlit_app.py

### 3. Vérification de la Dérive des Données

1. Exécutez le script de vérification de la dérive des données :

   python check_data_drift.py data/Sources/application_test.csv



