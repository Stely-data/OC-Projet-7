import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import argparse

# Chemins des fichiers par défaut
initial_train_data_path = "data/Sources/application_train.csv"
report_path = "data/data_drift_report.html"


# Fonction pour vérifier le data drift et déterminer si le modèle doit être réentraîné
def check_data_drift(initial_data_path, new_data_path, report_path):
    # Vérifier l'existence des fichiers
    if not os.path.exists(initial_data_path):
        raise FileNotFoundError(f"Le fichier {initial_data_path} n'existe pas.")
    if not os.path.exists(new_data_path):
        raise FileNotFoundError(f"Le fichier {new_data_path} n'existe pas.")

    # Charger les données d'entraînement initiales et les nouvelles données
    initial_data = pd.read_csv(initial_data_path)
    new_data = pd.read_csv(new_data_path)

    # Supprimer la colonne 'TARGET' si elle n'existe pas dans l'un des deux DataFrames
    if 'TARGET' not in initial_data.columns or 'TARGET' not in new_data.columns:
        initial_data = initial_data.drop(columns=['TARGET'], errors='ignore')
        new_data = new_data.drop(columns=['TARGET'], errors='ignore')

    # Créer un rapport pour la détection de la dérive des données
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(current_data=new_data, reference_data=initial_data, column_mapping=None)

    # Sauvegarder le rapport en HTML
    data_drift_report.save_html(report_path)
    print(f"Rapport de data drift sauvegardé sous : {report_path}")

    # Obtenir les résultats du rapport en tant que dictionnaire
    report_dict = data_drift_report.as_dict()


    # Déterminer si le data drift est significatif
    for metric in report_dict['metrics']:
        print(f"Inspecting metric: {metric}")
        if metric['metric'] == 'DatasetDriftMetric':
            drift_share = metric['result']['drift_share']
            if drift_share > 0.5:  # Seuil à définir selon les besoins
                print("Significant data drift detected. Model re-training may be required.")
                print(f"Please review the data drift report: {report_path}")
                return True
            else:
                print("No significant data drift detected. Model re-training is not required.")
                return False

    print("Dataset Drift metric not found in the report.")
    return False


# Fonction principale pour exécuter le script
def main(new_data_path):
    # Convertir le chemin en chemin relatif basé sur le répertoire courant du script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    new_data_path = os.path.join(script_dir, new_data_path)
    initial_train_data_path_abs = os.path.join(script_dir, initial_train_data_path)
    report_path_abs = os.path.join(script_dir, report_path)

    retrain_needed = check_data_drift(initial_train_data_path_abs, new_data_path, report_path_abs)


if __name__ == "__main__":
    # Argument parser pour obtenir le chemin du fichier new_data.csv
    parser = argparse.ArgumentParser(description='Check for data drift and determine if model retraining is required.')
    parser.add_argument('new_data_path', type=str, help='Path to the new data CSV file')
    args = parser.parse_args()

    # Exécuter la vérification du data drift avec le chemin spécifié
    main(args.new_data_path)
