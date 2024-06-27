# OC-projet-7



pour detection du datadrift avec application_test correspondant au nouveau jeu de données :
python check_data_drift.py data/Sources/application_test.csv


pour tester l'api en local :
python app.py

curl -X POST http://127.0.0.1:8000/predict/ -H "Content-Type: application/json" -d '{"feature1": value1, "feature2": value2, ...}'


