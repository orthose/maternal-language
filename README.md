# Introduction
Projet universitaire de prédiction de la langue maternelle basée sur un dataset du TOEFL.

# Exécution
Pour lancer les notebooks.
```shell
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

Pour lancer le script chargé de faire les prédictions sur le dataset de test.
```shell
source venv/bin/activate
python predict_data.py
```
