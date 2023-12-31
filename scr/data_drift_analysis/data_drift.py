"""
Ce script illustre l'analyse de la dérive des données en utilisant la bibliothèque Evidently.
Il compare la dérive des données entre les ensembles de données initiaux et actuels, à la fois pour les données non agrégées et agrégées.

Dépendances :
- pandas
- evidently

Assurez-vous d'installer les bibliothèques nécessaires avec la commande :
pip install pandas evidently

Utilisation :
- Mettez à jour les chemins d'accès dans le module 'config' pour INITIAL_DATA_FILENAME, CURRENT_DATA_FILENAME,
  TEST_X_SELECTED_HEAD_FILENAME et DATA_AGGREG_FILENAME.
- Exécutez le script pour générer des rapports de dérive des données au format HTML pour les ensembles de données non agrégées et agrégées.

Note : Le module 'pre_processing' et le module 'config' doivent être disponibles aux chemins spécifiés.

"""

import sys
sys.path.append("/Users/beatricetapin/Documents/2023/Data Science/Projet_7_Modele_API/")
from config import INITIAL_DATA_FILENAME, CURRENT_DATA_FILENAME, TEST_X_SELECTED_HEAD_FILENAME, DATA_AGGREG_FILENAME
from scr.preprocessing.pre_processing import preprocessor_api
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

# Partie non agrégée 
def analyser_drift_donnees_non_agregees():
    """
    Analyse la dérive des données entre les ensembles de données initiaux et actuels non agrégés.

    - Charge les ensembles de données initiaux et actuels.
    - Calcule la dérive des données en utilisant la bibliothèque Evidently.
    - Enregistre le rapport de dérive des données au format HTML.

    """
    initial_data = pd.read_csv(INITIAL_DATA_FILENAME)
    current_data = pd.read_csv(CURRENT_DATA_FILENAME)

    rapport_drift_donnees = Report(metrics=[DataDriftPreset()])
    rapport_drift_donnees.run(reference_data=initial_data, current_data=current_data)
    rapport_drift_donnees.save_html("rapport_drift_donnees.html")

# Partie agrégée 
def analyser_drift_donnees_agregees():
    """
    Analyse la dérive des données entre les ensembles de données initiaux et actuels agrégés.

    - Charge toutes les données agrégées.
    - Sépare les données initiales et actuelles.
    - Applique le prétraitement à l'aide de 'preprocessor_api'.
    - Calcule la dérive des données en utilisant la bibliothèque Evidently.
    - Enregistre le rapport de dérive des données au format HTML.

    """
    # Chargement des données agrégées
    toutes_donnees_agregees = pd.read_csv(DATA_AGGREG_FILENAME)

    # Séparation des données initiales et actuelles
    donnees_initiales_agregees = toutes_donnees_agregees[toutes_donnees_agregees['TARGET'].notna()]
    donnees_actuelles_agregees = toutes_donnees_agregees[toutes_donnees_agregees['TARGET'].isna()]

    # Prétraitement des données
    donnees_initiales = preprocessor_api(donnees_initiales_agregees, 'lgbm', balance=None)
    donnees_actuelles = preprocessor_api(donnees_actuelles_agregees, 'lgbm', balance=None)

    # Obtention des données agrégées
    donnees_test = pd.read_csv(TEST_X_SELECTED_HEAD_FILENAME)

    # Mise au bon format
    donnees_initiales_formatees = donnees_initiales.reindex(
            columns=donnees_test.columns,
            fill_value=0,
            )

    donnees_actuelles_formatees = donnees_actuelles.reindex(
            columns=donnees_test.columns,
            fill_value=0,
            )
    
    print(donnees_initiales_formatees.shape)
    print(donnees_actuelles_formatees.shape)

    rapport_drift_donnees = Report(metrics=[DataDriftPreset()])
    rapport_drift_donnees.run(reference_data=donnees_initiales_formatees, current_data=donnees_actuelles_formatees)
    rapport_drift_donnees.save_html("rapport_drift_donnees_agregees.html")

if __name__ == "__main__":
    analyser_drift_donnees_non_agregees()
    analyser_drift_donnees_agregees()
