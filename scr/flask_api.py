"""
Flask API pour la prédiction de la TARGET d'un client et la récupération d'informations client.

Ce script utilise Flask pour créer une API Web permettant d'obtenir la TARGET d'un client en fonction de son ID,
de récupérer des informations supplémentaires sur un client et d'effectuer des prédictions sur de nouvelles données client.

Endpoints :
- /get_target/<client_id> : Obtient la TARGET d'un client spécifique en fonction de son ID.
- /get_client_info/<int:client_id> : Récupère des informations supplémentaires sur un client en fonction de son ID.
- /predict/<int:client_id> : Effectue des prédictions sur les nouvelles données client en fonction de son ID.

Fonctions :
- load_model(filename): Charge un modèle à partir d'un fichier utilisant joblib.
- load_function(filename): Charge une fonction à partir d'un fichier utilisant joblib.
- predict_new_client(client_id): Effectue des prédictions sur les nouvelles données d'un client.

Variables globales :
- data : DataFrame contenant les données agrégées.
- model : Modèle entraîné utilisé pour les prédictions.
- preprocessor : Fonction de prétraitement des données utilisée pour les prédictions.
"""
from preprocessing import pre_processing
from flask import Flask, jsonify
import pandas as pd
import joblib
import lime
import lime.lime_tabular
import numpy as np
import logging
from flask import request
import dill

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Chemins de fichiers
model_filename = "models/best_model.pkl"
explainer_filename = "models/explainer_info.dill"
seuil_filename = "models/meilleur_seuil.txt"


# Fonctions de chargement
def load_model(filename):
    """
    Charge un modèle à partir d'un fichier utilisant la bibliothèque joblib.

    Args:
        filename (str): Le chemin vers le fichier du modèle.

    Returns:
        object: Le modèle chargé depuis le fichier.

    Raises:
        RuntimeError: Si une erreur se produit lors du chargement du modèle.
    """
    try:
        return joblib.load(filename)
    except Exception as e:
        raise RuntimeError(f"Error loading model from {filename}: {str(e)}")

def load_function(filename):
    """
    Charge une fonction à partir d'un fichier utilisant la bibliothèque joblib.

    Args:
        filename (str): Le chemin vers le fichier de la fonction.

    Returns:
        object: La fonction chargée depuis le fichier.

    Raises:
        RuntimeError: Si une erreur se produit lors du chargement de la fonction.
    """
    try:
        return joblib.load(filename)
    except Exception as e:
        raise RuntimeError(f"Error loading function from {filename}: {str(e)}")
    
def load_explainer(filename):
    """
    Charge un explainer à partir d'un fichier utilisant la bibliothèque Dill.

    Args:
        filename (str): Le chemin vers le fichier de l'explainer.

    Returns:
        object: L'explainer chargé depuis le fichier.

    Raises:
        RuntimeError: Si une erreur se produit lors du chargement de l'explainer.
    """
    try:
        with open(filename, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading explainer from {filename}: {str(e)}")
    
# Chargement du modèle entraîné et des fonctions
model = load_model(model_filename)
explainer = load_explainer(explainer_filename)


def format_client_data(client_data, donnees_test_path="Data/sampled/test_x_selected_head.csv"):
    """
    Met en forme les données du client pour la prédiction.

    Args:
    - client_data (pd.DataFrame): Les données du client.
    - donnees_test_path (str): Le chemin d'accès vers le fichier de données de test.

    Returns:
    - pd.DataFrame: Les données formatées pour la prédiction.
    """
    # Convertissez le dictionnaire en DataFrame
    client_data = pd.DataFrame.from_dict([client_data])
    
    # Prétraitement des données du client
    client_data_processed = pre_processing.preprocessor_api(client_data, 'lgbm', balance='None')

    # Obtention des données agrégées
    donnees_test = pd.read_csv(donnees_test_path)
    logging.info(donnees_test.head())
    logging.info(donnees_test.shape)
    # Mise au bon format
    client_data_formatted = client_data_processed.reindex(columns=donnees_test.columns, fill_value=0)
    logging.info(pd.DataFrame(client_data_formatted).head())
    logging.info(pd.DataFrame(client_data_formatted).shape)
    del donnees_test
    return pd.DataFrame(client_data_formatted)

def predict_new_client(model, new_client_data, threshold):
    """
    Effectue des prédictions sur les nouvelles données d'un client.

    Args:
    - client_id (int): L'ID du client.

    Returns:
    - str: La prédiction du modèle.
    """
    if pd.DataFrame(new_client_data).empty:
        return {'error': 'Client non trouvé'}
    # Prédictions avec le modèle
    # predictions = model.predict(new_client_data)
    new_client_data = pd.DataFrame(new_client_data)
    # Prédictions de probabilité avec le modèle
    proba_predictions = model.predict_proba(new_client_data)[:, 1]  # Assuming binary classification
    # Filtrer les prédictions en fonction du seuil
    predicted_class = '1' if proba_predictions > threshold else '0'
    # Modifier la structure du dictionnaire de sortie
    return int(predicted_class), float(proba_predictions[0])

def calculer_importance_caracteristiques(explainer, new_client_data):
    """
    Calcule l'importance des caractéristiques individuelles pour un client donné en utilisant LIME.

    Args:
        client_id (int): Identifiant du client.
        modele: Modèle d'apprentissage automatique préalablement importé.

    Returns:
        dict: Dictionnaire contenant les caractéristiques et leur importance.
    """
    # Explication LIME pour le client spécifié
    data_client = new_client_data.to_numpy().astype(int)
    explication = explainer.explain_instance(data_row=data_client[0],
                                             predict_fn=model.predict_proba,
                                             num_features=len(list(new_client_data.columns)))
    #importance_caracteristiques = {}
    #for feature, importance in explication.as_list():
    #    importance_caracteristiques[feature] = importance
    # Création d'un DataFrame pour stocker les importances des caractéristiques
    importance_df = pd.DataFrame(explication.as_list(), columns=['Caracteristique', 'Importance'])
    
    # Ajout d'une colonne pour la valeur absolue de l'importance
    importance_df['Importance_Absolue'] = importance_df['Importance'].abs()
    
    # Tri du DataFrame par l'importance absolue par ordre décroissant
    importance_df = importance_df.sort_values(by='Importance_Absolue', ascending=False)
    print(importance_df)
    # Création d'un dictionnaire à partir du DataFrame pour la valeur de retour
    importance_caracteristiques = dict(zip(importance_df['Caracteristique'], importance_df['Importance']))

    return importance_caracteristiques

# Routes Flask
@app.route('/get_importance-caracteristiques', methods=['POST'])
def get_importance_caracteristiques():
    try:
        features_client = request.get_json()
        formatted_data = format_client_data(features_client)
        print(formatted_data)
        importance_caracteristiques = calculer_importance_caracteristiques(explainer, formatted_data)
        return jsonify(importance_caracteristiques)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint pour effectuer des prédictions sur les nouvelles données client.

    Args:
    - client_id (int): L'ID du client.

    Returns:
    - JSON: Les prédictions du modèle et les probabilités de prédiction.
    """
    try:
        # Récupérer les données JSON de la requête
        client_data = request.get_json()
        formatted_data = format_client_data(client_data)
        with open('models/meilleur_seuil.txt', 'r') as file:
            file_content = file.read()
            threshold = float(file_content)
        # Appele la fonction de prédiction
        prediction, probability = predict_new_client(model, formatted_data, threshold)

        return jsonify({'prediction': prediction, 'probability': probability})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
# Route pour obtenir les informations à partir du fichier texte
@app.route('/get_info_seuil')
def get_info_from_file():
    # Chemin pour obtenir les infos sur le seuil de classification optimal 
    file_path = 'models/meilleur_seuil.txt'

    # Ouverture du fichier
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Renvoye le contenu en tant que réponse
    return file_content

@app.route('/get_global_feature_importance', methods=['GET'])
def get_global_feature_importance():
    # Chemin pour obtenir les infos de feature importance globale
    importance_df = pd.read_csv('feature_imortance_global.csv')
    return jsonify(importance_df.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True, port=5001)
