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
import random
import logging
from flask import request
import dill
import os 

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Chemins de fichiers
model_filename = "models/best_model.pkl"
explainer_filename = "models/explainer_info.dill"
seuil_filename = "models/meilleur_seuil.txt"
data_path = "https://projet-7-aws.s3.eu-north-1.amazonaws.com/data_agregg_selec.csv"


# Chargement des données
data = pd.read_csv(data_path)

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

def data_reader(file_path, chunk_size=2000000):
    # Ouvre le fichier en utilisant l'itérateur de pandas
    data_reader = pd.read_csv(file_path, chunksize=chunk_size)
    
    # Itére à travers les morceaux
    for chunk in data_reader:
        # Renvoye chaque morceau
        yield chunk

def obtenir_informations_brutes_client(client_id, data_reader):
    # Initialise le dictionnaire pour stocker les informations
    informations_client = {}

    # Étape 1: Obtenez les informations de l'application (application_train et application_test)
    informations_application = obtenir_informations_par_table("Data/sampled/application_train_selected.csv", client_id, data_reader)
    informations_client['informations_application'] = informations_application
    del informations_application

    # Étape 2: Obtenez les informations du bureau
    informations_bureau = obtenir_informations_par_table("Data/sampled/bureau_selected.csv", client_id, data_reader)
    informations_client['informations_bureau'] = informations_bureau

    # Étape 3: Obtenez les informations du bureau_balance en utilisant les informations du bureau
    bureau_balance_info = obtenir_informations_bureau_balance(informations_bureau, data_reader)
    informations_client['informations_bureau_balance'] = bureau_balance_info
    del bureau_balance_info, informations_bureau

    # Étape 4: Obtenez les informations des applications précédentes
    informations_previous_application = obtenir_informations_par_table("Data/sampled/previous_application_selected.csv", client_id, data_reader)
    informations_client['informations_previous_application'] = informations_previous_application

    # Étape 5: Obtenez les informations du POS_CASH_balance en utilisant les informations des applications précédentes
    POS_CASH_balance_info = obtenir_informations_POS_CASH_balance(informations_previous_application, data_reader)
    informations_client['informations_POS_CASH_balance'] = POS_CASH_balance_info
    del POS_CASH_balance_info

    # Étape 6: Obtenez les informations des paiements d'installments en utilisant les informations des applications précédentes
    installments_payments_info = obtenir_informations_installments_payments(informations_previous_application, data_reader)
    informations_client['informations_installments_payments'] = installments_payments_info
    del installments_payments_info, informations_previous_application

    # Étape 7: Obtenez les informations du credit_card_balance
    informations_credit_card_balance = obtenir_informations_par_table("Data/sampled/credit_card_balance_selected.csv", client_id, data_reader)
    informations_client['informations_credit_card_balance'] = informations_credit_card_balance
    del informations_credit_card_balance

    return informations_client


def obtenir_informations_par_table(url, client_id, data_reader):
    informations_table = pd.DataFrame()
    for morceau in data_reader(url):
        morceau_info = morceau[morceau['SK_ID_CURR'] == client_id]
        informations_table = pd.concat([informations_table, morceau_info], ignore_index=True)
    return informations_table.to_dict(orient='records')

def obtenir_informations_bureau_balance(informations_bureau, data_reader):
    bureau_balance_info = pd.DataFrame()
    bureau_balance_url = "Data/sampled/bureau_balance_selected.csv"
    informations_bureau = pd.DataFrame(informations_bureau)
    informations_bureau = informations_bureau.reset_index(drop=True)
    for morceau in data_reader(bureau_balance_url):
        if not informations_bureau.empty:
            morceau = morceau.reset_index(drop=True)
            morceau_info = morceau[morceau['SK_ID_BUREAU'].isin(informations_bureau['SK_ID_BUREAU'])]
        else:
            morceau_info = pd.DataFrame()
        bureau_balance_info = pd.concat([bureau_balance_info, morceau_info], ignore_index=True)
    return bureau_balance_info.to_dict(orient='records')


def obtenir_informations_POS_CASH_balance(informations_previous_application, data_reader):
    POS_CASH_balance_info = pd.DataFrame()
    POS_CASH_balance_url = "Data/sampled/POS_CASH_balance_selected.csv"
    informations_previous_application = pd.DataFrame(informations_previous_application)
    informations_previous_application = informations_previous_application.reset_index(drop=True)
    for morceau in data_reader(POS_CASH_balance_url):
        if not informations_previous_application.empty:
            morceau = morceau.reset_index(drop=True)
            morceau_info = morceau[morceau['SK_ID_PREV'].isin(informations_previous_application['SK_ID_PREV'])]
        else:
            morceau_info = pd.DataFrame()
        POS_CASH_balance_info = pd.concat([POS_CASH_balance_info, morceau_info], ignore_index=True)
    return POS_CASH_balance_info.to_dict(orient='records')


def obtenir_informations_installments_payments(informations_previous_application, data_reader):
    installments_payments_info = pd.DataFrame()
    installments_payments_url = "Data/sampled/installments_payments_selected.csv"
    informations_previous_application = pd.DataFrame(informations_previous_application)
    informations_previous_application = informations_previous_application.reset_index(drop=True)
    for morceau in data_reader(installments_payments_url):
        if not informations_previous_application.empty:
            morceau = morceau.reset_index(drop=True)
            morceau_info = morceau[morceau['SK_ID_PREV'].isin(informations_previous_application['SK_ID_PREV'])]
        else:
            morceau_info = pd.DataFrame()
        installments_payments_info = pd.concat([installments_payments_info, morceau_info], ignore_index=True)
    return installments_payments_info.to_dict(orient='records')


def format_client_data(client_data, donnees_test_path="Data/sampled/test_x_selected_head.csv"):
    """
    Met en forme les données du client pour la prédiction.

    Args:
    - client_data (pd.DataFrame): Les données du client.
    - donnees_test_path (str): Le chemin d'accès vers le fichier de données de test.

    Returns:
    - pd.DataFrame: Les données formatées pour la prédiction.
    """
    #preprocessor = load_function(preprocessing1_function_filename)

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
    
    # Prédictions de probabilité avec le modèle
    proba_predictions = model.predict_proba(new_client_data)[:, 1]  # Assuming binary classification
    print(proba_predictions)
    # Filtrer les prédictions en fonction du seuil
    predicted_class = '1' if proba_predictions > threshold else '0'
    print(predicted_class)
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

def get_client_data(client_id):
    return data.loc[data['SK_ID_CURR'] == client_id]

# Routes Flask
@app.route('/get_importance-caracteristiques/<int:client_id>', methods=['GET'])
def get_importance_caracteristiques(client_id):
    client_id = int(client_id)  # Convertir client_id en entier
    try:
        client_data = get_client_data(client_id)
        formatted_data = format_client_data(client_data)
        importance_caracteristiques = calculer_importance_caracteristiques(explainer, formatted_data)
        return jsonify(importance_caracteristiques)
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/get_target/<client_id>', methods=['GET'])
def get_target(client_id):
    """
    Endpoint pour obtenir la TARGET d'un client spécifique en fonction de son ID.

    Args:
    - client_id (int): L'ID du client.

    Returns:
    - JSON: Les données de TARGET du client.
    """
    data = pd.read_csv("https://projet-7-aws.s3.eu-north-1.amazonaws.com/data_agregg.csv")
    client_id = int(client_id)  # Convertir client_id en entier

    if client_id in data['SK_ID_CURR'].values:
        target = data.loc[data['SK_ID_CURR'] == client_id, 'TARGET'].values[0]
        response = {'client_id': client_id, 'TARGET': target}
        return jsonify(response)
    else:
        return jsonify({'error': 'Client non trouvé'}), 404

@app.route('/get_client_info/<int:client_id>', methods=['GET'])
def get_client_info(client_id):
    """
    Endpoint pour obtenir des informations supplémentaires sur un client en fonction de son ID.

    Args:
    - client_id (int): L'ID du client.

    Returns:
    - JSON: Les informations supplémentaires sur le client.
    """
    data = pd.read_csv("https://projet-7-aws.s3.eu-north-1.amazonaws.com/data_agregg.csv")

    client = data[data['SK_ID_CURR'] == client_id].to_dict(orient='records')

    if client:
        return jsonify(client[0])
    else:
        return jsonify({'error': 'Client non trouvé'}), 404

@app.route('/predict/<int:client_id>', methods=['GET'])
def predict(client_id):
    """
    Endpoint pour effectuer des prédictions sur les nouvelles données client.

    Args:
    - client_id (int): L'ID du client.

    Returns:
    - JSON: Les prédictions du modèle et les probabilités de prédiction.
    """
    client_id = int(client_id)  # Convertir client_id en entier
    try:
        client_data = get_client_data(client_id)
        formatted_data = format_client_data(client_data)
        with open('models/meilleur_seuil.txt', 'r') as file:
            file_content = file.read()
            threshold = float(file_content)
        # Appele la fonction de prédiction
        prediction, probability = predict_new_client(model, formatted_data, threshold)

        return jsonify({'prediction': prediction, 'probability': probability})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@app.route('/informations_client_brut/<int:client_id>', methods=['GET'])
def obtenir_informations_client(client_id):
    # Utiliser la fonction de génération pour lire les données par morceaux
    lecteur_donnees = data_reader
    client_id = int(client_id)  # Convertir client_id en entier
    informations_client = obtenir_informations_brutes_client(client_id, lecteur_donnees)
    return jsonify(informations_client)

def determine_age_group(age):
    if -(age/365) < 30:
        return 'Moins de 30 ans'
    elif -(age/365) <= age < 40:
        return '30-40 ans'
    elif -(age/365) <= age < 50:
        return '40-50 ans'
    else:
        return 'Plus de 50 ans'
    
@app.route('/get_group_info', methods=['GET'])
def get_group_info_raw():
    """
    Endpoint pour obtenir des informations brutes sur un groupe de clients en fonction de certaines colonnes.

    Returns:
    - JSON: Les informations brutes sur le groupe de clients.
    """
    #data = pd.read_csv("https://projet-7-aws.s3.eu-north-1.amazonaws.com/data_agregg_selec.csv")
    data = pd.read_csv("Data/sampled/application_train_selected.csv")
    # Ajoutez ici la logique pour filtrer les données en fonction des colonnes souhaitées (âge, sexe, emploi, etc.)
    # Par exemple, vous pouvez utiliser les arguments de la requête pour spécifier les filtres
    #age_filter = request.args.get('age')
    age_group = request.args.get('age')
    sex_filter = request.args.get('sex')
    job_filter = request.args.get('job')
    # Filtrez les données uniquement sur les colonnes "NAME_CONTRACT_TYPE" et "TARGET"
    filtered_data = data[['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'TARGET', 'DAYS_BIRTH', 'CODE_GENDER', 'OCCUPATION_TYPE', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', ]]

    # Filtrez les données en fonction des filtres spécifiés
    if age_group:
        # Filtrer les données par tranche d'âge
        filtered_data['AGE_GROUP'] = filtered_data['DAYS_BIRTH'].apply(determine_age_group)
        filtered_data = filtered_data[filtered_data['AGE_GROUP'] == age_group]

    if sex_filter:
        filtered_data = filtered_data[filtered_data['CODE_GENDER'] == sex_filter]
    if job_filter:
        filtered_data = filtered_data[filtered_data['OCCUPATION_TYPE'] == job_filter]

    # Convertissez le DataFrame en JSON
    filtered_data_json = filtered_data.to_json(orient='records')

    return jsonify(filtered_data_json)

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
