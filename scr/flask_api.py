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
from scr.preprocessing import pre_processing
# from scr.preprocessing.aggregation import aggreger
# from scr.models.feature_importance import explainer_lime
from flask import Flask, jsonify
import pandas as pd
import joblib
import lime
import lime.lime_tabular
import numpy as np
import random
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Déplacer ces informations vers un fichier de configuration + mettre en majuscule 
# Chemins de fichiers
# data_path = 'Data/cleaned/data_agregg2.parquet'
model_filename = "models/best_model.pkl"
# preprocessing1_function_filename = "models/preprocessing_function.pkl"

# Chargement des données
#data = pd.read_parquet('Data/cleaned/data_agregg.parquet', engine='fastparquet') # Mettre le chemin d'accès en variable global 
# data = pd.read_parquet('Data/cleaned/data_agregg2.parquet')

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
    print(pd.DataFrame(client_data_formatted).head())
    print(pd.DataFrame(client_data_formatted).shape)
    del donnees_test
    return pd.DataFrame(client_data_formatted)

def predict_new_client(client_id):
    """
    Effectue des prédictions sur les nouvelles données d'un client.

    Args:
    - client_id (int): L'ID du client.

    Returns:
    - str: La prédiction du modèle.
    """
    data = pd.read_csv("https://projet-7-aws.s3.eu-north-1.amazonaws.com/data_agregg_selec.csv")

    # Filtre sur les données du client à partir de l'ID client
    new_client_data = data.loc[data['SK_ID_CURR'] == client_id]
    del data

    new_client_data = format_client_data(new_client_data)
    print(f"Trying to predict for client ID: {client_id}")
    if pd.DataFrame(new_client_data).empty:
        return {'error': 'Client non trouvé'}
    print(new_client_data.head())
    # Prédictions avec le modèle
    predictions = model.predict(new_client_data)
    return {'predictions': predictions.tolist()}

#def calculer_importance_caracteristiques(client_id, modele, donnees):
    """
    Calcule l'importance des caractéristiques individuelles pour un client donné en utilisant LIME.

    Args:
        client_id (int): Identifiant du client.
        modele: Modèle d'apprentissage automatique préalablement importé.

    Returns:
        dict: Dictionnaire contenant les caractéristiques et leur importance.
    """
    # Filtre sur les données du client à partir de l'ID client
    #new_client_data = data.loc[data['SK_ID_CURR'] == client_id]

    #new_client_data = format_client_data(new_client_data)

    # Utilisation de LimeTabularExplainer
    #explainer = lime.lime_tabular.LimeTabularExplainer(training_data=np.array(donnees), # pas les bonnes données d'entrainement
                                                      #mode='classification',
                                                      #feature_names=list(donnees.columns),
                                                      #class_names=['TARGET'])

    # Explication LIME pour le client spécifié
    #explication = explainer.explain_instance(data_row=np.array(new_client_data),
                                             #predict_fn=modele.predict_proba,
                                             #num_features=len(list(donnees.columns)))

    # Récupération de l'importance des caractéristiques
   #importance_caracteristiques = {}
    #for feature, importance in explication.as_map()[1]:
    #   importance_caracteristiques[feature] = importance

    #return importance_caracteristiques

# Chargement du modèle entraîné et des fonctions
model = load_model(model_filename)

# Routes Flask
#@app.route('/api/importance-caracteristiques/<int:client_id>', methods=['GET'])
#def get_importance_caracteristiques(client_id):
    #try:
        #importance_caracteristiques = calculer_importance_caracteristiques(client_id, model, data)
        #return jsonify(importance_caracteristiques)
    #except Exception as e:
        #return jsonify({'error': str(e)})


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
    - JSON: Les prédictions du modèle.
    """
    client_id = int(client_id)  # Convertir client_id en entier

    try:
        # Appelez la fonction de prédiction
        predictions = predict_new_client(client_id)

        # Retournez les prédictions au format JSON
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/informations_client_brut/<int:client_id>', methods=['GET'])
def obtenir_informations_client(client_id):
    # Utiliser la fonction de génération pour lire les données par morceaux
    lecteur_donnees = data_reader
    client_id = int(client_id)  # Convertir client_id en entier
    informations_client = obtenir_informations_brutes_client(client_id, lecteur_donnees)
    return jsonify(informations_client)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
