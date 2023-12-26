"""
Projet 7 - Modèle API - Tests Unitaires

Ce script contient des tests unitaires pour les fonctions du projet, notamment celles
de prétraitement, d'entraînement de modèle, et d'API Flask.
Les modules testés sont les suivants :

- Module de prétraitement :
    - select_features
    - split_data
    - handle_missing_values
    - clean_feature_names
    - one_hot_encoder

- Module d'entraînement de modèle :
    - custom_scorer
    - find_optimal_threshold
    - create_pipeline

- Module API Flask :
    - predict
    - get_info_seuil
    - get_global_feature_importance
    - load_model
    - load_explainer
    - format_client_data
    - predict_new_client

Tests unitaires :
    - test_select_features: Teste la fonction select_features.
    - test_split_data: Teste la fonction split_data.
    - test_handle_missing_values: Teste la fonction handle_missing_values.
    - test_clean_feature_names: Teste la fonction clean_feature_names.
    - test_one_hot_encoder: Teste la fonction one_hot_encoder.
    - test_custom_scorer: Teste la fonction custom_scorer.
    - test_find_optimal_threshold: Teste la fonction find_optimal_threshold.
    - test_create_pipeline: Teste la fonction create_pipeline.
    - test_predict: Teste la fonction predict de l'API Flask.
    - test_get_info_seuil: Teste la fonction get_info_seuil de l'API Flask.
    - test_get_global_feature_importance: Teste la fonction 
    get_global_feature_importance de l'API Flask.
    - test_load_model: Teste la fonction load_model.
    - test_load_explainer: Teste la fonction load_explainer.
    - test_format_client_data: Teste la fonction format_client_data.
    - test_predict_new_client: Teste la fonction predict_new_client.
"""

import sys
import numpy as np
import pandas as pd
import pytest
from unittest import mock
from lightgbm import LGBMClassifier

#sys.path.append("/Users/beatricetapin/Documents/2023/Data Science/Projet_7_Modele_API/")
#sys.path.append("/Users/beatricetapin/Documents/2023/Data Science/Projet_7_Modele_API/scr/")

from .flask_api import (
    load_model,
    load_explainer,
    format_client_data,
    predict_new_client,
    app,
)

from .preprocessing.pre_processing import (
    select_features,
    split_data,
    handle_missing_values,
    clean_feature_names,
)

from .preprocessing.aggregation import one_hot_encoder

from ..models.model_training import (
    custom_scorer,
    find_optimal_threshold,
    create_pipeline,
)


# Test de la fonction select_features
def test_select_features():
    """
    Teste la fonction select_features.

    Vérifie que la fonction sélectionne correctement les caractéristiques 
    du DataFrame en fonction du seuil.
    """
    df = pd.DataFrame({
        'feature1': [1, 2, 3, None, 5],
        'feature2': [1, None, 3, 4, 5],
        'TARGET': [0, 1, 0, 1, 0]
    })

    df_selected = select_features(df, threshold=0.5)

    assert set(df_selected.columns) == {'feature1', 'feature2', 'TARGET'}
    assert 'TARGET' in df_selected.columns

# Test de la fonction split_data
def test_split_data():
    """
    Teste la fonction split_data.

    Vérifie que la fonction divise correctement le DataFrame en ensembles 
    d'entraînement et de test.
    """
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 3, 4, 5, 6],
        'TARGET': [0, 1, 0, 1, 0]
    })

    train_x, train_y, test_x, test_y = split_data(df)

    assert len(train_x) + len(test_x) == len(df)
    assert len(train_y) + len(test_y) == len(df)

# Test de la fonction handle_missing_values
def test_handle_missing_values():
    """
    Teste la fonction handle_missing_values.

    Vérifie que la fonction supprime correctement les lignes avec des valeurs manquantes 
    et maintient la cohérence entre les caractéristiques et les cibles.
    """
    # Création  d'un DataFrame pour le test avec des valeurs manquantes
    df = pd.DataFrame({
        'feature1': [1, 2, 3, None, 5],
        'feature2': [1, None, 3, 4, 5],
        'TARGET': [0, 1, 0, 1, 0]
    })

    # Séparation les caractéristiques et les cibles
    train_x = df[['feature1', 'feature2']]
    test_x = df[['feature1', 'feature2']]
    train_y = df['TARGET']
    test_y = df['TARGET']

    # Appele la fonction handle_missing_values
    train_x, test_x, train_y, test_y = handle_missing_values(train_x, test_x, train_y, test_y)

    # Verifie que les lignes avec des valeurs manquantes ont été supprimées
    assert len(train_x) == 3
    assert len(test_x) == 3

    # Verifie que la cohérence entre les caractéristiques et les cibles est maintenue
    assert len(train_y) == 3
    assert len(test_y) == 3

    # Verifie que les indices sont alignés
    assert train_x.index.equals(train_y.index)
    assert test_x.index.equals(test_y.index)

# Test de la fonction clean_feature_names
def test_clean_feature_names():
    """
    Teste la fonction clean_feature_names.

    Vérifie que la fonction nettoie correctement les noms de colonnes du DataFrame.
    """
    df = pd.DataFrame({
        'feature 1': [1, 2, 3],
        'feature,2': [4, 5, 6]
    })

    clean_feature_names(df)

    assert set(df.columns) == {'feature_1', 'feature_2'}

# Fonction de test pour one_hot_encoder
def test_one_hot_encoder():
    """
    Teste la fonction one_hot_encoder.

    Vérifie que la fonction applique correctement le codage one-hot aux 
    colonnes catégorielles du DataFrame.
    """
    df = pd.DataFrame({
        'A': ['cat1', 'cat2', 'cat1', None],
        'B': [1, 2, 3, 0]
    })

    encoded_df, new_columns = one_hot_encoder(df)
    print(encoded_df)
    assert 'A_cat1' in encoded_df.columns
    assert 'A_cat2' in encoded_df.columns
    assert 'A_nan' in encoded_df.columns
    assert 'B' in encoded_df.columns
    assert len(new_columns) == 3

@pytest.fixture
def sample_data():
    """
    Fixture pour générer des données d'exemple pour les tests.

    Returns:
        tuple: Un tuple contenant un DataFrame de caractéristiques (X) et un tableau de cibles (y).
    """
    # Génére des données d'exemple pour les tests
    np.random.seed(42)
    x = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
    y = np.random.randint(0, 2, 100)
    return x, y

def test_custom_scorer():
    """
    Teste la fonction custom_scorer.

    Vérifie que le score est correctement calculé en comparant les prédictions réelles et prédites.
    """
    # Teste custom_scorer avec des prédictions parfaites
    y_true = np.array([0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 0])
    score = custom_scorer(y_true, y_pred)
    assert score == 0.0, "Le score devrait être 0.0 avec des prédictions parfaites."

def test_find_optimal_threshold(sample_data):
    """
    Teste la fonction find_optimal_threshold.

    Vérifie que le seuil optimal est dans la plage valide pour un ensemble de données donné.
    """
    _, y = sample_data
    np.random.seed(42)
    probas = np.random.rand(len(y))
    threshold = find_optimal_threshold(y, probas)
    assert 0.0 <= threshold <= 1.0, "Le seuil optimal doit être entre 0 et 1."

def test_create_pipeline():
    """
    Teste la fonction create_pipeline.

    Vérifie que la fonction crée correctement un pipeline avec le modèle spécifié.
    """
    # Teste la création du pipeline avec un modèle de régression logistique
    model = LGBMClassifier()
    pipeline = create_pipeline(model)
    assert len(pipeline.steps) == 1, "Le pipeline devrait avoir deux étapes."

@pytest.fixture
def client():
    """
    Fixture pour créer un client de test Flask.

    Returns:
        FlaskClient: Un client de test Flask.
    """
    with app.test_client() as client:
        yield client

def test_predict(client):
    """
    Teste la fonction predict.

    Vérifie que la fonction renvoie une prédiction correcte en réponse à une requête POST simulée.
    """
    # Envoi d'une requête POST avec des données JSON simulées
    response = client.post('/predict', json={"feature1": 1, "feature2": 2})

    # Vérification du code de statut HTTP
    assert response.status_code == 200

    # Vérification du format de la réponse JSON
    data = response.get_json()
    assert isinstance(data, dict)
    assert "prediction" in data
    assert "probability" in data

def test_get_info_seuil(client):
    """
    Teste la fonction get_info_seuil.

    Vérifie que la fonction renvoie une réponse correcte en réponse à une requête GET simulée.
    """
    # Envoi d'une requête GET
    response = client.get('/get_info_seuil')

    # Vérification du code de statut HTTP
    assert response.status_code == 200

    # Vérification du contenu de la réponse
    assert isinstance(response.get_data(as_text=True), str)

def test_get_global_feature_importance(client):
    """
    Teste la fonction get_global_feature_importance.

    Vérifie que la fonction renvoie une liste de caractéristiques globales avec leurs importances.
    """
    # Envoi d'une requête GET
    response = client.get('/get_global_feature_importance')

    # Vérification du code de statut HTTP
    assert response.status_code == 200

    # Vérification du format de la réponse JSON
    data = response.get_json()
    assert isinstance(data, list)
    if data:
        assert isinstance(data[0], dict)

def test_load_model():
    """
    Teste la fonction load_model.

    Vérifie que la fonction charge un modèle avec succès à partir d'un fichier spécifié.
    """
    model_filename = "scr/models_saved/best_model.pkl"
    model = load_model(model_filename)
    assert model is not None  # Vérifie si le modèle est chargé avec succès

def test_load_explainer():
    """
    Teste la fonction load_explainer.

    Vérifie que la fonction charge un explainer avec succès à partir d'un fichier spécifié.
    """
    # Teste le chargement de l'explainer
    explainer_filename = "scr/models_saved/explainer_info.dill"
    explainer = load_explainer(explainer_filename)
    assert explainer is not None  # Vérifie si l'explainer est chargé avec succès


def test_format_client_data():
    """
    Teste la fonction format_client_data.

    Vérifie que la fonction formate correctement les données client en un DataFrame.
    """
    # Teste la fonction format_client_data
    client_data = {"feature1": 1, "feature2": 2}
    formatted_data = format_client_data(client_data)
    # Vérification que le résultat est du bon type et a la bonne forme
    assert isinstance(formatted_data, pd.DataFrame)

def test_predict_new_client():
    """
    Teste la fonction predict_new_client.

    Vérifie que la fonction renvoie une prédiction correcte pour un nouveau client.
    """
    # Teste la fonction predict_new_client
    model = mock.Mock()
    # Mock de la méthode predict_proba method
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    new_client_data = pd.DataFrame({"feature1": [1], "feature2": [2]})
    threshold = 0.5
    prediction, probability = predict_new_client(model, new_client_data, threshold)
    # Vérification que le résultat est du bon type et a la bonne forme
    assert isinstance(prediction, int)
    assert isinstance(probability, float)
    assert 0 <= probability <= 1
    assert prediction == 1
