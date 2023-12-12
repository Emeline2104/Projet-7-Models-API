"""
Fichier de tests pour l'API Flask de prédiction de la TARGET d'un client et la récupération d'informations client.

Ce fichier contient des tests unitaires et d'intégration pour vérifier le bon fonctionnement
des différentes fonctionnalités de l'API, notamment la prédiction de la TARGET, la récupération
d'informations sur un client, et l'obtention d'informations brutes sur un client.

Tests unitaires :
- test_predict_new_client : Vérifie si l'endpoint de prédiction renvoie une réponse avec un code 200 et une clé 'predictions'.
- test_obtenir_informations_client : Vérifie si l'endpoint d'informations client brut renvoie une réponse avec un code 200 et une clé 'application_info'.

Pour exécuter les tests, assurez-vous que l'application Flask est en cours d'exécution localement.

Exemple :
    pytest test_api.py
"""
import pytest
import requests

def test_predict_new_client():
    """
    Test unitaire pour la fonction de prédiction.
    Vérifie si l'endpoint de prédiction renvoie une réponse avec un code 200 et une clé 'predictions'.
    """
    # ID du client à utiliser dans le test
    client_id = 100001

    # Effectue une requête GET vers l'endpoint de prédiction
    response = requests.get(f'http://127.0.0.1:5001/predict/{client_id}')

    # Vérifie que la réponse a un code de statut HTTP 200 (OK)
    assert response.status_code == 200

    # Vérifie que la clé 'predictions' est présente dans la réponse JSON
    assert 'predictions' in response.json()

def test_obtenir_informations_client():
    """
    Test d'intégration pour l'endpoint de récupération d'informations client brut.
    Vérifie si l'endpoint renvoie une réponse avec un code 200 et une clé 'application_info'.
    """
    # ID du client à utiliser dans le test
    client_id = 100001

    # Effectue une requête GET vers l'endpoint d'informations client brut
    response = requests.get(f'http://127.0.0.1:5001/informations_client_brut/{client_id}')

    # Vérifie que la réponse a un code de statut HTTP 200 (OK)
    assert response.status_code == 200

    # Vérifie que la clé 'application_info' est présente dans la réponse JSON
    assert 'application_info' in response.json()
