import sys
sys.path.append("/Users/beatricetapin/Documents/2023/Data Science/Projet_7_Modele_API/")
sys.path.append("/Users/beatricetapin/Documents/2023/Data Science/Projet_7_Modele_API/scr/")


from scr.preprocessing.pre_processing import select_features, split_data, handle_missing_values, clean_feature_names, preprocessor, preprocessor_api
from scr.preprocessing.aggregation import (
    one_hot_encoder,
    application_train_test,
    bureau_and_balance,
    previous_applications,
    pos_cash,
    installments_payments,
    credit_card_balance,
    aggreger,
)
from models.model_training import (
    custom_scorer,
    find_optimal_threshold,
    create_pipeline,
    train_model_CV,
    select_important_features_threeshold,
    select_top_features,
    train_model_with_best_params,
    evaluate_model,
    train_and_evaluate_model,
)
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import numpy as np

# Test de la fonction select_features
def test_select_features():
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
    # Créez un DataFrame pour le test avec des valeurs manquantes
    df = pd.DataFrame({
        'feature1': [1, 2, 3, None, 5],
        'feature2': [1, None, 3, 4, 5],
        'TARGET': [0, 1, 0, 1, 0]
    })

    # Séparez les caractéristiques et les cibles
    train_x = df[['feature1', 'feature2']]
    test_x = df[['feature1', 'feature2']]
    train_y = df['TARGET']
    test_y = df['TARGET']

    # Appelez la fonction handle_missing_values
    train_x, test_x, train_y, test_y = handle_missing_values(train_x, test_x, train_y, test_y)

    # Assurez-vous que les lignes avec des valeurs manquantes ont été supprimées
    assert len(train_x) == 3
    assert len(test_x) == 3

    # Assurez-vous que la cohérence entre les caractéristiques et les cibles est maintenue
    assert len(train_y) == 3
    assert len(test_y) == 3

    # Assurez-vous que les indices sont alignés
    assert train_x.index.equals(train_y.index)
    assert test_x.index.equals(test_y.index)

# Test de la fonction clean_feature_names
def test_clean_feature_names():
    df = pd.DataFrame({
        'feature 1': [1, 2, 3],
        'feature,2': [4, 5, 6]
    })
    
    clean_feature_names(df)
    
    assert set(df.columns) == {'feature_1', 'feature_2'}

# Fonction de test pour one_hot_encoder
def test_one_hot_encoder():
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
    # Générer des données d'exemple pour les tests
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
    y = np.random.randint(0, 2, 100)
    return X, y

def test_custom_scorer():
    # Tester custom_scorer avec des prédictions parfaites
    y_true = np.array([0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 0])
    score = custom_scorer(y_true, y_pred)
    assert score == 0.0, "Le score devrait être 0.0 avec des prédictions parfaites."

def test_find_optimal_threshold(sample_data):
    X, y = sample_data
    np.random.seed(42)
    probas = np.random.rand(len(y))
    threshold = find_optimal_threshold(y, probas)
    assert 0.0 <= threshold <= 1.0, "Le seuil optimal doit être entre 0 et 1."

def test_create_pipeline():
    # Tester la création du pipeline avec un modèle de régression logistique
    model = LGBMClassifier()
    pipeline = create_pipeline(model)
    assert len(pipeline.steps) == 1, "Le pipeline devrait avoir deux étapes."
