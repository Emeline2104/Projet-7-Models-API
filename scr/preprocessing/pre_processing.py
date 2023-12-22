"""
Script de prétraitement pour la tâche de classification de crédit.

Auteur : Emeline Tapin
Date de création : 21/11/2023

Ce script comprend des fonctions pour charger, prétraiter, agréger et créer de nouvelles fonctionnalités 
pour la classification de crédit.
Le script contient des fonctions pour traiter différentes tables de données, telles que les données d'application, 
les données du bureau, les données d'application précédente, POS_CASH_balance, installments payments et credit card balance.

Dépendances :
- pandas
- numpy
- gc
- sklearn.model_selection (train_test_split)

Exemple :
1. Sélectionnez les caractéristiques avec un taux de valeur manquante inférieur à 0.5 :
   df = select_features(df)

2. Divisez les données en ensembles d'entraînement/validation et de test,
et traitez les valeurs manquantes :
   train_x, train_y, test_x, test_y = split_data(df)
   train_x, test_x, train_y, test_y = handle_missing_values(train_x, test_x, train_y, test_y)

3. Nettoyez les noms des caractéristiques dans le DataFrame :
   clean_feature_names(df)
"""
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

def select_features(df, threshold=0.5):
    """
    Sélectionne les caractéristiques avec un taux de valeur manquante de moins de `threshold`.

    Args:
        df (DataFrame): Le DataFrame contenant les données.
        threshold (float): Le seuil de taux de remplissage.

    Returns:
        selected_features (list): La liste des caractéristiques sélectionnées.
    """
    selected_columns = []

    selected_columns = [
        feature for feature in df.columns if df[feature].isna().sum() / len(df) < threshold
        ]

    keys = ['SK_ID_BUREAU',
                'SK_ID_CURR',
                'SK_ID_BUREAU',
                'SK_ID_PREV',
                'TARGET', 
                ]

    selected_columns.append(keys)
    selected_columns.append('TARGET')

    df = df.loc[:, df.columns.isin(selected_columns)]
    return df


def split_data(df):
    """
    Prétraitement des données.

    Cette fonction divise les données en ensembles d'entraînement/validation et de test, 
    extrait les indices, et sépare les caractéristiques (features) de la variable 
    cible pour l'entraînement et la validation.

    :param df: DataFrame contenant les données d'entraînement et de test.
    
    :return: train_x, train_y, test_x, test_y
    """

    # Divise les données en ensembles d'entraînement/validation et de test
    train_df = df[df['TARGET'].notnull()]

    # Sélectionne les caractéristiques pertinentes pour l'entraînement
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    # Extraie les indices pour l'ensemble d'entraînement et de test
    train_idx, test_idx = train_test_split(train_df.index, test_size=0.3, random_state=42)

    # Sépare les caractéristiques et la cible pour l'ensemble d'entraînement
    train_x, train_y = train_df[feats].loc[train_idx], train_df['TARGET'].loc[train_idx]

    # Sépare les caractéristiques et la cible pour l'ensemble de validation
    test_x, test_y = train_df[feats].loc[test_idx], train_df['TARGET'].loc[test_idx]

    return train_x, train_y, test_x, test_y

def handle_missing_values(train_x, test_x, train_y, test_y):
    """
    Gestion des valeurs manquantes dans les ensembles d'entraînement et de test.

    Cette fonction permet de supprimer les lignes contenant des valeurs manquantes dans les ensembles
    d'entraînement et de test, tout en maintenant la cohérence entre les caractéristiques 
    et les cibles.

    :param train_x: Caractéristiques de l'ensemble d'entraînement.
    :param test_x: Caractéristiques de l'ensemble de test.
    :param train_y: Cible de l'ensemble d'entraînement.
    :param test_y: Cible de l'ensemble de test.

    :return: train_x, test_x, train_y, test_y après suppression des valeurs manquantes.
    """
    # Supprime les lignes contenant des valeurs manquantes
    train_x = train_x.dropna()
    test_x = test_x.dropna()

    # Assure la cohérence entre les caractéristiques et les cibles
    train_y = train_y.loc[train_x.index]
    test_y = test_y.loc[test_x.index]

    return train_x, test_x, train_y, test_y

def clean_feature_names(df):
    """
    Nettoie les noms des caractéristiques dans un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.

    Returns:
        None: Modifie directement le DataFrame en renommant les colonnes.
    """
    new_feature_names = []
    for column in df.columns:
        # Remplace les espaces, virgules et autres caractères spéciaux par des tirets bas
        new_name = column.replace(" ", "_").replace(",", "_").replace("(", "_").replace(")", "_").replace(":", "_")
        new_feature_names.append(new_name)
    df.columns = new_feature_names


def preprocessor(df, model_type, balance=None):
    """
    Réalise le prétraitement des données en fonction du type de modèle et de la gestion du déséquilibre.

    Args:
        df (DataFrame): DataFrame contenant les données d'entraînement et de test.
        model_type (str): Type de modèle ('dummy', 'reg_log', 'kfold_lightgbm',
        'random_forest', etc.).
        balance (str): Indique le mode de gestion du déséquilibre de classes.

    Returns:
        Tuple: Un tuple contenant les DataFrames d'entraînement et de test prétraités.
    """
    # Sélection des caractéristiques
    if model_type in ['dummy', 'reg_log', 'random_forest'] or (model_type == 'lgbm' and balance == 'SMOTE'):
        df = select_features(df, threshold=0.3)

    # Séparation des données en entraînement et test
    train_x, train_y, test_x, test_y = split_data(df)

    # Gestion des valeurs manquantes
    if model_type in ['dummy', 'reg_log', 'random_forest'] or (model_type == 'lgbm' and balance == 'SMOTE'):
        train_x, test_x, train_y, test_y = handle_missing_values(train_x, test_x, train_y, test_y)

    # Gestion du déséquilibre de classes (sauf pour le modèle dummy)
    if model_type != 'dummy':
        if balance == 'SMOTE':
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            train_x, train_y = smote.fit_resample(train_x, train_y)
            class_weight_dict = None
            print("Ah")
        elif balance == 'class_weight':
            class_weights = class_weight.compute_class_weight(
                'balanced',
                classes=np.unique(train_y),
                y=train_y,
                )
            class_weight_dict = dict(enumerate(class_weights))
            print("B")
        else:
            class_weight_dict = None
            print("C")
    else:
        class_weight_dict = None

    # Nettoyage des noms de colonnes
    clean_feature_names(train_x)
    clean_feature_names(pd.DataFrame(train_y))

    # Retourne les DataFrames prétraités
    return train_x, train_y, test_x, test_y, class_weight_dict

def preprocessor_api(df, model_type, balance=None):
    """
    Réalise le prétraitement des données en fonction du type de modèle 
    et de la gestion du déséquilibre.

    Args:
        df (DataFrame): DataFrame contenant les données d'entraînement et de test.
        model_type (str): Type de modèle ('dummy', 'reg_log', 'kfold_lightgbm', 
        'random_forest', etc.).
        balance (str): Indique le mode de gestion du déséquilibre de classes.

    Returns:
        Tuple: Un tuple contenant les DataFrames d'entraînement et de test prétraités.
    """
    # Sélection des caractéristiques
    if model_type in ['dummy', 'reg_log', 'random_forest','lgbm'] or (model_type == 'lgbm' and balance == 'SMOTE'):
        df = select_features(df, threshold=0.3)

    feats = [f for f in df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    df = df[feats]

    # Remplacement des valeurs None par NaN
    df.replace({None: np.nan}, inplace=True)

    # Gestion des valeurs manquantes
    if model_type in ['dummy', 'reg_log', 'random_forest','lgbm'] or (model_type == 'lgbm' and balance == 'SMOTE'):
        df = df.dropna()

    # Nettoyage des noms de colonnes
    clean_feature_names(df)

    # Retourne les DataFrames prétraités
    return df
