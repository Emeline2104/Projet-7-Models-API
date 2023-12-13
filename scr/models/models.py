"""
Module de classification utilisant divers modèles de régression.

Ce module propose des fonctions pour entraîner et évaluer des modèles de classification, y compris des modèles de régression 
logistique, Light GBM, Random Forest, et un Dummy Classifier en baseline.

Auteur: Emeline Tapin
Date de création: 23/11/2023

Fonctions:
    - dummy_classifier(df, balance=None): Réalise la régression dummy Classifier en baseline.
    - reg_log(df, balance=None): Réalise la régression logistique en utilisant un ensemble de données et effectue l'optimisation des hyperparamètres.
    - kfold_lightgbm(df, balance=None): Réalise la régression light GBM en utilisant un ensemble de données et effectue l'optimisation des hyperparamètres.
    - random_forest(df, balance=None): Réalise la régression Random Forest en utilisant un ensemble de données et effectue l'optimisation des hyperparamètres.

Dépendances:
    - numpy
    - pandas
    - sklearn
    - mlflow
"""
# Importations nécessaires
from models.model_training import train_and_evaluate_model
from preprocessing.pre_processing import preprocessor
from models.feature_importance import explainer_lime
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import class_weight
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import mlflow.sklearn


def dummy_classifier(df, balance=None):
    """
    Réalise la régression dummy Classifier en baseline.

    :param df: DataFrame contenant les données d'entraînement et de test.
    :param balance: Indique le mode de gestion du déséquilibre de classes.

    :return: Le meilleur modèle de régression trouvé.
    """
    train_x, train_y, test_x, test_y, class_weight_dict = preprocessor(df, 'dummy', balance)
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

    model_selec = DummyClassifier(strategy='stratified')
        
    print("Début du Dummy Classifier. Forme de l'ensemble d'entraînement : {}, forme de l'ensemble de validation : {}".format(train_x.shape, test_x.shape))
    
    param_grid = {
    }
    
    # Evaluation du modèle
    best_model, _ = train_and_evaluate_model(train_x, train_y, test_x, test_y, model_selec, param_grid, balance, sample=2000)

    return best_model, _


def reg_log(df, balance=None):
    """
    Réalise la régression logistique en utilisant un ensemble de données et effectue l'optimisation des hyperparamètres.

    :param df: DataFrame contenant les données d'entraînement et de test.
    :param balance: Indique le mode de gestion du déséquilibre de classes.

    :return: Le meilleur modèle de régression logistique trouvé.
    """
    mlflow.start_run()
    train_x, train_y, test_x, test_y, class_weight_dict = preprocessor(df, 'reg_log', balance)
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

    # Modèle de regression logistique
    model_selec = LogisticRegression(max_iter=3000, class_weight=class_weight_dict)

    mlflow.log_param("Gestion déséquilibre des classes", balance)
    mlflow.log_param("Type de modèle", "Regression Logistique")

    # Grille d'hyperparamètres
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10]
        }
    print("Début de la Régression Logistique. Forme de l'ensemble d'entraînement : {}, forme de l'ensemble de validation : {}".format(train_x.shape, test_x.shape))

    # Evaluation du modèle
    best_model, explainer_features_importance = train_and_evaluate_model(train_x, train_y, test_x, test_y, model_selec, param_grid, balance, sample=2000)

    return best_model, explainer_features_importance



def kfold_lightgbm(df, balance=None):
    """
    Réalise la régression light GBM en utilisant un ensemble de données et effectue l'optimisation des hyperparamètres.

    :param df: DataFrame contenant les données d'entraînement et de test.
    :param balance: Indique le mode de gestion du déséquilibre de classes.

    :return: Le meilleur modèle de régression trouvé.
    """
        
    mlflow.start_run()

    train_x, train_y, test_x, test_y, class_weight_dict = preprocessor(df, 'lgbm', balance)
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

    train_y = np.array(train_y)

    # Modèle de Light GBM
    model_selec = LGBMClassifier(class_weight=class_weight_dict)

    # Grille d'hyperparamètres
    param_grid = {
        'classifier__n_estimators': [100, 200, 500],
        'classifier__learning_rate': [0.05, 0.1, 0.3],
        'classifier__num_leaves': [31, 63, 100],
        }  
    print("Début de la Régression LGBM. Forme de l'ensemble d'entraînement : {}, forme de l'ensemble de validation : {}".format(train_x.shape, test_x.shape))
    
    mlflow.log_param("Gestion déséquilibre des classes", balance)
    mlflow.log_param("Type de modèle", "LGBM")

    print("train_x shape",train_x.shape)
    print("train_y shape",train_y.shape)
    print("test_x shape",test_x.shape)
    print("test_y shape",test_y.shape)
    # Evaluation du modèle
    best_model, explainer_features_importance = train_and_evaluate_model(train_x, train_y, test_x, test_y, model_selec, param_grid, balance, sample=2000)

    return best_model, explainer_features_importance



    # Features importance
    # show_feature_importance_lgbm(best_model, train_x.columns)

def random_forest(df, balance=None):
    """
    Réalise la régression Random Forest en utilisant un ensemble de données et effectue l'optimisation des hyperparamètres.

    :param df: DataFrame contenant les données d'entraînement et de test.
    :param balance: Indique le mode de gestion du déséquilibre de classes.

    :return: Le meilleur modèle de régression trouvé.
    """
    mlflow.start_run()

    train_x, train_y, test_x, test_y, class_weight_dict = preprocessor(df, 'random_forest', balance)
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

    train_y = np.array(train_y)

    # Modèle de Random Forest
    model_selec = RandomForestClassifier(class_weight=class_weight_dict)

    # Grille d'hyperparamètres
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],  # Nombre d'arbres dans la forêt
        'classifier__max_depth': [None, 10, 20, 30],  # Profondeur maximale de chaque arbre
        'classifier__min_samples_split': [2, 5, 10],  # Nombre minimum d'échantillons requis pour diviser un nœud interne
        'classifier__min_samples_leaf': [1, 2, 4]  # Nombre minimum d'échantillons requis à chaque nœud feuille
    }
        
    print("Début de la Régression Random Forest. Forme de l'ensemble d'entraînement : {}, forme de l'ensemble de validation : {}".format(train_x.shape, test_x.shape))
    
    mlflow.log_param("Gestion déséquilibre des classes", balance)
    mlflow.log_param("Type de modèle", "Random Forest")


    # Evaluation du modèle
    best_model, explainer_features_importance = train_and_evaluate_model(train_x, train_y, test_x, test_y, model_selec, param_grid, balance, sample=2000)

    return best_model, explainer_features_importance
