"""
Module pour calculer et afficher l'importance des caractéristiques d'un modèle à l'aide 
de la méthode LIME.

Ce module propose des fonctions pour expliquer l'importance des caractéristiques individuelles p
our un client donné, ainsi que pour afficher l'importance globale des caractéristiques d'un modèle.

Fonctions:
    - expliquer_importance_caracteristiques(train_x, train_y, predict_fn):
        Calcule l'importance des caractéristiques individuelles pour un client 
        donné en utilisant LIME.

    - afficher_importance_caracteristiques(meilleur_modele, caracteristiques):
        Affiche l'importance des caractéristiques d'un modèle.

Dépendances:
    - lime
    - numpy
    - pandas
    - lightgbm
    - joblib
"""

from lime import lime_tabular
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

def explainer_lime(train_x, train_y, predict_fn):
    """
    Calcule l'importance des caractéristiques individuelles pour un client donné en utilisant LIME.

    Args:
        train_x (pd.DataFrame): Données d'entraînement.
        train_y (pd.Series): Étiquettes correspondantes aux données d'entraînement.
        model: Le modèle à expliquer.
        predict_fn: Fonction de prédiction personnalisée.

    Returns:
        lime.lime_tabular.LimeTabularExplainer: Explorateur LimeTabular.
    """

    # Utilisation de LimeTabularExplainer
    lime_train_x = train_x.copy()
    lime_train_x.fillna(0, inplace=True)
    lime_train_x.replace([np.inf, -np.inf, np.nan], [0, 0, 0], inplace=True)
    lime_train_x = lime_train_x.astype(float)
    explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(lime_train_x),
                                                mode='classification',
                                                feature_names=list(lime_train_x.columns),
                                                class_names=['TARGET'],
                                                training_labels=train_y,
                                                discretize_continuous=True)

    explainer.predict_fn = predict_fn

    return explainer


def show_feature_importance(best_model, feats):
    """
    Affiche l'importance des caractéristiques d'un modèle.

    Args:
        meilleur_modele: Le modèle pour lequel vous souhaitez afficher 
        l'importance des caractéristiques.
        caracteristiques: La liste des noms de caractéristiques correspondant au modèle.

    Returns:
        pd.DataFrame: DataFrame contenant les caractéristiques et leur importance.
    """
    model = best_model
    # Obtention les coefficients du modèle
    if isinstance(model, (LGBMClassifier, RandomForestClassifier)):
        coefficients =  model.feature_importances_
    else:
        coefficients = model.coef_[0]

    # Création d'un DataFrame pour visualiser les coefficients et les caractéristiques
    importance_df = pd.DataFrame({'Feature': feats, 'Coefficient': coefficients})
    importance_df['Absolute_Coefficient'] = np.abs(importance_df['Coefficient'])

    # Trie le DataFrame par ordre décroissant d'importance absolue
    importance_df = importance_df.sort_values(by='Absolute_Coefficient', ascending=False)

    # Enregistre le DataFrame au format CSV
    importance_df.to_csv("feature_imortance_global.csv", index=False)

    return importance_df
