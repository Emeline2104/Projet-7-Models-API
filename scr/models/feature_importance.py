# feature_importance.py

import lime
from lime import lime_tabular
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import joblib


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
    lime_train_x.fillna(0, inplace=True)  # Remplacez par la valeur que vous jugez appropriée
    lime_train_x.replace([np.inf, -np.inf, np.nan], [0, 0, 0], inplace=True)

    lime_train_x = lime_train_x.astype(float)

    # Afficher les valeurs infinies
    print("Valeurs infinies :", lime_train_x[np.isinf(lime_train_x)])
    
    print("nb NA :", lime_train_x.isna().sum())
    print(np.any(np.isfinite(lime_train_x)))

    explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(lime_train_x),
                                                mode='classification',
                                                feature_names=list(lime_train_x.columns),
                                                class_names=['TARGET'],
                                                training_labels=train_y,
                                                discretize_continuous=True)

    explainer.predict_fn = predict_fn  # Utilisez la fonction de prédiction personnalisée

    return explainer


def show_feature_importance(best_model, feats):
    """
    Affiche l'importance des caractéristiques d'un modèle.

    :param best_model: Le modèle pour lequel vous souhaitez afficher l'importance des caractéristiques.
    :param feats: La liste des noms de caractéristiques correspondant au modèle.
    """
    #print(model.named_steps.keys())
    #model = best_model.named_steps['classifier']
    model = best_model
    # Obtention les coefficients du modèle
    if isinstance(model, LGBMClassifier):
        coefficients =  model.feature_importances_
    else: 
        coefficients = model.coef_[0]

    # Création d'un DataFrame pour visualiser les coefficients et les caractéristiques correspondantes
    importance_df = pd.DataFrame({'Feature': feats, 'Coefficient': coefficients})
    importance_df['Absolute_Coefficient'] = np.abs(importance_df['Coefficient'])

    # Trie le DataFrame par ordre décroissant d'importance absolue
    importance_df = importance_df.sort_values(by='Absolute_Coefficient', ascending=False)

    # Affichage les caractéristiques les plus importantes
    print(importance_df)

    # Enregistre le DataFrame au format CSV
    importance_df.to_csv("feature_imortance_global.csv", index=False)

    return importance_df


def show_feature_importance_lgbm(best_model, feats): # a sup
    """
    Affiche l'importance des caractéristiques d'un modèle.

    :param best_model: Le modèle pour lequel vous souhaitez afficher l'importance des caractéristiques.
    :param feats: La liste des noms de caractéristiques correspondant au modèle.
    """
    # Obtention les coefficients du modèle
    print(best_model.named_steps.keys())

    # Accès aux coefficients
    model1 = best_model.named_steps['classifier']
    coefficients =  model1.feature_importances_

    # Création d'un DataFrame pour visualiser les coefficients et les caractéristiques correspondantes
    importance_df = pd.DataFrame({'Feature': feats, 'Coefficient': coefficients})
    importance_df['Absolute_Coefficient'] = np.abs(importance_df['Coefficient'])

    # Trie le DataFrame par ordre décroissant d'importance absolue
    importance_df = importance_df.sort_values(by='Absolute_Coefficient', ascending=False)

    # Enregistre le DataFrame au format CSV
    importance_df.to_csv("feature_imortance_global.csv", index=False)