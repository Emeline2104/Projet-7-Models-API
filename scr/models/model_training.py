"""
Module de prétraitement et d'entraînement de modèle.

Ce module contient des fonctions pour le prétraitement des données, la sélection de caractéristiques,
l'entraînement de modèles, et l'évaluation des performances. Les fonctions sont conçues pour faciliter
le développement et la validation de modèles de prédiction de défaut de paiement.

Auteur: Emeline Tapin
Date de création: 21/11/2023

Fonctions:
    - custom_scorer(y_true, y_pred): Calcule un score personnalisé basé sur le déséquilibre du coût métier entre les faux positifs (FP) et les faux négatifs (FN).
    - find_optimal_threshold(y_true, probas): Trouve le seuil optimal pour la classification binaire en évaluant différentes valeurs de seuil.
    - create_pipeline(model_selec): Crée un pipeline avec mise à l'échelle des caractéristiques et le modèle sélectionné.
    - train_model_CV(pipeline, train_x, train_y, param_grid, balance): Entraîne un modèle à l'aide d'une GridSearchCV.
    - select_important_features_threeshold(model, train_x, test_x, threshold=0.005): Sélectionne les caractéristiques importantes basées sur le modèle.
    - select_top_features(model, train_x, test_x, num_features=50): Sélectionne les caractéristiques importantes basées sur le modèle.
    - train_model_with_best_params(pipeline, train_x, train_y, best_params, balance): Entraîne un modèle avec les meilleurs hyperparamètres spécifiés.
    - evaluate_model(trained_model, test_x, test_y): Évalue le modèle entraîné sur l'ensemble de test.
    - train_and_evaluate_model(train_x_all, train_y_all, test_x, test_y, model_selec, param_grid, balance, sample=None): Entraîne et évalue un modèle.

Dépendances:
    - numpy
    - time
    - pandas
    - sklearn
    - imblearn
    - mlflow
"""
from scr.models.feature_importance import explainer_lime
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import joblib
import dill

# Dépendances MLflow
import mlflow

def custom_scorer(y_true, y_pred):
    """
    Calcule un score personnalisé basé sur le déséquilibre du coût métier entre les faux positifs (FP) et les faux négatifs (FN).

    Parameters:
        y_true (array-like): Les vraies étiquettes de classe.
        y_pred (array-like): Les étiquettes de classe prédites par le modèle.

    Returns:
        float: Le score personnalisé basé sur le coût métier normalisé par le nombre d'individus.
    """
    fp = 0  # Initialisation du compteur de faux positifs
    fn = 0  # Initialisation du compteur de faux négatifs
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 0:  # Bon client prédit mauvais
            fp += 1  # Faux positif
        elif y_pred[i] == 0 and y_true[i] == 1:  # Mauvais client prédit bon client
            fn += 1  # Faux négatif

    # Calcul du score personnalisé (coût métier) normalisé par le nombre d'individus
    num_individuals = len(y_true)
    score_personalized = (10 * fn + fp) / num_individuals

    return score_personalized


def find_optimal_threshold(y_true, probas):
    """
    Trouve le seuil optimal pour la classification binaire en évaluant différentes valeurs de seuil.

    Args:
    y_true (array-like): Les vraies valeurs des étiquettes.
    probas (array-like): Les probabilités de classe positives.

    Returns:
    float: Le seuil optimal.
    """

    best_threshold = 0  # Initialisation du seuil optimal
    best_score = 1  # Initialisation du meilleur score

    # Parcours d'une plage de valeurs de seuil de 0 à 1
    for threshold in np.linspace(0, 1, 100):
        # Conversion des probabilités en prédictions binaires basées sur le seuil actuel
        predictions = (probas >= threshold).astype(int)
        score = custom_scorer(y_true, predictions)  # Évaluation de la performance

        # Mise à jour du seuil si le score est meilleur
        if score < best_score:
            best_threshold = threshold
            best_score = score
    print(predictions)
    # Affichage des résultats
    print(f'Best threshold: {best_threshold:.2f}')
    print(f'Best score: {best_score:.4f}')

    mlflow.log_param("Threshold", best_threshold)

    return best_threshold

def create_pipeline(model_selec):
    """
    Crée un pipeline avec mise à l'échelle des caractéristiques et le modèle sélectionné.

    Args:
    model_selec (estimator): Modèle sélectionné pour l'entraînement.
    sample (int): Nombre de lignes à sélectionner pour l'entraînement.

    Returns:
    Pipeline: Un pipeline avec les étapes nécessaires.
    """
    if isinstance(model_selec, LogisticRegression):
        steps = [('scaler', StandardScaler()), ('classifier', model_selec)]
    else:
        steps = [('classifier', model_selec)]

    return Pipeline(steps)


def train_model_CV(pipeline, train_x, train_y, param_grid, balance):
    """
    Entraîne un modèle à l'aide d'une GridSearchCV.

    Args:
    pipeline (Pipeline): Le pipeline contenant le modèle et les étapes associées.
    train_x (array-like): Features de l'ensemble d'entraînement.
    train_y (array-like): Labels de l'ensemble d'entraînement.
    param_grid (dict): Grille des hyperparamètres à rechercher.
    balance: Stratégie de gestion de l'umbalance des classes.

    Returns:
    estimator: Meilleur modèle entraîné.
    """    
    if balance == 'class_weight':
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
        class_weight_dict = dict(enumerate(class_weights))
        pipeline.set_params(classifier__class_weight=class_weight_dict)
    elif balance == 'SMOTE':
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        train_x, train_y = smote.fit_resample(train_x, train_y)
    else: 
        class_weight_dict = None
    
    cv=StratifiedKFold(n_splits=5)

    debut_temps_train = time.time()
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv)
    grid_search.fit(train_x, train_y)
    fin_temps_train = time.time()
    temps_train = fin_temps_train - debut_temps_train

    print("Best params",grid_search.best_params_)
    
    mlflow.log_param("Best param", grid_search.best_params_)
    mlflow.log_metric("Temps entrainement", temps_train)

    return grid_search.best_estimator_, grid_search.best_params_

def select_important_features_threeshold(model, train_x, test_x, threshold=0.005):
    """
    Sélectionne les caractéristiques importantes basées sur le modèle.

    Args:
    model: Le modèle entraîné.
    train_x (array-like): Features de l'ensemble d'entraînement.
    threshold (float): Seuil d'importance pour la sélection des caractéristiques.

    Returns:
    DataFrame: Ensemble d'entraînement avec des caractéristiques sélectionnées.
    """
    if isinstance(model.named_steps['classifier'], LogisticRegression):
        # Pour la régression logistique, nous n'avons pas de feature_importances_
        # Nous pourrions utiliser les coefficients absolus comme mesure d'importance
        coefficients = np.abs(model.named_steps['classifier'].coef_[0])
    else:
        coefficients = model.named_steps['classifier'].feature_importances_

    important_features = train_x.columns[coefficients > threshold]

    print('Nombre de features sélectionné:', len(important_features))
    
    # Ensure selected features exist in both train_x and test_x
    selected_features = train_x.columns.intersection(test_x.columns)
    
    train_x = train_x[selected_features]
    test_x = test_x[selected_features]

    test_x = test_x.reindex(columns=train_x.columns, fill_value=0)
    mlflow.log_param("Nombre de features sélectionnées", len(selected_features))

    return train_x, test_x


def select_top_features(model, train_x, test_x, num_features=50):
    """
    Sélectionne les caractéristiques importantes basées sur le modèle.

    Args:
    model: Le modèle entraîné.
    train_x (array-like): Features de l'ensemble d'entraînement.
    num_features (int): Nombre de caractéristiques à sélectionner.

    Returns:
    DataFrame: Ensemble d'entraînement avec des caractéristiques sélectionnées.
    """
    num_features = len(train_x.columns)
    if isinstance(model.named_steps['classifier'], LogisticRegression):
        # Pour la régression logistique, nous n'avons pas de feature_importances_
        # Nous pourrions utiliser les coefficients absolus comme mesure d'importance
        coefficients = np.abs(model.named_steps['classifier'].coef_[0])
    elif isinstance(model.named_steps['classifier'], DummyClassifier):
        # Pour le DummyClassifier, nous n'avons pas de mécanisme standard pour mesurer l'importance des caractéristiques
        # Vous pouvez ajouter le traitement spécifique pour le DummyClassifier ici
        # Par exemple, sélectionnez simplement les premières caractéristiques sans critère d'importance
        coefficients = np.ones(len(train_x.columns))
    else:
        coefficients = model.named_steps['classifier'].feature_importances_

    # Obtenez les indices des caractéristiques triées par importance décroissante
    feature_indices = np.argsort(coefficients)[::-1]

    # Sélectionnez les indices des premières caractéristiques
    selected_feature_indices = feature_indices[:num_features]

    # Obtenez les noms des caractéristiques sélectionnées
    important_features = train_x.columns[selected_feature_indices]

    print('Nombre de features sélectionné:', len(important_features))
    
    # Ensure selected features exist in both train_x and test_x
    selected_features = train_x.columns.intersection(test_x.columns)
    
    train_x = train_x[selected_features]
    test_x = test_x[selected_features]
    print(train_x.shape)
    print(test_x.shape)
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0)
    mlflow.log_param("Nombre de features sélectionnées", len(important_features))

    return train_x, test_x


def train_model_with_best_params(pipeline, train_x, train_y, best_params, balance):
    """
    Entraîne un modèle avec les meilleurs hyperparamètres spécifiés.

    Args:
    pipeline (Pipeline): Le pipeline contenant le modèle et les étapes associées.
    train_x (array-like): Features de l'ensemble d'entraînement.
    train_y (array-like): Labels de l'ensemble d'entraînement.
    best_params (dict): Dictionnaire des meilleurs hyperparamètres.

    Returns:
    estimator: Modèle entraîné.
    """
    if balance == 'class_weight':
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
        class_weight_dict = dict(enumerate(class_weights))
        pipeline.set_params(classifier__class_weight=class_weight_dict)
    elif balance == 'SMOTE':
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        train_x, train_y = smote.fit_resample(train_x, train_y)
    else: 
        class_weight_dict = None
    
    # Mise à jour des hyperparamètres du modèle dans le pipeline avec les meilleurs paramètres
    updated_pipeline = pipeline.set_params(**best_params)

    # Entrainement du modèle mis à jour sur les données d'entraînement
    updated_pipeline.fit(train_x, train_y)

    return updated_pipeline.named_steps['classifier']


def evaluate_model(trained_model, test_x, test_y):
    """
    Évalue le modèle entraîné sur l'ensemble de test.

    Args:
    trained_model: Le modèle entraîné.
    test_x (array-like): Caractéristiques de l'ensemble de test.
    test_y (array-like): Étiquettes de l'ensemble de test.

    Returns:
    dict: Dictionnaire contenant les métriques d'évaluation.
    """
    # Prédictions sur l'ensemble de test
    debut_temps_inference = time.time()
    probas = trained_model.predict_proba(test_x)[:, 1]
    fin_temps_inference = time.time()
    temps_inference = fin_temps_inference - debut_temps_inference

    # Calcul du seuil optimal
    meilleur_seuil = find_optimal_threshold(test_y, probas)

    # Utilisation du seuil optimal pour prédire les classes
    predictions_avec_seuil = (probas >= meilleur_seuil).astype(int)

    # Calcul des métriques de performance
    precision = accuracy_score(test_y, predictions_avec_seuil)
    auc = roc_auc_score(test_y, predictions_avec_seuil)
    score_personnalise_test = custom_scorer(np.array(test_y), np.array(predictions_avec_seuil))

    # Affichage des métriques de performance
    print("Précision sur l'ensemble de test:", precision)
    print("AUC sur l'ensemble de test:", auc)
    print("Score personnalisé sur l'ensemble de test:", score_personnalise_test)
    print("Temps d'inférence sur l'ensemble de test:", temps_inference)

    # Enregistrement des métriques dans MLflow
    mlflow.log_metric("Précision", precision)
    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("Score Personnalisé", score_personnalise_test)
    mlflow.log_metric("Temps inférence", temps_inference)

    # Retourne les métriques sous forme de dictionnaire
    metrics_dict = {
        "Précision": precision,
        "AUC": auc,
        "Score Personnalisé": score_personnalise_test,
        "Temps d'inférence": temps_inference
    }

    mlflow.end_run()
    
    return metrics_dict


def train_and_evaluate_model(train_x_all, train_y_all, test_x, test_y, model_selec, param_grid, balance, sample=None):
    """
    Entraîne et évalue un modèle.

    Args:
    train_x (array-like): Features de l'ensemble d'entraînement.
    train_y (array-like): Labels de l'ensemble d'entraînement.
    test_x (array-like): Features de l'ensemble de test.
    test_y (array-like): Labels de l'ensemble de test.
    model_selec (estimator): Modèle sélectionné pour l'entraînement.
    param_grid (dict): Grille des hyperparamètres à rechercher.
    cv: Stratégie de validation croisée.
    sample (int): Nombre de lignes à sélectionner pour l'entraînement.

    Returns:
    Pipeline: Meilleur modèle entraîné.
    """
    if sample is not None: 
        train_x, _, train_y, _ = train_test_split(train_x_all, train_y_all, test_size=0.92, random_state=42)
    else : 
        train_x = train_x_all
        train_y = train_y_all
        
    # Crée un pipeline
    pipeline = create_pipeline(model_selec)

    # Entraîne le modèle
    model_HP, best_params = train_model_CV(pipeline, train_x, train_y, param_grid, balance)

    # Sélectionne les caractéristiques importantes
    train_x_selected, test_x_selected = select_important_features_threeshold(model_HP, train_x_all, test_x)
    print("Taille jeux d'entrainement sélec",train_x_selected.shape)
    print("Taille jeux de test sélec",test_x_selected.shape)

    # Utilisation du nouveau modèle pour l'entraînement
    trained_model = train_model_with_best_params(pipeline, train_x_selected, train_y_all, best_params, balance)

    # Evaluation du modèlé
    evaluate_model(trained_model, test_x_selected, test_y)
    test_x_selected.head().to_csv("Data/sampled/test_x_selected_head.csv", index=False)

    # feature importance 
    print(np.any(np.isfinite(train_x)))
    # print(train_x.max())
    # Fonction de prédiction pour LIME
    #def predict_fn(x):
        # Assurez-vous que le modèle retourne des probabilités pour chaque classe
        #return trained_model.predict_proba(x, num_iteration=trained_model.best_iteration_)[:, 1]

    def predict_fn(x):
        # Assuming 'model' is your trained LightGBM binary classification model
        return trained_model.predict(x, num_iteration=trained_model.best_iteration_)

    # Expliquer les caractéristiques importantes avec LIME
    #explainer_features_importance = explainer_lime(train_x_selected, train_y_all, predict_fn=predict_fn)

    # Calcul des caractéristiques importantes pour une instance spécifique
    #instance_idx = 0
    #explanation = explainer_features_importance.explain_instance(
    #    train_x_selected.iloc[instance_idx],
    #    predict_fn,
    #    num_features=len(train_x_selected.columns),
    #)

    # Obtention des caractéristiques importantes sous forme de liste
    #feature_importance_list = explanation.as_list()
    #print("Feature Importance List:", feature_importance_list)

    # Sauvegarde manuelle des caractéristiques importantes
    #explainer_features_importance_info = {
    #    'feature_importance': feature_importance_list,
    #    'other_info': 'other_info_value'  # Ajoutez d'autres informations nécessaires
    #}

    # Enregistrez l'explainer
    #with open("explainer_info.dill", "wb") as file:
    #    dill.dump(explainer_features_importance, file)


    return trained_model, _
