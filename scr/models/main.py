"""
Script principal pour entraîner différents modèles de machine learning en gérant le déséquilibre de classes.
"""
import sys
sys.path.append("/Users/beatricetapin/Documents/2023/Data Science/Projet_7_Modele_API/")
sys.path.append("/Users/beatricetapin/Documents/2023/Data Science/Projet_7_Modele_API/scr/")
from preprocessing.aggregation import aggreger
from preprocessing.pre_processing import preprocessor
from models.models_selec import dummy_classifier, reg_log, kfold_lightgbm, random_forest
import joblib

def main(type_model, balance):
    """
    Entraîne un modèle spécifié par type_model en gérant le déséquilibre de classe par un équilibreur de classes défini par balance.

    Args:
    type_model (str): Le type de modèle à entraîner. Choix possibles : 'log', 'lgbm' ou 'random_forest'.
    balance (str): Le mode de gestion du déséquilibre des classes. Choix possibles : 'class_weight', 'SMOTE' ou None.

    Returns:
    object: Le modèle entraîné selon le type spécifié.
    """
    # Obtention des données agrégées
    df = aggreger(debug=False)

    # Modèle baseline
    # dummy_classifier(df, balance=None)

    # Sélection du modèle à entraîner
    if type_model == 'log':
        # Entraînement du modèle de régression logistique
        model = reg_log(df, balance=balance)
    elif type_model == 'lgbm':
        # Entraînement du modèle Light GBM
        model = kfold_lightgbm(df, balance=balance)
    elif type_model == 'random_forest':
        # Entraînement du modèle Random Forest
        model= random_forest(df, balance=balance)
    else:
        raise ValueError("Le type de modèle spécifié n'est pas pris en charge.")

    # Sérialiasation du prétraitement des données 
    preprocessing_function_filename = "models/preprocessing_function.pkl"
    joblib.dump(preprocessor, preprocessing_function_filename)


    # Sérialisation du modèle
    model_filename = "models/best_model.pkl"
    joblib.dump(model, model_filename)

    # Chargement du modèle sérialisé (pour vérification)
    joblib.load(model_filename)

    print(f"Le modèle a été sérialisé avec succès dans {model_filename}")
    return model

if __name__ == "__main__":
    # Entry point for script execution
    #model = main('lgbm', 'None')
    #for modele in ['lgbm', 'random_forest', 'log']:
    #    for gestion_desequilibre in [None, 'class_weight', 'SMOTE']:
    #        main(modele, gestion_desequilibre)

    model = main('lgbm', 'None')
