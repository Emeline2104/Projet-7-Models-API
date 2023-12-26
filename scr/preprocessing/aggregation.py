"""
Script de prétraitement des données pour une tâche de classification de crédit.
Le script effectue le chargement des données, l'encodage one-hot, l'agrégation, et 
la création de nouvelles fonctionnalités.

Auteur: Emeline TAPIN
Date de création: 16/11/2023

Fonctions disponibles:
- one_hot_encoder: Effectue le codage one-hot des colonnes catégorielles d'un DataFrame.
- select_features: Sélectionne les caractéristiques avec un taux de remplissage d'au moins `threshold`
et selon une liste de features pré-défini.
- application_train_test: Charge les données d'entraînement et de test, effectue le prétraitement,
et crée de nouvelles fonctionnalités.
- bureau_and_balance: Agrège les données des tableaux bureau et bureau_balance et réalise 
des transformations.
- previous_applications: Traite et prépare les données du tableau previous_application.
- pos_cash: Traite et prépare les données du tableau POS_CASH_balance.
- installments_payments: Traite et prépare les données du tableau installments_payments.
- credit_card_balance: Traite les données de solde de carte de crédit, effectue des agrégations, 
et crée de nouvelles fonctionnalités.
- aggreger: Aggrège les différentes tables et réalise le pré-traitement global.

Exemple d'utilisation:
1. Charger les données d'entraînement et de test :
   df = application_train_test()

2. Réaliser l'agrégation des données bureau et bureau_balance :
   bureau_data = bureau_and_balance()

3. Traiter les données du tableau previous_application :
   previous_data = previous_applications()

4. Traiter les données du tableau POS_CASH_balance :
   pos_data = pos_cash()

5. Traiter les données du tableau installments_payments :
   installments_data = installments_payments()

6. Traiter les données du tableau credit_card_balance :
   credit_card_data = credit_card_balance()

7. Aggréger l'ensemble des données pour la modélisation :
   aggregated_data = aggreger()
"""
import sys
sys.path.append("/Users/beatricetapin/Documents/2023/Data Science/Projet_7_Modele_API/")
from config import (
    APPLICATION_TRAIN_FILENAME,
    APPLICATION_TEST_FILENAME,
    APPLICATION_TRAIN_FILENAME, 
    APPLICATION_TEST_FILENAME, 
    BUREAU_FILENAME, 
    BB_FILENAME, 
    PREV_FILENAME, 
    POS_FILENAME,
    INSTALLMENTS_PAYMENTS_FILENAME, 
    CREDIT_CARD_BALANCE_FILENAME, 
    DATA_AGGREG_FILENAME,
)
import gc
import pandas as pd
import numpy as np

# Définition de la fonction de codage one-hot
def one_hot_encoder(df, nan_as_category=True):
    """
    Effectue le codage one-hot des colonnes catégorielles d'un DataFrame.

    Args:
        df (DataFrame): Le DataFrame contenant les données.
        nan_as_category (bool, optional): Indique si les valeurs manquantes 
        doivent être traitées comme une catégorie. 
            Par défaut, True.

    Returns:
        DataFrame: Le DataFrame avec les colonnes catégorielles encodées en one-hot.
        list: La liste des nouvelles colonnes créées.
    """
    # Récupère les noms des colonnes d'origine
    original_columns = list(df.columns)

    # Sélectionne les colonnes catégorielles
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']

    # Effectue le codage one-hot des colonnes catégorielles
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)

    # Obtiens les noms des nouvelles colonnes créées
    new_columns = [c for c in df.columns if c not in original_columns]

    return df, new_columns

# Définition de la fonction de sélection des caractéristiques
def select_features(df, columns_list, columns_to_keep, threshold=0.7):
    """
    Sélectionne les caractéristiques avec un taux de remplissage d'au moins
    `threshold` et selon une liste de features pré-défini.

    Args:
        df (DataFrame): Le DataFrame contenant les données.
        feature_list (list): La liste des caractéristiques à considérer.
        threshold (float): Le seuil de taux de remplissage.

    Returns:
        selected_features (list): La liste des caractéristiques sélectionnées.
    """
    selected_columns = []

    for feature in columns_list:
        fill_rate = df[feature].isna().sum()/ len(df)
        if fill_rate >= threshold:
            selected_columns.append(feature)

    selected_columns.append(columns_to_keep)
    df = df[selected_columns]

    return df

# Définition de la fonction d'application_train_test
def application_train_test(nan_as_category=False, selected_columns=None):
    """
    Chargement des données d'entraînement et de test, prétraitement et création 
    de nouvelles fonctionnalités.

    :param nan_as_category: Convertir les valeurs NaN en catégories (True/False).
    :param selected_columns: Liste des colonnes à conserver (None pour tout conserver).
    
    :return: DataFrame contenant les données d'entraînement et de test traitées.
    """
    # Charge les données d'entraînement
    df = pd.read_csv(APPLICATION_TRAIN_FILENAME)
    test_df = pd.read_csv(APPLICATION_TEST_FILENAME)

    print("Échantillons d'entraînement : {}, échantillons de test : {}".format(len(df), len(test_df)))

    # Fusionne les données d'entraînement et de test
    df = pd.concat([df, test_df], axis=0).reset_index(drop=True)

    # Filtre les colonnes sélectionnées
    if selected_columns:
        df = df.loc[:, df.columns.isin(selected_columns)]

    # Supprime les 4 applications avec CODE_GENDER 'XNA'
    # (dans l'ensemble d'entraînement)
    df = df[df['CODE_GENDER'] != 'XNA']

    # Encode les caractéristiques catégorielles binaires (0 ou 1 ; deux catégories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], _ = pd.factorize(df[bin_feature])

    # Encode les caractéristiques catégorielles avec une technique "One-Hot"
    df, _ = one_hot_encoder(df, nan_as_category)

    # Remplace les valeurs NaN pour DAYS_EMPLOYED: 365.243 avec NaN
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # Crée de nouvelles caractéristiques simples (pourcentage)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # Supprime les données de test
    del test_df
    gc.collect()

    return df

# Définition de la fonction bureau_and_balance
def bureau_and_balance(nan_as_category=True):
    """
    Cette fonction réalise une agrégation des données des tableaux bureau et bureau_balance, 
    effectue des transformations et renvoie un nouveau dataframe contenant les résultats.

    :param nan_as_category: Détermine si les valeurs NaN doivent être traitées 
    comme une catégorie (par défaut à True).

    :return: Un dataframe contenant les agrégations et transformations 
    des données bureau et bureau_balance.
    """
    # Chargement des données
    bureau = pd.read_csv(BUREAU_FILENAME)
    bb = pd.read_csv(BB_FILENAME)

    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance : Réalise des agrégations et fusionne avec bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Caractéristiques numériques de bureau et bureau_balance
    num_aggregations = {
        'DAYS_CREDIT': ['sum', 'mean', 'max'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_SIZE': ['sum']
    }

    # Caractéristiques catégorielles de bureau et bureau_balance
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Bureau : Crédits actifs - en utilisant uniquement des agrégations numériques
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Bureau : Crédits clos - en utilisant uniquement des agrégations numériques
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()

    return bureau_agg

# Définition de la fonction previous application
def previous_applications(nan_as_category=True):
    """
    Cette fonction traite et prépare les données du tableau previous_application.

    :param nan_as_category: Détermine si les valeurs NaN doivent être traitées 
    comme une catégorie (par défaut à True).
    :return: Le dataframe previous_application traité.
    """
    # Chargement des données
    prev = pd.read_csv(PREV_FILENAME)

    # Encodage one-hot des caractéristiques catégorielles
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=nan_as_category)


    # Ajout de la caractéristique : pourcentage entre la valeur demandée et la valeur reçue
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # Agrégation des caractéristiques numériques
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }


    # Agrégation des caractéristiques catégorielles
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    # Agrégation des données pour chaque SK_ID_CURR
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
  
    # Applications précédentes approuvées : uniquement des caractéristiques numériques
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Applications précédentes refusées : uniquement des caractéristiques numériques
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    del refused, approved, prev, refused_agg, approved_agg, 
    gc.collect()

    return prev_agg

# Définition de la fonction pos_cash
def pos_cash(nan_as_category=True):
    """
    Cette fonction traite et prépare les données du tableau POS_CASH_balance.

    :param nan_as_category: Détermine si les valeurs NaN doivent être traitées comme une 
    catégorie (par défaut à True).
    :return: Le dataframe POS_CASH_balance traité.
    """
    pos = pd.read_csv(POS_FILENAME)

    # Encodage one-hot des caractéristiques catégorielles
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=nan_as_category)

    # Caractéristiques à agréger
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    # Agrégation des données pour chaque SK_ID_CURR
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # Comptage des comptes POS cash
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()

    del pos
    gc.collect()

    return pos_agg

# Définition de la fonction installments_payments
def installments_payments(nan_as_category=True):
    """
    Cette fonction traite et prépare les données du tableau installments_payments.

    :param nan_as_category: Détermine si les valeurs NaN doivent être traitées 
    comme une catégorie (par défaut à True).
    :return: Le dataframe installments_payments traité.
    """

    installments_payments_df = pd.read_csv(INSTALLMENTS_PAYMENTS_FILENAME)

    # Encodage one-hot des caractéristiques catégorielles
    ins, cat_cols = one_hot_encoder(installments_payments_df, nan_as_category=nan_as_category)

    # Pourcentage et différence payée dans chaque versement (montant payé et valeur du versement)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    # Jours de retard et jours avant l'échéance (pas de valeurs négatives)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    # Caractéristiques: Effectuer des agrégations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }

    for cat in cat_cols:
        aggregations[cat] = ['mean']

    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    # Comptage des comptes de versements
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()

    del ins
    gc.collect()

    return ins_agg

# Définition de la fonction credit_card_balance
def credit_card_balance(nan_as_category=True):
    """
    Traite les données de solde de carte de crédit, effectue des 
    agrégations et crée de nouvelles fonctionnalités.

    Args:
        nan_as_category (bool): Si True, traite les valeurs manquantes comme une catégorie.

    Returns:
        pd.DataFrame: Un DataFrame contenant les nouvelles fonctionnalités agrégées.

    """
    # Chargez les données du solde de la carte de crédit
    cc = pd.read_csv(CREDIT_CARD_BALANCE_FILENAME)

    # Applique l'encodage one-hot si nécessaire
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=nan_as_category)

    # Supprime la colonne SK_ID_PREV
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)

    # Caractéristiques: Effectuer des agrégations
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'sum'],
        'AMT_BALANCE': ['max', 'mean', 'sum'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean', 'sum'],
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'sum'], 
        'AMT_INST_MIN_REGULARITY': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max', 'mean', 'sum'],
        'AMT_RECIVABLE': ['min', 'max', 'mean', 'sum'],
        'AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_OTHER_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_MATURE_CUM': ['min', 'max', 'mean', 'sum'],
    }

    for cat in cat_cols:
        aggregations[cat] = ['mean']
 
    # Effectue des agrégations
    cc_agg = cc.groupby('SK_ID_CURR').agg(aggregations)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

    # Compte le nombre de lignes de carte de crédit
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

    # Nettoie la mémoire
    del cc
    gc.collect()

    return cc_agg

# Définition de la fonction aggreg
def aggreger(debug=False):
    """
    Fonction qui aggrège les différentes tables et réalise le pré-traitement
    (encodqge, valeurs abbérantes, etc.)

    Args:
        debug (bool): Indique si le mode de débogage est activé 
        (utilisation d'un nombre limité de lignes).

    Returns:
        None
    """

    # Déterminez le nombre de lignes à charger en fonction du mode de débogage
    num_rows = 10000 if debug else None

    # Chargez et préparez les données de l'application
    df = application_train_test(num_rows)
    print("Application shape:", df.shape)

    # Étape 1 : Bureau and Bureau Balance
    bureau = bureau_and_balance()
    print("Bureau df shape:", bureau.shape)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau
    gc.collect()

    # Étape 2 : Previous Applications
    prev = previous_applications(num_rows)
    print("Previous applications df shape:", prev.shape)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    del prev
    gc.collect()


    # Étape 3 : POS Cash
    pos = pos_cash(num_rows)
    print("Pos-cash balance df shape:", pos.shape)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    del pos
    gc.collect()

    # Étape 4 : Installments Payments
    ins = installments_payments(num_rows)
    print("Installments payments df shape:", ins.shape)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    del ins
    gc.collect()

    # Étape 5 : Credit Card Balance
    cc = credit_card_balance(num_rows)
    print("Credit card balance df shape:", cc.shape)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    del cc
    gc.collect()

    return df

if __name__ == '__main__':
    data = aggreger()

    data.to_csv(DATA_AGGREG_FILENAME, index=False)

    print(f"Fichier CSV enregistré avec succès à l'emplacement : {DATA_AGGREG_FILENAME}")
