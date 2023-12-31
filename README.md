# Projet-7 - Prêt à Dépenser - Modèle de Scoring & API

[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Emeline2104/Projet-7-Models-API/Tests?label=Tests)](https://github.com/Emeline2104/Projet-7-Models-API/actions/workflows/tests.yml)

Ce projet a été réalisé dans le cadre de la formation diplomante de Data Scientist d'OpenClassRooms & CentraleSupelec.

## A propos du projet : 
Ce projet "Prêt à Dépenser" vise à développer un modèle de scoring prédictif pour évaluer la probabilité de remboursement des clients dans le secteur financier. En parallèle, la création d'un dashboard interactif transparent permettra aux chargés de relation client d'interpréter les prédictions du modèle, répondant ainsi à la demande croissante des clients en matière de transparence dans le processus d'octroi de crédit.

### Objectifs : 
- Développer un modèle de scoring pour prédire la probabilité de remboursement des clients (repository ci-dessous).
- Créer un dashboard interactif pour les chargés de relation client ([repository ci-après](https://github.com/Emeline2104/Projet-7-Dashboard)).
  
### Données : 
Les données nécessaires au projet sont disponibles [ici](https://www.kaggle.com/c/home-credit-default-risk/data).
Elles incluent des informations comportementales et financières.

### Méthodologie : 
#### 1. Analyse exploratoire des données
Un notebook dédié à l'analyse exploratoire et à l'analyse de la qualité des données a été créé ([*EDA.ipynb*](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/notebook/EDA.ipynb)).
#### 2. Exploration des méthodes de pré-traitement et de modèles de classification
Un pipeline a été mis en place pour le pré-traitement des données et les classification d'octroi de crédit (regression logistique, random forest, LGBM) ([*main.py*](https://github.com/Emeline2104/Projet-7-Models-API/tree/main/scr)).
Ce projet intègre MLflow, une plateforme open source pour la gestion du cycle de vie des modèles machine learning. Les étapes majeures, de l'entraînement initial à l'enregistrement des modèles, sont enregistrées et suivies grâce à MLflow. 
#### 3. Analyse Data Drift
Une analyse de data drift a été réalisé entre les données d'entrainement et test ([*data_drift.py*](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/scr/data_drift_analysis/data_drift.py)) et ([data_drift_report.html](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/scr/data_drift_analysis/data_drift_report.html)).
#### 4. API
Une API Flask a été déployé avec le modèle sélectionné ([*flask_api.py*](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/scr/flask_api.py)) ainsi que les test unitaires nécessaires pour le déploiement automatique ([*test_models.py*](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/scr/test_model.py)).

### Livrables : 

#### Notebooks :
- Notebook de l'analyse exploratoire et de l'analyse de la qualité des données ([EDA.ipynb](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/notebook/EDA.ipynb)); 
  
#### Scripts : 
##### Script de modélisation : Traitant du prétraitement à la prédiction intégrant via MLFlow le tracking d’expérimentations et le stockage centralisé des modèles
- Script principal du projet (*[main.py](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/scr/models/main.py)*) qui effectue les étapes suivantes :
  - Chargement des données à partir du fichier spécifié dans le fichier de configuration (*[config.py](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/scr/config.py)*);
  - Aggrégation des données (*[aggregation.py](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/scr/preprocessing/aggregation.py)*);
  - Nettoyage des données et feature engineering à l'aide d'un pipeline défini dans le module pre_processing (*[pre_processing.py](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/scr/preprocessing/pre_processing.py)*);
  - Fonctions d'entraînement et évaluation de modèles en utilisant le pipeline défini dans le module model_selec (*[model_training.py](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/scr/models/model_training.py)*);
  - Pipeline d'entrainement et évaluation de modèles pour les différents alogorithme de classification (DummyClassifier, regression logistique, random forest, LGBM) (*[models_selec.py](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/scr/models/models_selec.py)*);
  - Analyse de la feature importance (*[feature_importance.py](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/scr/models/feature_importance.py)*)

##### Scripts de déploiement du modèle via API 
- Script principal de l'API ([*flask_api.py*](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/scr/flask_api.py)).
  
#### Note méthodologique 
La [note méthodologique](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/TAPIN_Emeline_3_note_m%C3%A9thodologique_122023.pdf) comprenant : 
  - La méthodologie d'entraînement du modèle ; 
  - Le traitement du déséquilibre des classes ; 
  - La fonction coût métier, l'algorithme d'optimisation et la métrique d'évaluation ;
  - Un tableau de synthèse des résultats ; 
  - L’interprétabilité globale et locale du modèle ; 
  - Les limites et les améliorations possibles ; 
  - L’analyse du Data Drift. 

#### Support de présentation : 
Un support de présentation pour la soutenance, détaillant le travail réalisé (*[TAPIN_Emeline_4_presentation_072023.pdf](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/TAPIN_Emeline_4_presentation_072023.pdf)*).

### Le tableau HTML d’analyse de data drift 
Tableau HTML du data drift réalisé à partir d’evidently (*[data_drift_report.html](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/scr/data_drift_analysis/data_drift_report.html)*).


## Structure du Projet

Le projet est organisé de la manière suivante :
- **.github/workflows**: 
  - **tests.yml**: Fichier de configuration pour les workflows GitHub.
- **scr**: Le répertoire principal du code source.
  - **data_drift_analysis**: Contient les scripts liés à l'analyse de data drift.
    - **data_drift.py**: Script d'analyse de data drift.
    - **data_drift_report.html**: Rapport HTML généré à partir de l'analyse de data drift.
  - **flask_api.py**: Script principal de l'API Flask.
  - **models**: Contient les scripts liés à la modélisation.
    - **feature_importance.py**: Script pour l'analyse de l'importance des fonctionnalités.
    - **main.py**: Script principal pour le prétraitement et l'entraînement des modèles.
    - **model_training.py**: Script contenant les fonctions d'entraînement des modèles.
    - **models_selec.py**: Script pour la sélection des modèles.
  - **models_saved**: Contient les modèles et les variables sauvegardés.
  - **preprocessing**: Scripts pour le prétraitement des données.
    - **aggregation.py**: Script pour l'agrégation des données.
    - **pre_processing.py**: Script pour le nettoyage et l'ingénierie des fonctionnalités.
  - **test_model.py**: Script pour les tests unitaires du modèle.
- **config.py**: Fichier de configuration pour le projet.
- **.gitignore**: Fichier spécifiant les fichiers et dossiers à ignorer dans le suivi git.
- **Procfile**: Fichier spécifiant les commandes à exécuter lors du déploiement de l'application.
- **README.md**: Documentation principale du projet.
- **makefile**: Fichier de configuration pour la compilation et l'exécution du projet.
- **requirements.txt**: Liste des dépendances du projet.
- **run_tests.sh**: Script pour exécuter les tests du projet sur GitHub.
- **runtime.txt**: Fichier spécifiant la version de Python à utiliser pour le projet.

## Exigences

### Installation

Pour exécuter le code de ce projet, vous aurez besoin de Python 3.11 ou supérieur. Installez les dépendances à l'aide du fichier `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Execution du script
Pour exécuter le script, assurez-vous d'avoir Python 3.11 ou supérieur installé et exécutez la commande suivante dans le terminal :

```bash
python scr/models/main.py
```
Assurez-vous également de personnaliser les chemins et les paramètres dans le fichier *[config.py](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/config.py)* selon les besoins de votre projet.
Pour exécuter le code de ce projet, vous aurez besoin de Python 3.11 ou supérieur. Installez les dépendances à l'aide du fichier `requirements.txt`.

```bash
pip install -r requirements.txt
