# Projet-7 - Prêt à Dépenser - Modèle de Scoring & Dashboard

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
Un notebook dédié à l'analyse exploratoire et à l'analyse de la qualité des données a été créé ([EDA.ipynb](https://github.com/Emeline2104/Projet-7-Models-API/blob/main/notebook/EDA.ipynb).
#### 2. Exploration des méthodes de pré-traitement et de modèles de classification
Des pipelines ont été mis en place pour le pré-traitement des données et la modélisation de la prédiction des consommations énergétiques des bâtiments [main](https://github.com/Emeline2104/Projet-7-Models-API/tree/main/scr).
Cela est composé de la manière suivante : 
#### 3. Analyse Data Drift
#### 4. API

### Outils MLOps : 
#### MLFlow : 
Ce projet intègre MLflow, une plateforme open source pour la gestion du cycle de vie des modèles machine learning (ML). 
Les étapes majeures, de l'entraînement initial à l'enregistrement des modèles, sont enregistrées et suivies grâce à MLflow. 

### Livrables : 
#### Notebooks :
- Notebook de l'analyse exploratoire et de l'analyse de la qualité des données ([1_EDA.ipynb](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/notebooks/1_EDA.ipynb)); 
- Notebook exploratoire des méthodes utilisées (features engineering & modèles de prédiction) ([2_Prediction_consommation.ipynb](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/notebooks/2_Prediction_consommation.ipynb)); 
  
#### Scripts : 
- Script principal du projet (*[main.py](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/main.py)*) qui effectue les étapes suivantes :
  - Chargement des données à partir du fichier spécifié dans le fichier de configuration (*[config.py](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/scr/config.py)*); 
  - Nettoyage des données à l'aide d'un pipeline défini dans le module data_cleaning (*[data_cleaning.py](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/scr/preprocessing/data_cleaning.py)*);
  - Feature Engineering à l'aide d'un pipeline défini dans le module feature_engineering (*[feature_engineering.py](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/scr/preprocessing/feature_engineering.py)*);
  - Entraînement et évaluation d'un modèle de régression baseline (régression linéaire (RL)) en utilisant le pipeline défini dans le module baseline_model (*[baseline_model.py](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/scr/models/baseline_model.py)*);
  - Entraînement et évaluation d'un modèle XGBoost en utilisant le pipeline défini dans le module xgboost_model (*[xgboost_model.py](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/scr/models/xgboost_model.py)*).
    
#### Support de présentation : 
Un support de présentation de l'analyse exploratoire pour la soutenance est également disponible (*[3_Presentation](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/Presentation.pdf)*).

## Installation et exécution 

### Installation

Pour exécuter le code de ce projet, vous aurez besoin de Python 3.11 ou supérieur. Installez les dépendances à l'aide du fichier `requirements.txt`.

```bash
pip install -r requirements.txt
```

Le fichier setup.py est également inclus pour permettre l'installation et la distribution du projet en tant que package Python.
```bash
pip install .
```

### Execution du script
Pour exécuter le script, assurez-vous d'avoir Python 3.11 ou supérieur installé et exécutez la commande suivante dans le terminal :

```bash
python main.py
```
Assurez-vous également de personnaliser les chemins et les paramètres dans le fichier [config.py](https://github.com/Emeline2104/Predictive_energy_consumption/blob/master/scr/config.py) selon les besoins de votre projet.




# 
## A propos du projet : 

### Objectifs : 
- Développer un modèle de scoring pour prédire la probabilité de remboursement des clients.
- Créer un dashboard interactif pour les chargés de relation client.

### Données : 
- Télécharger les données ici : 
  
### Méthodologie : 
- Analyse exploratoire.
- Pré-traitement des données et modèles de scoring (dummyClassifier, regression logistique, Light GBM, Random Forest). 
- MLOps pour gérer le cycle de vie du modèle (MLFlow).
- Utilisation de Dash, Bokeh ou Streamlit pour le dashboard interactif. -> à maj
- Utilisation de la librairie evidently pour détecter le Data Drift. -> à maj
- 
### Livrables : 
- Notebook de l'analyse exploratoire et de l'analyse de la qualité des données (*A_MAJ.ipynb*).
- Notebooks pour chaque prédiction (pré-traitement et modèles de classification) (*A_MAJ.ipynb*).
- Support de présentation pour la soutenance (*A_MAJ.ipynb*).
- Autres livrables -> à compléter

## Installation

Pour exécuter le code de ce projet, vous aurez besoin de Python 3.11 ou supérieur. Installez les dépendances à l'aide du fichier `requirements.txt`. -> fichier à faire

```bash
pip install -r requirements.txt
