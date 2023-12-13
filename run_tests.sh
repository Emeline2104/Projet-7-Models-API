#!/bin/bash

# Installer les dépendances ou les outils nécessaires
pip install -r requirements.txt

# Se déplace dans le répertoire racine du projet
cd $GITHUB_WORKSPACE

# Ajoute le répertoire actuel au PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Lance les tests unitaires
pytest scr/test_model.py

# Capture le code de sortie des tests
TEST_EXIT_CODE=$?

# Affiche le résultat des tests
if [ $TEST_EXIT_CODE -eq 0 ]; then
  echo "Les tests ont réussi."
else
  echo "Les tests ont échoué. Veuillez corriger les erreurs avant de continuer."
fi

# Renvoye le code de sortie des tests pour être utilisé par le script d'intégration continue
exit $TEST_EXIT_CODE
