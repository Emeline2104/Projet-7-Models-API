#!/bin/bash

# Installer les dépendances ou les outils nécessaires
pip install -r requirements.txt

# Lance les tests unitaires
pytest scr/tests/test_model.py

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
