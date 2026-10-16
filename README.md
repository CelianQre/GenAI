# Programmateur

Application Streamlit du projet **IA générative : agents intelligents et raisonnement avancé**.
Elle transforme une demande fonctionnelle en une implémentation Python et un fichier de tests `pytest`.
Les deux fichiers sont affichés et peuvent être téléchargés depuis l'interface.

## Installation

Installer [Ollama](https://ollama.com/), puis télécharger le modèle local :

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b
streamlit run app.py
```

Le modèle par défaut est `llama3.2:3b`. Il peut être modifié dans un fichier `.env` :

```text
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_TIMEOUT=90
OLLAMA_MAX_TOKENS=3600
```

L'application utilise l'API compatible OpenAI d'Ollama. Aucune clé API ni aucun service payant
ne sont nécessaires.

## Fonctionnement de l'agent

Le sujet choisi est celui du **programmateur** : l'utilisateur décrit une fonctionnalité, puis
l'agent produit le code et ses tests unitaires.

Les techniques **Chain of Thought**, **Tree of Thoughts**, **ReAct** et
**Self-Correction** sont utilisées sans afficher la chaîne de pensée détaillée du modèle.

Le déroulement est le suivant :

1. **Chain of Thought** : décomposer la demande en contraintes, entrées, sorties et cas limites ;
2. **Tree of Thoughts** : comparer plusieurs approches et retenir la plus simple et robuste ;
3. **ReAct** : générer les fichiers, puis observer leur cohérence et leur syntaxe ;
4. **Self-Correction** : corriger le brouillon si la validation détecte une erreur ;
5. afficher le résultat validé.

La validation vérifie notamment la présence de `import pytest` et d'au moins une fonction `test_*`.
En cas d'échec, le brouillon et les erreurs sont renvoyés au modèle pour une nouvelle génération.

Après la génération, le bloc repliable **Raisonnement de l'agent** présente une synthèse des étapes :
analyse de la demande, génération, observation des résultats et corrections appliquées. Cette trace
est volontairement courte et vérifiable ; elle ne montre pas la chaîne de pensée privée du modèle.

Le raisonnement interne détaillé du modèle n'est pas affiché ; seules les étapes vérifiables et
le résultat sont conservés.

## Fichiers

- `app.py` : interface Streamlit, génération Ollama, validation et auto-correction.
- `requirements.txt` : dépendances Python.
