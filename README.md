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

Le choix principal est une architecture **ReAct + Self-Correction**. Ce choix est cohérent avec
le sujet du programmateur : l'agent doit produire du code, utiliser un outil d'exécution et
réagir à l'observation obtenue.

Une **Chain of Thought structurée** et un **Tree of Thoughts limité** complètent ce fonctionnement.
L'application demande une analyse courte, au moins deux approches candidates et une justification
du choix, sans afficher la chaîne de pensée privée du modèle.

Le déroulement est le suivant :

1. **Analyse structurée** : identifier les contraintes, entrées, sorties et cas limites ;
2. **Tree of Thoughts limité** : proposer deux approches, puis retenir la plus adaptée ;
3. **ReAct** : générer les fichiers, exécuter l'outil pytest et observer sa sortie ;
4. **Self-Correction** : corriger le brouillon si la validation ou pytest détecte une erreur ;
5. afficher le résultat validé.

La validation vérifie la syntaxe AST, la présence de `import pytest`, l'import de `solution.py` et
d'au moins une fonction `test_*`. Un outil dédié écrit ensuite les fichiers dans un dossier temporaire
et exécute réellement `pytest`. La sortie du test (succès ou erreur) devient l'observation de l'agent.
En cas d'échec de syntaxe ou de test, le brouillon et l'observation sont renvoyés au modèle pour une
nouvelle génération, jusqu'à six tentatives. Si le Python utilisé par Streamlit ne contient pas
pytest, l'application recherche automatiquement celui de `.venv` avant d'exécuter les tests.

Après la génération, le bloc repliable **Raisonnement de l'agent** présente l'analyse retournée,
les pistes candidates, le choix effectué, l'action pytest, sa sortie réelle et les corrections
appliquées. Cette trace est courte et vérifiable ; elle ne montre pas la chaîne de pensée privée
du modèle.

Le raisonnement interne détaillé du modèle n'est pas affiché ; seules les étapes vérifiables et
le résultat sont conservés.

## Méthodes écartées

- **Chain of Thought détaillée** : écartée sous sa forme brute, car afficher la chaîne de pensée
    privée n'est pas nécessaire pour évaluer l'agent. Elle est remplacée par une analyse structurée
    et vérifiable.
- **Tree of Thoughts complet** : écarté, car le sujet ne nécessite pas un grand arbre de recherche
    ni l'exploration de nombreuses branches. Deux approches candidates suffisent pour comparer la
    simplicité, la robustesse et la testabilité d'une solution Python.
- **Conversation libre de type chatbot** : écartée, car elle ne fournit ni outil, ni validation,
	ni boucle de correction explicite.

## Fichiers

- `app.py` : interface Streamlit, génération Ollama, validation et auto-correction.
- `requirements.txt` : dépendances Python.
