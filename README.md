# Classifiez automatiquement des informations
L’objectif de ce projet est de prédire la probabilité de démission des employés de l’ESN TechNova Partners.
Pour cela, nous explorons les données RH, appliquons plusieurs modèles de machine learning et comparons leurs performances afin d’identifier les facteurs clés influençant les départs.

## Table des matières
- [Problématique](#problematique)
- [Données utilisées](#donnees-utilisees)
- [Organisation du projet](#organisation-du-projet)
- [Installation et utilisation](#organisation-du-projet)

### Problématique
L'objectif ici est :
- de préparer une analyse exploratoire afin de déterminer les différences entre les salariés ayant démissionné et ceux qui sont encore dans l’entreprise.
- de réaliser des tests des différentes modèles supervisés.
- la détermination des facteurs principaux impactant le plus le modèle.

### Données utilisées
Trois fichiers csv de l'entreprise :
- extrait_eval.csv : évaluations des employés
- extrait_sirh.csv : données RH
- extrait_sondage.csv : résultats d’enquêtes internes

Après traitement, les données sont séparées en :
- train_data_df.csv (jeu d’apprentissage)
- test_data_df.csv (jeu de test)

### Organisation du projet
├── Data
│   ├── Raw/          # Données brutes (csv fournis par l’entreprise)
│   └── Processed/    # Données nettoyées et préparées
│
├── Graph/            # Visualisations générées (EDA, arbres de décision, etc.)
│
├── notebooks/        # Étapes du projet sous forme de notebooks
│   ├── notebook_1_analyse_exploratoire.ipynb
│   ├── notebook_2_feature_engineering.ipynb
│   ├── notebook_3_modele_classification.ipynb
│   ├── notebook_4_amelioration_classification.ipynb
│   └── notebook_5_optimisation_interpretation.ipynb
│
├── main.py           # Script principal d’exécution
├── utils.py          # Fonctions utilitaires
├── README.md         # Documentation du projet
└── pyproject.toml    # Dépendances et configuration

### Installation et utilisation
1. Cloner le projet :
``` 
git clone https://github.com/ton-repo/Classifiez_automatiquement_des_informations.git
cd Classifiez_automatiquement_des_informations
```
2. Installer les dépendances :
Le projet utilise pyproject.toml pour la gestion des dépendances.
```
poetry install
```
3. Ouvrir le projet dans VS Code
```
code .
```
4. Configurer l’environnement Python dans VS Code
	1.	Installez l’extension Python (si ce n’est pas déjà fait).
	2.	Appuyez sur Ctrl+Shift+P (Windows/Linux) ou Cmd+Shift+P (Mac).
	4.	Recherchez “Python: Select Interpreter”.
	5.	Sélectionnez l’environnement créé par Poetry ou celui dans lequel tu as installé le projet.

5. Travailler avec les notebooks
- Les notebooks se trouvent dans le dossier :
```
notebooks/
```

- Ouvrez n’importe quel fichier .ipynb (ex. notebook_1_analyse_exploratoire.ipynb).
- VS Code activera automatiquement l’éditeur interactif Jupyter.
- Vous pouvez exécuter les cellules avec Shift+Enter.

### Résultats attendus
- Scores de performance des différents modèles (précision, rappel, F1-score).
- Visualisations des features les plus importantes.
- Interprétation avec SHAP et autres méthodes d’explicabilité.

### Résultats de la validation croisée de 2 modèles non-linéaires
<img width="1278" height="555" alt="Capture d’écran 2025-09-25 à 15 49 29" src="https://github.com/user-attachments/assets/7c3933f2-10c1-40eb-b59d-5b038642ee28" />

### Optimisation des paramètres pour le meilleur modèle
<img width="1335" height="688" alt="Capture d’écran 2025-09-25 à 15 49 47" src="https://github.com/user-attachments/assets/ba230742-d8ac-4f12-a21c-cfd50f64e12f" />
