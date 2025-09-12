# LIBRAIRIES
# Librairies "classiques"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from IPython.display import Image, display
import shap

# Librairies scikit-learn
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV, 
    cross_validate,
)
# Preprocess et modèles
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, auc, f1_score, balanced_accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool



# FONCTION POUR SEPARATION DE NOTRE X ET Y
def prepare_xy(donnees_features, target_col="a_quitte_l_entreprise"):
# Suppression de 3 colonnes redondantes par matrice de Pearson
    colonnes_a_supprimer = ['niveau_hierarchique_poste','annees_dans_le_poste_actuel','annes_sous_responsable_actuel']
    donnees_features = donnees_features.drop(columns = colonnes_a_supprimer)
# Plusieurs étapes de feature engineering
# Ajout de nouvelles variables
    donnees_features['experience_externe'] = donnees_features['annee_experience_totale'] - donnees_features['annees_dans_l_entreprise']
    
    donnees_features['score_satisfaction'] = (donnees_features[
    'satisfaction_employee_environnement'] + donnees_features['satisfaction_employee_nature_travail']
    + donnees_features['satisfaction_employee_equipe']+ donnees_features['satisfaction_employee_equilibre_pro_perso'])/4

    donnees_features['augmentation_par_formation'] = (donnees_features[
        'augmentation_salaire_precedente_pourcent']*100) / (donnees_features['nb_formations_suivies']+1)

    donnees_features['pee_par_anciennete'] = donnees_features['nombre_participation_pee'] / (donnees_features['annees_dans_l_entreprise']+1)
# Suppression de 3 colonnes redondantes par matrice de Spearman
    donnees_features = donnees_features.drop(columns='nombre_participation_pee')
    donnees_features = donnees_features.drop(columns='nombre_experiences_precedentes')
    donnees_features = donnees_features.drop(columns='annee_experience_totale')
# Modification des colonnes avec encodage
    donnees_features["a_suivi_formation"] = (donnees_features["nb_formations_suivies"] >= 1).astype(int)
    donnees_features = donnees_features.drop(columns='nb_formations_suivies')

    donnees_features['tranche_age'] = pd.cut(donnees_features[
        'age'],bins=[17, 30, 36, 43, 60],
        labels=['18-30', '31-36', '37-43','44+'])
    labelencoder = LabelEncoder()
    donnees_features['tranche_age']= labelencoder.fit_transform(donnees_features['tranche_age'])
    donnees_features = donnees_features.drop(columns='age')

    donnees_features['genre'] = donnees_features['genre'].map({'F': 1, 'M': 0})

    donnees_features = pd.get_dummies(donnees_features, columns=["statut_marital"], dtype=int)

    donnees_features = pd.get_dummies(donnees_features, columns=["departement"], dtype=int)

    donnees_features = pd.get_dummies(donnees_features, columns=["poste"], dtype=int)

    donnees_features = pd.get_dummies(donnees_features, columns=["domaine_etude"], dtype=int)

    donnees_features['heure_supplementaires'] = donnees_features['heure_supplementaires'].map({'Oui': 1, 'Non': 0})

    map_frequence = {"Aucun": 0, "Occasionnel": 1, "Frequent": 2}
    donnees_features["frequence_deplacement"] = donnees_features["frequence_deplacement"].map(map_frequence)

    donnees_features['promotion_recente'] = (donnees_features['annees_depuis_la_derniere_promotion'] <= 2).astype(int)

    donnees_features['a_quitte_l_entreprise'] = donnees_features['a_quitte_l_entreprise'].map({'Oui': 1, 'Non': 0})

    y = donnees_features[target_col]
    X = donnees_features.drop(columns=[target_col])
    return X, y


# FONCTION POUR RECUPERER LES FEATURES A SCALER OU NON

def scaler_ou_non():

    features_a_scaler = [
        'revenu_mensuel','annees_dans_l_entreprise','satisfaction_employee_environnement','note_evaluation_precedente','satisfaction_employee_nature_travail','satisfaction_employee_equipe',
        'satisfaction_employee_equilibre_pro_perso','note_evaluation_actuelle','augmentation_salaire_precedente_pourcent','distance_domicile_travail','niveau_education','annees_depuis_la_derniere_promotion',
        'experience_externe','score_satisfaction','augmentation_par_formation','pee_par_anciennete'
    ]
    features_encodees = [
        'genre','heure_supplementaires','frequence_deplacement','a_suivi_formation','tranche_age','statut_marital_Celibataire','statut_marital_Divorce',
        'statut_marital_Marie','departement_Commercial','departement_Consulting','departement_RessourcesHumaines','poste_AssistantdeDirection','poste_CadreCommercial',
        'poste_Consultant','poste_DirecteurTechnique','poste_Manager','poste_ReprésentantCommercial','poste_RessourcesHumaines','poste_SeniorManager','poste_TechLead',
        'promotion_recente','domaine_etude_Autre','domaine_etude_Entrepreunariat','domaine_etude_InfraCloud','domaine_etude_Marketing','domaine_etude_RessourcesHumaines',
        'domaine_etude_TransformationDigitale']

    return features_a_scaler, features_encodees

# FONCTION POUR GENERATION DES MODELES
def modelisation(
    model,
    train_path="../Data/Processed/train_data_df.csv",
    test_path="../Data/Processed/test_data_df.csv",
    target_col="a_quitte_l_entreprise",
    n_splits=3,
    n_jobs=-1
):
    # Charger les splits figés
    train_data_df = pd.read_csv(train_path)
    test_data_df = pd.read_csv(test_path)

    X_train = train_data_df.drop(columns=[target_col])
    y_train = train_data_df[target_col]

    X_test = test_data_df.drop(columns=[target_col])
    y_test = test_data_df[target_col]

    features_a_scaler, features_encodees = scaler_ou_non()
    # Préprocessing et pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), features_a_scaler),
            ("cat", "passthrough", features_encodees)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Validation croisée
    scoring = ["precision","recall","f1","average_precision","balanced_accuracy"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = cross_validate(
        pipeline,
        X_train, y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=n_jobs
    )

    print("=== Résultats CV (train vs val) ===")
    for metric in scoring:
        tr = cv_results[f"train_{metric}"]
        te = cv_results[f"test_{metric}"]
        print(f"{metric:18s}: train {tr.mean():.3f} ± {tr.std():.3f} "
              f"vs val {te.mean():.3f} ± {te.std():.3f}")

    # Fit final
    pipeline.fit(X_train, y_train)

    # Prédictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test  = pipeline.predict(X_test)

# Classification reports
    print(" Classification Report — TRAIN")
    print(classification_report(y_train, y_pred_train))

    print("Classification Report — TEST")
    print(classification_report(y_test, y_pred_test))

    return pipeline