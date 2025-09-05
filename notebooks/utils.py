# Fonction pour séparation de notre X et y
def prepare_xy(donnees_modelisation, target_col="a_quitte_l_entreprise"):

    y = donnees_modelisation[target_col]
    X = donnees_modelisation.drop(columns=[target_col])
    return X, y

# Fonction pour récupérer les features à scaler ou non

def scaler_ou_non():

    features_a_scaler = [
    'revenu_mensuel','annees_dans_l_entreprise','distance_domicile_travail',
    'annees_depuis_la_derniere_promotion','experience_externe','score_satisfaction',
    'augmentation_par_formation','pee_par_anciennete','niveau_education']
    features_encodees = [
    'genre','heure_supplementaires',
    'frequence_deplacement','a_suivi_formation','tranche_age','statut_marital_Celibataire',
    'statut_marital_Divorce','statut_marital_Marie','promotion_recente','poste_AssistantdeDirection',
    'poste_CadreCommercial','poste_Consultant','poste_DirecteurTechnique','poste_Manager',
    'poste_ReprésentantCommercial','poste_RessourcesHumaines','poste_SeniorManager','poste_TechLead']

    return features_a_scaler, features_encodees
