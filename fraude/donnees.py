"""Chargement et nettoyage du jeu de données public « Vehicle Insurance Claim Fraud » (Oracle).

Chaque ligne est une déclaration de sinistre automobile (1994-1996) avec une étiquette
FraudFound_P (1 = fraude avérée après enquête).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CHEMIN_DEFAUT = Path(__file__).resolve().parent.parent / "data" / "fraud_oracle.csv"

CIBLE = "FraudFound_P"

# Colonnes ajoutées par certaines copies du jeu de données, absentes de l'original.
# « Statement » recopie le type de garantie : on la retire pour éviter toute fuite.
COLONNES_PARASITES = ["Unnamed: 33", "Unnamed: 34", "Statement"]

# Libellés français pour l'affichage
TRADUCTIONS = {
    "BasePolicy": {"Liability": "RC seule", "Collision": "Collision", "All Perils": "Tous risques"},
    "Fault": {"Policy Holder": "Assuré", "Third Party": "Tiers"},
    "AccidentArea": {"Urban": "Urbaine", "Rural": "Rurale"},
    "PoliceReportFiled": {"Yes": "Oui", "No": "Non"},
    "WitnessPresent": {"Yes": "Oui", "No": "Non"},
}


def charger(chemin: str | Path = CHEMIN_DEFAUT) -> pd.DataFrame:
    """Charge le CSV, retire les colonnes parasites et ajoute un identifiant de dossier."""
    df = pd.read_csv(chemin)
    df = df.drop(columns=[c for c in COLONNES_PARASITES if c in df.columns])
    df[CIBLE] = df[CIBLE].astype(int)
    df["Dossier"] = "D-" + df["PolicyNumber"].astype(str).str.zfill(5)
    return df


def libelle(colonne: str, valeur) -> str:
    """Traduit une valeur pour l'affichage, ou la renvoie telle quelle."""
    return TRADUCTIONS.get(colonne, {}).get(valeur, str(valeur))
