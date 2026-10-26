"""Découpage du problème : un périmètre, puis des scénarios concrets.

La fraude n'est pas un score unique. On procède en deux temps :

1. Le périmètre : seuls les sinistres où l'assureur indemnise le véhicule de l'assuré
   (garantie Collision ou Tous risques) ET où l'assuré est responsable laissent une
   place réelle à la fraude. Hors de ce périmètre, l'outil ne dit rien.
2. Les scénarios : à l'intérieur du périmètre, un signal ne devient une alerte que s'il
   correspond à un mécanisme de fraude identifiable ET s'il tient chaque année.

Chaque scénario porte ses données, son signal et la décision attendue du gestionnaire.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

Masque = Callable[[pd.DataFrame], pd.Series]


def dans_perimetre(df: pd.DataFrame) -> pd.Series:
    """Garantie dommages (Collision / Tous risques) et assuré responsable."""
    return (df["BasePolicy"] != "Liability") & (df["Fault"] == "Policy Holder")


@dataclass(frozen=True)
class Scenario:
    code: str
    nom: str
    mecanisme: str          # ce que fait le fraudeur
    donnees: str            # colonnes utilisées, en clair
    masque: Masque          # le signal, en code
    verifications: list[str] = field(default_factory=list)  # la décision du gestionnaire


def _contrat_recent(df: pd.DataFrame) -> pd.Series:
    recents_accident = ["none", "1 to 7", "8 to 15", "15 to 30"]
    recents_declaration = ["none", "8 to 15", "15 to 30"]
    return df["Days_Policy_Accident"].isin(recents_accident) | df["Days_Policy_Claim"].isin(
        recents_declaration
    )


# Scénarios candidats. Tous sont testés ; seuls ceux qui tiennent dans le temps deviennent
# des alertes (voir evaluation.stabilite).
CANDIDATS: list[Scenario] = [
    Scenario(
        code="CONTRAT_RECENT",
        nom="Sinistre juste après la souscription",
        mecanisme="Le véhicule est assuré après un accident déjà survenu, ou en vue d'un sinistre prévu.",
        donnees="Délai entre la souscription et l'accident, et entre la souscription et la déclaration.",
        masque=_contrat_recent,
        verifications=[
            "Comparer la date et l'heure de souscription avec la date de l'accident déclarée.",
            "Demander le relevé d'information de l'assureur précédent.",
            "Faire vérifier par l'expert si les dommages sont antérieurs (rouille, traces anciennes).",
        ],
    ),
    Scenario(
        code="DEMENAGEMENT",
        nom="Déménagement récent avant le sinistre",
        mecanisme="Changement d'adresse pour brouiller l'historique ou le lieu de garage.",
        donnees="Délai depuis le dernier changement d'adresse.",
        masque=lambda df: df["AddressChange_Claim"].isin(["under 6 months", "1 year", "2 to 3 years"]),
        verifications=["Vérifier l'adresse de garage et la cohérence avec le lieu de l'accident."],
    ),
    Scenario(
        code="TOUS_RISQUES",
        nom="Garantie tous risques",
        mecanisme="Plus la garantie est large, plus un dommage arrangé est indemnisé.",
        donnees="Type de garantie.",
        masque=lambda df: df["BasePolicy"] == "All Perils",
    ),
    Scenario(
        code="HAUT_DE_GAMME",
        nom="Véhicule de plus de 69 000",
        mecanisme="Réparations surévaluées sur un véhicule coûteux.",
        donnees="Prix du véhicule.",
        masque=lambda df: df["VehiclePrice"] == "more than 69000",
    ),
    Scenario(
        code="BAS_DE_GAMME",
        nom="Véhicule de moins de 20 000",
        mecanisme="Véhicule peu coûteux « mis à la casse » via un sinistre.",
        donnees="Prix du véhicule.",
        masque=lambda df: df["VehiclePrice"] == "less than 20000",
    ),
    Scenario(
        code="ZONE_RURALE",
        nom="Accident en zone rurale",
        mecanisme="Accident mis en scène loin des témoins.",
        donnees="Zone de l'accident.",
        masque=lambda df: df["AccidentArea"] == "Rural",
    ),
    Scenario(
        code="VEHICULE_RECENT",
        nom="Véhicule de moins de 5 ans",
        mecanisme="Véhicule récent dont la valeur à neuf est réclamée.",
        donnees="Âge du véhicule.",
        masque=lambda df: df["AgeOfVehicle"].isin(["new", "2 years", "3 years", "4 years"]),
    ),
    Scenario(
        code="PREMIER_SINISTRE",
        nom="Aucun sinistre antérieur",
        mecanisme="Nouveau client sans historique.",
        donnees="Nombre de sinistres passés.",
        masque=lambda df: df["PastNumberOfClaims"] == "none",
    ),
]

# Variables volontairement exclues des alertes, même si elles sont corrélées à la fraude :
# elles décrivent la personne, pas le sinistre, et exposeraient à de la discrimination.
EXCLUES = {
    "AgeOfPolicyHolder": "âge de l'assuré",
    "Sex": "sexe",
    "MaritalStatus": "situation familiale",
}


def par_code(code: str) -> Scenario:
    return next(s for s in CANDIDATS if s.code == code)
