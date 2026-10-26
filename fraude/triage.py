"""Triage d'un dossier : ce que le gestionnaire voit au moment de décider."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .evaluation import Taux, taux
from .scenarios import Scenario, dans_perimetre

PAS_D_ALERTE = "Pas d'alerte"
REVUE_STANDARD = "Revue standard"
ALERTE = "Alerte"


@dataclass
class Verdict:
    niveau: str
    raison: str
    scenarios: list[Scenario] = field(default_factory=list)
    reference: Taux | None = None  # taux de fraude observé sur les dossiers comparables


class Trieur:
    """Applique le découpage. Les taux de référence viennent des années d'apprentissage."""

    def __init__(self, historique: pd.DataFrame, retenus: list[Scenario]):
        self.retenus = retenus
        perim = dans_perimetre(historique)
        self.taux_hors = taux(historique, ~perim)
        self.taux_perim = taux(historique, perim)
        self.taux_scenario = {
            s.code: taux(historique, perim & s.masque(historique)) for s in retenus
        }

    def niveaux(self, df: pd.DataFrame) -> pd.Series:
        perim = dans_perimetre(df)
        niveau = pd.Series(PAS_D_ALERTE, index=df.index)
        niveau[perim] = REVUE_STANDARD
        for s in self.retenus:
            niveau[perim & s.masque(df)] = ALERTE
        return niveau

    def evaluer(self, dossier: pd.Series) -> Verdict:
        ligne = dossier.to_frame().T
        if not bool(dans_perimetre(ligne).iloc[0]):
            if dossier["BasePolicy"] == "Liability":
                raison = "Contrat RC seule : l'assureur n'indemnise pas le véhicule de l'assuré."
            else:
                raison = "Le tiers est responsable : l'indemnisation passe par son assureur."
            return Verdict(PAS_D_ALERTE, raison, reference=self.taux_hors)

        declenches = [s for s in self.retenus if bool(s.masque(ligne).iloc[0])]
        if declenches:
            s = declenches[0]
            return Verdict(ALERTE, s.nom, declenches, self.taux_scenario[s.code])
        return Verdict(
            REVUE_STANDARD,
            "Assuré responsable avec garantie dommages, sans signal de scénario.",
            reference=self.taux_perim,
        )
