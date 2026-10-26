"""Triage de la fraude à l'assurance automobile, par découpage en scénarios."""
from .donnees import CIBLE, charger
from .evaluation import backtest, scenarios_retenus, tableau_candidats
from .scenarios import CANDIDATS, dans_perimetre
from .triage import ALERTE, PAS_D_ALERTE, REVUE_STANDARD, Trieur

__all__ = [
    "CIBLE", "charger", "backtest", "scenarios_retenus", "tableau_candidats",
    "CANDIDATS", "dans_perimetre", "ALERTE", "PAS_D_ALERTE", "REVUE_STANDARD", "Trieur",
]
