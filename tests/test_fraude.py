import pandas as pd

from fraude import ALERTE, PAS_D_ALERTE, REVUE_STANDARD, Trieur, charger, scenarios_retenus
from fraude.evaluation import ANNEES_APPRENTISSAGE, intervalle_wilson


def _donnees():
    return charger()


def test_colonnes_parasites_retirees():
    df = _donnees()
    assert "Statement" not in df.columns
    assert len(df) == 15420


def test_seul_le_contrat_recent_tient_dans_le_temps():
    df = _donnees()
    # Sur 1994-1995 seules, plusieurs signaux semblent tenir...
    app = df[df["Year"].isin(ANNEES_APPRENTISSAGE)]
    assert len(scenarios_retenus(app)) > 1
    # ... mais un seul résiste à l'année suivante.
    assert [s.code for s in scenarios_retenus(df)] == ["CONTRAT_RECENT"]


def test_triage_trois_niveaux():
    df = _donnees()
    trieur = Trieur(df, scenarios_retenus(df))
    base = df.iloc[0].copy()

    rc = base.copy(); rc["BasePolicy"] = "Liability"
    assert trieur.evaluer(rc).niveau == PAS_D_ALERTE

    std = base.copy()
    std.update({"BasePolicy": "Collision", "Fault": "Policy Holder",
                "Days_Policy_Accident": "more than 30", "Days_Policy_Claim": "more than 30"})
    assert trieur.evaluer(std).niveau == REVUE_STANDARD

    recent = std.copy(); recent["Days_Policy_Accident"] = "1 to 7"
    assert trieur.evaluer(recent).niveau == ALERTE


def test_wilson():
    bas, haut = intervalle_wilson(5, 20)
    assert 0.1 < bas < 0.25 < haut < 0.5
