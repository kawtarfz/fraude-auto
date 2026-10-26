"""Évaluation honnête : apprendre sur le passé (1994-1995), juger sur l'année suivante (1996)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .donnees import CIBLE
from .scenarios import CANDIDATS, Scenario, dans_perimetre

ANNEES_APPRENTISSAGE = (1994, 1995)
ANNEE_TEST = 1996

# Un signal devient une alerte s'il multiplie le taux de fraude du périmètre
# par au moins ce facteur, chaque année, apprentissage comme test.
SEUIL_STABILITE = 1.2


def intervalle_wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance à 95 % d'une proportion, utile pour les petits effectifs."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    marge = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - marge), min(1.0, centre + marge))


@dataclass
class Taux:
    fraudes: int
    dossiers: int

    @property
    def valeur(self) -> float:
        return self.fraudes / self.dossiers if self.dossiers else float("nan")

    @property
    def intervalle(self) -> tuple[float, float]:
        return intervalle_wilson(self.fraudes, self.dossiers)


def taux(df: pd.DataFrame, masque: pd.Series) -> Taux:
    sous = df.loc[masque, CIBLE]
    return Taux(int(sous.sum()), int(sous.size))


def stabilite(df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    """Pour chaque année : taux de fraude avec / sans le signal, à l'intérieur du périmètre."""
    perim = dans_perimetre(df)
    signal = scenario.masque(df)
    lignes = []
    for annee in sorted(df["Year"].unique()):
        a = df["Year"] == annee
        avec = taux(df, perim & signal & a)
        sans = taux(df, perim & ~signal & a)
        lignes.append(
            {
                "Année": int(annee),
                "Dossiers avec le signal": avec.dossiers,
                "Taux avec": avec.valeur,
                "Taux sans": sans.valeur,
                "Rapport": avec.valeur / sans.valeur if sans.valeur else float("nan"),
            }
        )
    return pd.DataFrame(lignes)


def est_stable(df: pd.DataFrame, scenario: Scenario) -> bool:
    return bool((stabilite(df, scenario)["Rapport"] >= SEUIL_STABILITE).all())


def scenarios_retenus(df: pd.DataFrame) -> list[Scenario]:
    return [s for s in CANDIDATS if est_stable(df, s)]


def tableau_candidats(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par signal testé : ce qu'il vaut chaque année, et la décision.

    Sélection sur 1994-1995, puis confirmation sur 1996 que le signal tient encore.
    """
    lignes = []
    for s in CANDIDATS:
        st = stabilite(df, s).set_index("Année")["Rapport"]
        app_ok = all(st.get(a, 0) >= SEUIL_STABILITE for a in ANNEES_APPRENTISSAGE)
        test_ok = st.get(ANNEE_TEST, 0) >= SEUIL_STABILITE
        if app_ok and test_ok:
            decision = "Retenu : tient les 3 années"
        elif app_ok:
            decision = "Écarté : s'effondre en 1996"
        else:
            decision = "Écarté : instable dès 1994-1995"
        lignes.append(
            {
                "Signal": s.nom,
                **{str(a): st.get(a, float("nan")) for a in sorted(df["Year"].unique())},
                "Décision": decision,
            }
        )
    return pd.DataFrame(lignes)


def _matrice(df: pd.DataFrame, colonnes: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df[colonnes].astype(str))


COLONNES_MODELE = [
    "BasePolicy", "Fault", "VehiclePrice", "AgeOfVehicle", "AddressChange_Claim",
    "Days_Policy_Accident", "Days_Policy_Claim", "PastNumberOfClaims", "AccidentArea",
    "NumberOfSuppliments", "PoliceReportFiled", "WitnessPresent", "AgentType",
    "VehicleCategory", "Make", "Deductible", "DriverRating",
]


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Compare l'approche par règles à deux modèles statistiques sur l'année 1996.

    Pour chaque approche : AUC, et part des fraudes trouvées si l'équipe ne revoit
    que la même proportion de dossiers que le périmètre.
    """
    app = df["Year"].isin(ANNEES_APPRENTISSAGE)
    test = df["Year"] == ANNEE_TEST
    y_test = df.loc[test, CIBLE].to_numpy()

    perim = dans_perimetre(df)
    retenus = scenarios_retenus(df[app])
    alerte = np.zeros(len(df), dtype=bool)
    for s in retenus:
        alerte |= (perim & s.masque(df)).to_numpy()
    score_regles = perim.astype(float).to_numpy() + alerte.astype(float)

    X = _matrice(df, COLONNES_MODELE)
    modeles = {
        "Régression logistique (17 variables)": LogisticRegression(C=0.3, max_iter=3000),
        "Gradient boosting (17 variables)": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, random_state=0
        ),
    }
    scores = {"Règles : périmètre + scénarios": score_regles[test.to_numpy()]}
    for nom, m in modeles.items():
        m.fit(X[app], df.loc[app, CIBLE])
        scores[nom] = m.predict_proba(X[test])[:, 1]

    part_revue = perim[test].mean()
    k = int(round(part_revue * test.sum()))
    rng = np.random.default_rng(0)
    lignes = [{"Approche": "Hasard", "AUC": 0.5, "Fraudes trouvées": part_revue, "Explicable": "—"}]
    for nom, s in scores.items():
        # départage aléatoire des ex aequo pour ne pas favoriser les règles
        ordre = np.lexsort((rng.random(len(s)), -s))[:k]
        lignes.append(
            {
                "Approche": nom,
                "AUC": roc_auc_score(y_test, s),
                "Fraudes trouvées": y_test[ordre].sum() / y_test.sum(),
                "Explicable": "Oui, en une phrase" if nom.startswith("Règles") else "Partiellement",
            }
        )
    resultat = pd.DataFrame(lignes)
    resultat.attrs["part_revue"] = part_revue
    resultat.attrs["dossiers_revus"] = k
    resultat.attrs["dossiers_test"] = int(test.sum())
    resultat.attrs["fraudes_test"] = int(y_test.sum())
    return resultat


def entonnoir(df: pd.DataFrame, retenus: list[Scenario]) -> dict[str, Taux]:
    perim = dans_perimetre(df)
    alerte = pd.Series(False, index=df.index)
    for s in retenus:
        alerte |= perim & s.masque(df)
    return {
        "Tous les dossiers": taux(df, pd.Series(True, index=df.index)),
        "Hors périmètre": taux(df, ~perim),
        "Périmètre": taux(df, perim),
        "Alerte scénario": taux(df, alerte),
    }
