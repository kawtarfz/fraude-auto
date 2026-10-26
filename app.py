"""Interface du gestionnaire sinistres : triage de la fraude automobile par scénarios.

Lancer en local :  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from fraude import ALERTE, PAS_D_ALERTE, REVUE_STANDARD, Trieur, charger
from fraude.donnees import CIBLE, libelle
from fraude.evaluation import (
    ANNEE_TEST,
    ANNEES_APPRENTISSAGE,
    SEUIL_STABILITE,
    backtest,
    entonnoir,
    scenarios_retenus,
    tableau_candidats,
)
from fraude.scenarios import EXCLUES

st.set_page_config(page_title="Triage fraude auto", page_icon="🔎", layout="wide")

st.markdown(
    """
    <style>
      .badge {display:inline-block;padding:.25rem .7rem;border-radius:999px;font-weight:600;font-size:.9rem}
      .b-alerte {background:#fde2e1;color:#9b1c1c}
      .b-revue {background:#fef3c7;color:#854d0e}
      .b-rien {background:#e0f2e9;color:#166534}
      .note {color:#6b7280;font-size:.9rem}
      .chiffre {font-size:1.6rem;font-weight:700;line-height:1.1}
    </style>
    """,
    unsafe_allow_html=True,
)

BADGES = {ALERTE: "b-alerte", REVUE_STANDARD: "b-revue", PAS_D_ALERTE: "b-rien"}
ORDRE = {ALERTE: 0, REVUE_STANDARD: 1, PAS_D_ALERTE: 2}


@st.cache_data
def preparer():
    df = charger()
    retenus_codes = [s.code for s in scenarios_retenus(df)]
    return df, retenus_codes


@st.cache_data
def resultats_backtest(_df: pd.DataFrame):
    b = backtest(_df)
    return b, dict(b.attrs)


df, codes_retenus = preparer()
from fraude.scenarios import par_code  # noqa: E402

retenus = [par_code(c) for c in codes_retenus]
historique = df[df["Year"].isin(ANNEES_APPRENTISSAGE)]
courants = df[df["Year"] == ANNEE_TEST].copy()
trieur = Trieur(historique, retenus)
courants["Niveau"] = trieur.niveaux(courants)


def nombre(n: int) -> str:
    return f"{n:,}".replace(",", "\u202f")


def pct(x: float) -> str:
    return f"{x * 100:.1f} %".replace(".", ",")


def fr(valeur, unite: str = "") -> str:
    """Traduit les tranches du jeu de données (« 8 to 15 », « more than 30 »...)."""
    v = str(valeur)
    speciaux = {"none": "aucun", "new": "neuf", "no change": "aucun", "under 6 months": "moins de 6 mois"}
    if v in speciaux:
        return speciaux[v]
    v = (v.replace("more than", "plus de").replace("less than", "moins de")
          .replace(" to ", " à ").replace("years", "ans").replace("year", "an"))
    return v + (unite if v[-1].isdigit() else "")


def badge(niveau: str) -> str:
    return f'<span class="badge {BADGES[niveau]}">{niveau}</span>'


def faits(dossier: pd.Series) -> pd.DataFrame:
    lignes = [
        ("Garantie", libelle("BasePolicy", dossier["BasePolicy"])),
        ("Responsable de l'accident", libelle("Fault", dossier["Fault"])),
        ("Délai souscription → accident", fr(dossier["Days_Policy_Accident"], " jours")),
        ("Délai souscription → déclaration", fr(dossier["Days_Policy_Claim"], " jours")),
        ("Véhicule", dossier["Make"]),
        ("Prix du véhicule", fr(dossier["VehiclePrice"])),
        ("Âge du véhicule", fr(dossier["AgeOfVehicle"])),
        ("Zone de l'accident", libelle("AccidentArea", dossier["AccidentArea"])),
        ("Procès-verbal de police", libelle("PoliceReportFiled", dossier["PoliceReportFiled"])),
        ("Témoin présent", libelle("WitnessPresent", dossier["WitnessPresent"])),
        ("Sinistres passés", fr(dossier["PastNumberOfClaims"])),
        ("Dernier changement d'adresse", fr(dossier["AddressChange_Claim"])),
    ]
    return pd.DataFrame(lignes, columns=["Élément", "Valeur"])


def fiche(dossier: pd.Series, montrer_verite: bool = True, seulement: list[str] | None = None) -> None:
    v = trieur.evaluer(dossier)
    st.markdown(badge(v.niveau), unsafe_allow_html=True)
    st.markdown(f"#### {v.raison}")

    gauche, droite = st.columns([3, 2], gap="large")
    with gauche:
        if v.niveau == ALERTE:
            s = v.scenarios[0]
            st.markdown(f"**Le mécanisme suspecté.** {s.mecanisme}")
            st.markdown("**Ce que vous vérifiez avant de décider :**")
            for etape in s.verifications:
                st.markdown(f"- {etape}")
        elif v.niveau == REVUE_STANDARD:
            st.markdown(
                "Ce dossier entre dans la zone où se concentre la fraude, mais aucun scénario "
                "précis ne se déclenche. **Traitement habituel, avec expertise du véhicule.** "
                "L'outil ne recommande pas d'enquête particulière."
            )
        else:
            st.markdown(
                "**Aucune action spécifique.** Sur ce type de dossier, la fraude est rare ; "
                "un signal ici créerait surtout de fausses alertes."
            )

        if v.reference is not None:
            bas, haut = v.reference.intervalle
            st.markdown(
                f'<p class="note">Sur les dossiers comparables de {ANNEES_APPRENTISSAGE[0]}-'
                f'{ANNEES_APPRENTISSAGE[1]} : <b>{v.reference.fraudes} fraudes sur '
                f'{nombre(v.reference.dossiers)} dossiers</b> ({pct(v.reference.valeur)}, '
                f'entre {pct(bas)} et {pct(haut)} avec 95 % de confiance).</p>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<p class="note"><b>Ce que l\'outil ne voit pas :</b> montants, dates exactes, photos, '
            "rapport d'expertise, et les liens entre dossiers (même garage, même témoin). "
            "Une alerte est une raison de vérifier, jamais une preuve.</p>",
            unsafe_allow_html=True,
        )
    with droite:
        st.markdown("**Les faits du dossier**")
        tableau_faits = faits(dossier)
        if seulement:
            tableau_faits = tableau_faits[tableau_faits["Élément"].isin(seulement)]
        st.dataframe(tableau_faits, hide_index=True, width="stretch",
                     height=35 * (len(tableau_faits) + 1) + 3)
        if montrer_verite and CIBLE in dossier:
            with st.expander("Résultat réel de l'enquête (démonstration uniquement)"):
                st.write("Fraude avérée" if dossier[CIBLE] == 1 else "Pas de fraude constatée")


st.title("Triage de la fraude automobile")
st.markdown(
    "Une aide à la décision pour le gestionnaire sinistres : **dire où regarder, pourquoi, et "
    "quoi vérifier**, et se taire quand les données ne permettent pas de conclure."
)

onglets = st.tabs(["File du gestionnaire", "Tester un dossier", "Le découpage", "Validation 1996"])

# ---------------------------------------------------------------- File du gestionnaire
with onglets[0]:
    comptes = courants["Niveau"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Dossiers {ANNEE_TEST}", nombre(len(courants)))
    c2.metric("Alertes", nombre(int(comptes.get(ALERTE, 0))))
    c3.metric("Revue standard", nombre(int(comptes.get(REVUE_STANDARD, 0))))
    c4.metric("Pas d'alerte", nombre(int(comptes.get(PAS_D_ALERTE, 0))))

    choix = st.segmented_control(
        "Afficher", [ALERTE, REVUE_STANDARD, PAS_D_ALERTE], default=ALERTE, key="filtre"
    )
    vue = courants if choix is None else courants[courants["Niveau"] == choix]
    vue = vue.assign(ordre=vue["Niveau"].map(ORDRE)).sort_values(["ordre", "Dossier"])
    tableau = pd.DataFrame(
        {
            "Dossier": vue["Dossier"],
            "Niveau": vue["Niveau"],
            "Garantie": vue["BasePolicy"].map(lambda x: libelle("BasePolicy", x)),
            "Responsable": vue["Fault"].map(lambda x: libelle("Fault", x)),
            "Souscription → accident": vue["Days_Policy_Accident"].map(lambda x: fr(x, " jours")),
            "Véhicule": vue["Make"],
            "Prix du véhicule": vue["VehiclePrice"].map(fr),
        }
    )
    sel = st.dataframe(
        tableau,
        hide_index=True,
        width="stretch",
        height=320,
        on_select="rerun",
        selection_mode="single-row",
        key="table_dossiers",
    )
    lignes = sel.selection.rows if sel and sel.selection else []
    st.divider()
    if lignes:
        dossier = vue.iloc[lignes[0]]
        st.subheader(f"Fiche {dossier['Dossier']}")
        fiche(dossier)
    else:
        st.info("Cliquez sur une ligne du tableau pour ouvrir la fiche du dossier.")

# ---------------------------------------------------------------- Tester un dossier
with onglets[1]:
    st.markdown("Décrivez un sinistre : l'outil applique le même découpage.")
    with st.form("nouveau"):
        a, b, c = st.columns(3)
        garantie = a.selectbox("Garantie", ["Collision", "All Perils", "Liability"],
                               format_func=lambda x: libelle("BasePolicy", x))
        faute = a.selectbox("Responsable", ["Policy Holder", "Third Party"],
                            format_func=lambda x: libelle("Fault", x))
        delai_acc = b.selectbox("Délai souscription → accident",
                                ["more than 30", "15 to 30", "8 to 15", "1 to 7", "none"], format_func=fr)
        delai_decl = b.selectbox("Délai souscription → déclaration",
                                 ["more than 30", "15 to 30", "8 to 15", "none"], format_func=fr)
        prix = c.selectbox("Prix du véhicule", sorted(df["VehiclePrice"].unique()), format_func=fr)
        zone = c.selectbox("Zone", ["Urban", "Rural"], format_func=lambda x: libelle("AccidentArea", x))
        envoye = st.form_submit_button("Analyser", type="primary")
    if envoye:
        modele = df.iloc[0].copy()
        modele.update({
            "Dossier": "Nouveau", "BasePolicy": garantie, "Fault": faute,
            "Days_Policy_Accident": delai_acc, "Days_Policy_Claim": delai_decl,
            "VehiclePrice": prix, "AccidentArea": zone,
        })
        fiche(
            modele.drop(labels=[CIBLE]),
            montrer_verite=False,
            seulement=["Garantie", "Responsable de l'accident", "Délai souscription → accident",
                       "Délai souscription → déclaration", "Prix du véhicule", "Zone de l'accident"],
        )

# ---------------------------------------------------------------- Le découpage
with onglets[2]:
    st.subheader("1. Le périmètre : où la fraude est possible")
    e = entonnoir(courants, retenus)
    cols = st.columns(4)
    textes = {
        "Tous les dossiers": "toutes déclarations",
        "Hors périmètre": "RC seule, ou tiers responsable",
        "Périmètre": "garantie dommages + assuré responsable",
        "Alerte scénario": "sinistre peu après la souscription",
    }
    for col, (nom, t) in zip(cols, e.items()):
        col.markdown(
            f"**{nom}**<br><span class='note'>{textes[nom]}</span><br>"
            f"<span class='chiffre'>{pct(t.valeur)}</span><br>"
            f"<span class='note'>{t.fraudes} fraudes / {nombre(t.dossiers)} dossiers</span>",
            unsafe_allow_html=True,
        )
    hors, perim = e["Hors périmètre"], e["Périmètre"]
    st.markdown(
        f"Quand l'assureur n'indemnise pas le véhicule de l'assuré (RC seule) ou que le tiers "
        f"est responsable, il n'y a presque rien à gagner à frauder : **{pct(hors.valeur)}** de "
        f"fraude. En sortant ces dossiers, l'équipe concentre son attention sur "
        f"**{pct(perim.dossiers / (perim.dossiers + hors.dossiers))}** des dossiers, qui contiennent "
        f"**{pct(perim.fraudes / (perim.fraudes + hors.fraudes))}** des fraudes. Cette règle tient en "
        "une phrase et se vérifie chaque année."
    )

    st.subheader("2. Les scénarios : un signal doit tenir dans le temps")
    st.markdown(
        f"Huit signaux plausibles ont été testés à l'intérieur du périmètre. Chaque chiffre indique "
        f"par combien le signal multiplie le taux de fraude, année par année. Un signal est retenu "
        f"s'il dépasse **×{str(SEUIL_STABILITE).replace(".", ",")}** sur 1994-1995, **puis** encore en 1996."
    )
    cand = tableau_candidats(df)
    annees = [c for c in cand.columns if c.isdigit()]
    st.dataframe(
        cand.style.format({a: (lambda x: f"×{x:.2f}".replace(".", ",")) for a in annees}).map(
            lambda v: "color:#9b1c1c" if isinstance(v, float) and v < SEUIL_STABILITE else "", subset=annees
        ),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        "Trois signaux semblaient solides sur 1994-1995 (tous risques, véhicule haut de gamme, zone "
        "rurale) et s'effondrent l'année suivante. Les transformer en alertes aurait produit des "
        "centaines de fausses pistes. **Un seul scénario résiste : le sinistre déclaré peu après la "
        "souscription.** Il touche peu de dossiers, mais près d'un sur trois s'est révélé frauduleux en 1996."
    )

    st.subheader("3. Ce que l'outil s'interdit")
    st.markdown(
        "Certaines variables sont corrélées à la fraude dans ces données, mais décrivent la personne "
        f"et non le sinistre : **{', '.join(EXCLUES.values())}**. Elles ne déclenchent jamais d'alerte. "
        "Un gestionnaire doit pouvoir justifier une vérification par un fait du dossier."
    )

# ---------------------------------------------------------------- Validation
with onglets[3]:
    b, infos = resultats_backtest(df)
    st.markdown(
        f"Toutes les approches apprennent sur **1994-1995** et sont jugées sur **{ANNEE_TEST}** "
        f"({nombre(infos['dossiers_test'])} dossiers, {infos['fraudes_test']} fraudes). Pour comparer à "
        f"effort égal, chaque approche revoit le même nombre de dossiers que le périmètre : "
        f"**{nombre(infos['dossiers_revus'])}**, soit {pct(infos['part_revue'])}."
    )
    st.dataframe(
        b.style.format({"AUC": lambda x: f"{x:.3f}".replace(".", ","), "Fraudes trouvées": lambda x: pct(x)}),
        hide_index=True,
        width="stretch",
    )
    regles = b.iloc[1]
    meilleur = b.iloc[1:]["Fraudes trouvées"].max()
    st.markdown(
        f"Les règles retrouvent **{pct(regles['Fraudes trouvées'])}** des fraudes ; le meilleur modèle "
        f"en retrouve **{pct(meilleur)}**. L'écart est faible, alors que la règle s'explique en une phrase "
        "à un gestionnaire et à un assuré. À l'intérieur du périmètre, aucun modèle ne classe "
        "nettement mieux que le hasard sur 1996 : les données disponibles ne contiennent pas le "
        "signal fin, et le bon réflexe est de le dire plutôt que de le masquer."
    )
    st.markdown("**Pour aller plus loin avec de vraies données d'assureur :**")
    st.markdown(
        "- les montants et dates exactes, pour mesurer l'écart entre devis et valeur du véhicule ;\n"
        "- les liens entre dossiers (garage, expert, témoin, adresse), où se cachent les fraudes organisées ;\n"
        "- le retour des enquêteurs sur chaque alerte, pour savoir quels scénarios méritent leur temps."
    )

st.markdown(
    '<p class="note" style="margin-top:2rem">Données publiques « Vehicle Insurance Claim Fraud Detection » '
    "(Oracle, 15 420 déclarations, 1994-1996). Projet de démonstration de Kawtar Faouzi.</p>",
    unsafe_allow_html=True,
)
