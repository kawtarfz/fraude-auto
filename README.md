# Triage de la fraude automobile

Une aide à la décision pour le gestionnaire sinistres : **dire où regarder, pourquoi, et quoi vérifier**, et se taire quand les données ne permettent pas de conclure.

**Application en ligne :** https://fraude-auto-ltsmjtkggggqxfwtoyae78.streamlit.app/

## Pourquoi ce projet

Pendant mon stage de recherche chez Framatome, je devais prédire la conformité de pièces produites par fabrication additive. J'ai comparé Random Forest, XGBoost, CatBoost et d'autres modèles, et c'est finalement un GLM, plus simple, qui a donné les meilleurs résultats en backtesting. Surtout, avec SHAP, nous avons pu le traduire en une plage de paramètres que les ingénieurs procédé utilisaient directement.

J'en ai retenu deux choses que j'ai voulu appliquer ici, sur un sujet d'assureur :

- **un modèle n'a de valeur que si le métier peut s'en servir** : le gestionnaire sinistres doit savoir pourquoi un dossier est signalé et quoi vérifier ;
- **un résultat doit être confirmé sur des données qu'on n'a pas vues** : ici, apprendre sur 1994-1995 et juger sur 1996.

Ce second principe a été décisif : trois signaux qui semblaient solides sur les deux premières années ne tiennent plus en 1996.

## Le parti pris

La fraude n'est pas un score. Plutôt qu'un modèle unique, le problème est découpé :

1. **Un périmètre.** Quand le contrat est en RC seule ou que le tiers est responsable, l'assureur n'indemnise pas le véhicule de l'assuré : il n'y a presque rien à gagner à frauder (0,7 % de fraude en 1996). Hors de ce périmètre, l'outil ne dit rien. Cette seule règle concentre l'attention sur 44 % des dossiers, qui contiennent 92 % des fraudes.
2. **Des scénarios qui tiennent dans le temps.** Huit signaux plausibles ont été testés. Un signal ne devient une alerte que s'il multiplie le taux de fraude par au moins 1,2 sur 1994-1995, **puis** encore en 1996. Trois signaux qui semblaient solides (tous risques, véhicule haut de gamme, zone rurale) s'effondrent l'année suivante. Un seul résiste : **le sinistre déclaré peu après la souscription** (29 % de fraude en 1996).
3. **Une décision, pas un chiffre.** Chaque alerte dit au gestionnaire le mécanisme suspecté, ce qu'il vérifie, le taux observé sur des dossiers comparables avec son intervalle de confiance, et ce que l'outil ne voit pas.
4. **Des variables interdites.** L'âge, le sexe et la situation familiale sont corrélés à la fraude dans ces données, mais décrivent la personne et non le sinistre. Ils ne déclenchent jamais d'alerte.

## Validation sur une année non vue

Apprentissage sur 1994-1995, test sur 1996 (4 083 dossiers, 213 fraudes), à effort de revue égal (1 812 dossiers) :

| Approche | AUC | Fraudes trouvées |
|---|---|---|
| Hasard | 0,500 | 44,4 % |
| **Règles : périmètre + scénarios** | **0,749** | **92,0 %** |
| Régression logistique (17 variables) | 0,763 | 93,4 % |
| Gradient boosting (17 variables) | 0,789 | 94,4 % |

Le meilleur modèle ne trouve que 2 points de plus, pour une décision qu'on ne peut pas expliquer en une phrase. À l'intérieur du périmètre, aucun modèle ne fait nettement mieux que le hasard sur 1996 : le signal fin n'est pas dans ces données.

## Ce que je ferais avec de vraies données d'assureur

- Les montants et dates exactes, pour comparer devis et valeur du véhicule.
- Les liens entre dossiers (garage, expert, témoin, adresse) : c'est là que se trouvent les fraudes organisées, et un modèle ligne à ligne ne les voit pas.
- Le retour des enquêteurs sur chaque alerte, pour mesurer quels scénarios méritent leur temps et en ajouter de nouveaux.

## Organisation du code

```
fraude/
  donnees.py     chargement, nettoyage (retrait d'une colonne qui fuit la cible)
  scenarios.py   périmètre et scénarios candidats : données, signal, vérifications
  evaluation.py  stabilité année par année, backtest 1996, intervalles de Wilson
  triage.py      verdict d'un dossier : niveau, raison, taux de référence
app.py           interface Streamlit
tests/           tests du découpage, du triage et de l'interface
```

## Comment ce projet a été construit

J'ai utilisé un assistant IA pour accélérer l'écriture du code et de l'interface. Le découpage du problème, le critère de stabilité, le choix de valider sur 1996 et l'exclusion des variables personnelles sont des choix que j'ai faits et que je peux défendre ; les tests automatiques vérifient que le code fait bien ce qui est annoncé.

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
pytest
```

## Données

« Vehicle Insurance Claim Fraud Detection », jeu public publié par Oracle et diffusé sur Kaggle : 15 420 déclarations de sinistres automobiles (1994-1996), avec l'issue de l'enquête.
