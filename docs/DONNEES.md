# Données — provenance, acquisition et profil

## Provenance et conditions d'utilisation

| | |
|---|---|
| Auteur des données | Jeff Sackmann — [`JeffSackmann/tennis_atp`](https://github.com/JeffSackmann/tennis_atp) |
| Source principale | Hugging Face [`Aneeshers/tennis-sackmann-archive`](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive), dossier `atp/` |
| Source de secours | Miroir GitHub du même dataset : [`Aneeshers/tennis-sackmann-archive`](https://github.com/Aneeshers/tennis-sackmann-archive) |
| Licence | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) : attribution, usage **non commercial**, partage dans les mêmes conditions |
| Conséquence pour le dépôt | Aucun fichier de données n'est versionné (`data/` et `*.csv` sont ignorés par Git) ; seuls le script de téléchargement et le fichier de verrouillage le sont |
| Usage dans le projet | Projet pédagogique, non commercial. Le dataset historique est **rejoué par lots** (1 lot = 1 tournoi) par `mock_source` pour simuler l'arrivée de nouvelles données |

## Acquisition reproductible

Mode **pull** : c'est notre code qui va chercher les fichiers, aucune source ne nous pousse de données.

```sh
make data           # télécharge (ou réutilise) les 23 fichiers et vérifie chaque SHA-256
make data-verify    # revérifie les fichiers locaux sans rien télécharger
make data-profile   # vérifie puis écrit data/raw/profile.json
make data-lock      # (mainteneurs) re-résout les révisions et régénère le verrou
```

**Fichiers téléchargés** (≈ 45 Mo) : `atp_matches_2010.csv` → `atp_matches_2026.csv`, `atp_rankings_10s.csv`,
`atp_rankings_20s.csv`, `atp_rankings_current.csv`, `atp_players.csv`, `matches_data_dictionary.txt`,
`UPSTREAM_README.md`. La liste est calculée par `DatasetCatalog` à partir de `configs/project.yaml`.

**Verrouillage** — `configs/data.lock.json` (versionné) joue pour les données le rôle de `uv.lock` pour les
dépendances :

- la **révision résolue** (sha de commit) de chaque source : un même `make data` télécharge toujours le même état,
  même si la source évolue ensuite ;
- le **SHA-256** de chaque fichier : deux machines qui passent `make data-verify` ont des données identiques à l'octet ;
- la **`dataset_version`** (12 caractères), calculée à partir du contenu seul : identique que les fichiers viennent
  de Hugging Face ou de GitHub. Elle sera posée en tag sur les runs MLflow et renvoyée dans l'en-tête
  `X-Dataset-Version`.

**Comportement en cas d'incident**

| Situation | Comportement | Code de sortie |
|---|---|---|
| Source principale injoignable | Bascule sur la source suivante, avertissement « mode dégradé » dans les logs | 0 |
| Toutes les sources injoignables | Arrêt, message listant les sources essayées | 2 |
| Contenu différent du verrou | **Arrêt immédiat, sans fallback** (une corruption n'est pas une panne) ; aucun fichier partiel laissé | 1 |
| Fichier local modifié ou supprimé | `make data-verify` échoue ; `make data` ne retélécharge que ce fichier | 1 puis 0 |
| Fichiers déjà conformes | Réutilisés sans téléchargement (cache) | 0 |

Chaque fichier est d'abord écrit en `.part`, vérifié, puis renommé : il n'existe jamais de fichier à moitié écrit
sous son nom définitif.

## Profil (dataset_version `73331ef02f46`)

46 964 matchs, 66 912 joueurs, 1 468 547 lignes de classement sur 735 semaines (2010-01-04 → 2026-06-08).

| Année | Matchs | Tournois | Dates des tournois | Grand Chelem | Masters | ATP 250/500 | Coupe Davis | Finals / JO |
|---|---|---|---|---|---|---|---|---|
| 2010 | 3 030 | 148 | 2010-01-03 → 2010-12-03 | 508 | 567 | 1615 | 325 | 15 |
| 2011 | 3 015 | 147 | 2011-01-02 → 2011-12-02 | 508 | 567 | 1611 | 314 | 15 |
| 2012 | 3 009 | 147 | 2012-01-01 → 2012-11-16 | 508 | 559 | 1615 | 312 | 15 |
| 2013 | 2 944 | 147 | 2012-12-30 → 2013-11-15 | 508 | 567 | 1541 | 313 | 15 |
| 2014 | 2 901 | 143 | 2013-12-29 → 2014-11-21 | 508 | 567 | 1510 | 301 | 15 |
| 2015 | 2 943 | 147 | 2015-01-04 → 2015-11-27 | 508 | 567 | 1540 | 313 | 15 |
| 2016 | 2 941 | 134 | 2016-01-04 → 2016-11-25 | 508 | 567 | 1600 | 251 | 15 |
| 2017 | 2 911 | 135 | 2017-01-02 → 2017-11-24 | 508 | 567 | 1552 | 253 | 31 |
| 2018 | 2 897 | 136 | 2018-01-01 → 2018-11-23 | 508 | 567 | 1555 | 236 | 31 |
| 2019 | 2 806 | 128 | 2018-12-31 → 2019-11-24 | 508 | 567 | 1529 | 172 | 30 |
| 2020 | 1 462 | 67 | 2020-01-06 → 2020-11-16 | 381 | 164 | 793 | 109 | 15 |
| 2021 | 2 733 | 114 | 2021-01-04 → 2021-12-05 | 508 | 512 | 1569 | 129 | 15 |
| 2022 | 2 917 | 145 | 2022-01-03 → 2022-11-27 | 508 | 520 | 1666 | 208 | 15 |
| 2023 | 2 986 | 144 | 2023-01-02 → 2023-11-27 | 508 | 695 | 1548 | 220 | 15 |
| 2024 | 3 076 | 160 | 2024-01-01 → 2024-12-18 | 508 | 695 | 1519 | 260 | 94 |
| 2025 | 2 944 | 141 | 2024-12-27 → 2025-12-17 | 508 | 775 | 1379 | 252 | 30 |
| 2026 | 1 449 | 68 | 2026-01-04 → 2026-05-25 | 254 | 435 | 632 | 128 | 0 |

**Issue des matchs** (déduite du score par `classify_score`) : 45 268 terminés, 1 380 abandons (`RET`),
299 forfaits (`W/O`, `Walkover`), 12 disqualifications (`DEF`, `Def.`, `Default`), 5 interrompus
(`Played and abandoned`, `ABD`). Les super tie-breaks (`[10-8]`) sont des matchs terminés.

**Valeurs manquantes** : classement du perdant 1,6 %, du vainqueur 0,6 % (joueurs non classés) ; taille 1,3 à 2,7 % ;
durée 10,5 % ; statistiques de service 6,9 % ; surface 0,1 %.

## Constats et décisions

| Constat | Décision |
|---|---|
| 2020 ne compte que 1 462 matchs (circuit suspendu de mars à août) | Sert de démo « période vide » : `mock_source` renvoie `200` + `reason` |
| 2026 s'arrête au 25 mai 2026 | `replay_end: 2026-05-31` |
| Les classements vont jusqu'au 8 juin 2026, après le dernier match | `mock_source` filtre **tout** par l'horloge simulée, classements compris |
| La `tourney_date` peut tomber l'année précédente (fichier 2013 qui commence le 30/12/2012) | On travaille sur les dates, jamais sur l'année du fichier |
| Les 53 surfaces manquantes sont toutes des rencontres de Coupe Davis | Phase 4 : Coupe Davis (`D`, ~4 100 matchs) exclue de l'entraînement (compétition par équipes, matchs parfois sans enjeu) — à confirmer |
| Aucun doublon `(tourney_id, match_num)` | `match_id = {tourney_id}-{match_num}` est une clé stable, utilisée pour le feedback |
| Classement manquant = joueur non classé | Feature « non classé » plutôt que suppression de la ligne |
| Durée, aces, balles de break… ne sont connus qu'**après** le match | Interdits comme features (fuite d'information) ; utilisables seulement en contrôle qualité |
| Formats de score variés | Base de l'expectation custom `ExpectTennisScoreToBeValid` (phase 3) et du `status` de `mock_source` |
