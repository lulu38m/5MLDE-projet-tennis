# Ingestion et contrôles qualité

Le flow Prefect `ingestion-atp` tire (mode **pull**) les tournois terminés depuis `mock_source`, contrôle chaque lot
avec Great Expectations et ne stocke dans PostgreSQL que ce qui passe. Un lot refusé part en quarantaine, le flow
échoue (une carte Teams, voir [`ALERTES.md`](ALERTES.md)) et le lot ne compte pas pour le déclenchement de
l'entraînement ([`ORCHESTRATION.md`](ORCHESTRATION.md)).

## Lancer

```sh
make up          # mock_source + PostgreSQL (127.0.0.1:5433) + Prefect (http://127.0.0.1:4201)
make ingest      # un passage d'ingestion, visible dans l'UI Prefect
make lots        # les derniers lots : statut, volumes, contrôles en échec
make gx-docs     # rapports HTML Great Expectations (Data Docs)
make db          # console SQL (base tennis)
```

Ports d'hôte décalés (5433, 4201) pour coexister avec la stack des TP (5432, 4200).

## Déroulé d'un passage

| Étape | Task Prefect | Détail |
|---|---|---|
| 1 | `GET /admin/clock` | Date simulée de la source |
| 2 | `GET /results (amorçage)` | Une seule fois : l'historique 2010 → début du rejeu (14 899 matchs), contrôlé comme un lot |
| 3 | `GET /tournaments since=…` | Tournois terminés depuis le watermark, du plus ancien au plus récent |
| 4 | `GET /tournaments/{id}/results` | Un lot = un tournoi ; un lot déjà stocké est sauté |
| 5 | `contrôles {lot}` | Suite bloquante + surveillance de dérive |
| 6 | `stockage {lot}` ou `quarantaine {lot}` | Transaction PostgreSQL + archive brute, ou archive en quarantaine |
| 7 | — | Watermark avancé jusqu'au premier lot refusé (retenté au passage suivant) |

**Une task par appel HTTP**, chacune avec 3 retries et un backoff de 1, 2 puis 4 s (`configs/project.yaml`,
section `ingestion`). Seule une panne est retentée (`503`, timeout, connexion) ; un `4xx` échoue tout de suite.

## Issues possibles

| Issue | État Prefect | Code de sortie (`python -m tennis_mlops.flows.ingest`) | Effet |
|---|---|---|---|
| Lots valides | `Completed` | 0 | Matchs stockés, watermark = date simulée |
| Rien de nouveau | `Completed` | 0 | `up_to_date` |
| Lot refusé | `Failed` | 1 | Quarantaine, rien stocké pour ce lot ; les autres lots valides sont stockés ; watermark arrêté avant le lot refusé |
| Source indisponible après 4 essais | `Dégradé` (état personnalisé) | 2 | Rien ne bouge : les données déjà ingérées restent utilisables, reprise au watermark |

## Contrôles Great Expectations

Contexte GX sur disque (`state/gx/`), checkpoints et Data Docs HTML, comme dans les TP. Toute erreur d'exécution d'un
contrôle (colonne absente…) compte comme un échec.

**Suite bloquante `atp_lot_blocking`** (36 expectations sur un lot de tournoi)

| Famille | Contrôles |
|---|---|
| Schéma | 14 colonnes obligatoires ; `winner_id`, `loser_id` entiers ; `minutes` numérique |
| Identité | `match_id` non nul et unique ; `match_date`, `winner_id`, `loser_id` non nuls |
| Domaines | niveau ∈ {G, M, A, D, F, O} ; surface ∈ {Hard, Clay, Grass, Carpet} ; tour ; `best_of` ∈ {3, 5} ; statut ; date ISO ; `minutes` ∈ [0 ; 1 200] ; classements ∈ [1 ; 5 000] |
| Custom (métriques propres) | `ExpectTennisScoreToBeValid` · `ExpectWinnerAndLoserToDiffer` · `ExpectNoMatchAfterSimulatedNow` |
| Custom (spécialisée) | `ExpectBatchVolumeToMatchDraw` : nb de matchs = `expected_matches` |

**Suite de surveillance `atp_lot_monitoring`** (non bloquante) : `ExpectRankGapDistributionStable`, la médiane de
l'écart de classement du tournoi doit rester dans la plage [0,5 % ; 99,5 %] observée sur les tournois **du même
niveau déjà validés**. La plage varie énormément selon le niveau (Masters de fin d'année ≈ 3 places, ATP 250 ≈ 43,
Coupe Davis jusqu'à 1 500) : une référence unique produirait des fausses alertes. Il faut au moins 20 tournois de
référence. Un écart est tracé (`warnings`) sans bloquer : un tournoi atypique n'est pas une donnée fausse.

**Ce que chaque incident déclenche** (vérifié sur Brisbane 2015 avec `make chaos MODE=corrupt`)

| Défaut injecté | Contrôle qui le bloque |
|---|---|
| Vainqueur = perdant | `expect_winner_and_loser_to_differ` |
| Score `6-4 7-` | `expect_tennis_score_to_be_valid` |
| Durée `-12` | `expect_column_values_to_be_between(minutes)` |
| Match daté 30 jours dans le futur | `expect_no_match_after_simulated_now` |
| Surface `Moon` | `expect_column_values_to_be_in_set(surface)` |
| Lignes supprimées (20 sur 27) | `expect_batch_volume_to_match_draw` |
| `MODE=schema` : `matchId`, `score` absent, `minutes` en texte | `expect_column_to_exist`, `expect_column_values_to_be_in_type_list` |

## Stockage

| Où | Quoi |
|---|---|
| PostgreSQL `matches` | Un match validé par ligne (clé `match_id`), avec `lot_id` et `dataset_version` |
| PostgreSQL `ingestion_lots` | Un lot par ligne : `valid` / `quarantined`, reçus / attendus / ajoutés, contrôles en échec, écarts de dérive, chemin de l'archive et du rapport GX, run Prefect |
| PostgreSQL `ingestion_state` | Le watermark |
| `state/lots/<lot>/` | Réponse brute + verdict (`payload.json`, `validation.json`) |
| `state/quarantine/<lot>/` | Idem pour les lots refusés ; conservés comme historique des incidents même après une reprise |
| `state/gx/` | Contexte Great Expectations et Data Docs |

Un lot validé est écrit en une transaction (matchs + trace). Réécrire un lot déjà stocké n'ajoute rien
(idempotence par `match_id`) : une reprise est toujours sans risque. Au démarrage, une colonne texte plus courte en
base que dans le schéma est élargie (seule migration automatique, sans perte) : `tourney_id` passe de 32 à 64
caractères pour la Coupe Davis (`2016-M-DC-2016-G1-AM-M-BAR-ECU-01`).

Les classements hebdomadaires ne sont pas ingérés à part : chaque match porte déjà le classement et les points des
deux joueurs au début du tournoi, ce dont l'entraînement a besoin.

## Démo (vérifiée de bout en bout)

```sh
make ingest                                   # amorçage : 17 842 matchs (2010 → 2015)
make advance DAYS=14 && make ingest           # nouveaux tournois stockés
make chaos MODE=corrupt && make advance DAYS=7 && make ingest   # Failed : lots en quarantaine
make lots                                     # statut quarantined + contrôles en échec
make chaos MODE=off && make ingest            # reprise : les mêmes lots passent
make chaos MODE=503 FAILURES=2 && make advance DAYS=7 && make ingest   # absorbé par les retries
make chaos MODE=503 && make advance DAYS=7 && make ingest       # état « Dégradé », rien perdu
make chaos MODE=slow DELAY=15 && make ingest  # timeouts (10 s) puis retries
make chaos MODE=off
```

`make` affiche `Error 1` pour une quarantaine et `Error 2` pour l'état « Dégradé » : ce sont les codes de sortie
attendus (tableau ci-dessus), visibles aussi dans Prefect.
