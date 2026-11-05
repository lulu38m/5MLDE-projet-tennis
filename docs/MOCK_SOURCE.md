# `mock_source` — API de rejeu ATP

Le simulateur joue le rôle d'un **fournisseur de données externe** : il rejoue l'historique ATP 2010 → mai 2026
selon une **horloge simulée**, pour que la chaîne MLOps vive « le temps qui passe » (nouveaux résultats, nouveaux
classements, nouveaux joueurs) et puisse être mise en difficulté par des **incidents contrôlés**.

Il lit uniquement les fichiers vérifiés contre `configs/data.lock.json` (refus de démarrer sinon) et expose leur
`dataset_version`. Aucun CSV n'est dans Git ni dans l'image Docker : `data/raw` est monté en lecture seule.

## Lancer

```sh
make up        # Docker : construit l'image et démarre mock_source sur http://127.0.0.1:8001 (attend /health)
make mock      # ou en local, sans Docker
```

Documentation interactive : <http://127.0.0.1:8001/docs>.

**Dépannage** — `container … is unhealthy` juste après `make up` : lire `docker compose logs mock_source`. Le
service refuse de démarrer, avec un message explicite, si les données sont absentes, altérées (SHA-256 ≠ verrou) ou
illisibles par l'utilisateur non-root du conteneur (uid 10001). Dans ce dernier cas, `make data` rétablit les droits
de lecture (fichiers créés en mode 600 par certains outils).

Variables (`.env` ou environnement) : `MOCK_START_DATE` (défaut : `simulation.replay_start` = 2015-01-01),
`MOCK_SPEED` (jours simulés par minute réelle, défaut 0 = horloge arrêtée), `MOCK_PORT`.

## Endpoints

| Méthode | Chemin | Rôle |
|---|---|---|
| GET | `/health` | État, `dataset_version`, date simulée, mode de chaos |
| GET | `/admin/clock` | Date simulée, vitesse, bornes |
| POST | `/admin/clock` | `{"date": "2019-06-01"}` et/ou `{"speed": 3.5}` |
| POST | `/admin/clock/advance` | `{"days": 7}` → ancienne/nouvelle date, matchs et tournois qui viennent de se terminer |
| GET / POST | `/admin/chaos` | Incident contrôlé (voir plus bas) |
| GET | `/results?since=&until=` | Matchs terminés dans `[since ; until[` (`since` obligatoire, `until` ≤ now) |
| GET | `/fixtures?days=7` | Matchs à venir, **sans aucun résultat**, `player1`/`player2` dans un ordre neutre |
| GET | `/tournaments?since=&until=&status=` | Calendrier visible (filtre sur la date de fin, statut `upcoming / in_progress / completed`) |
| GET | `/tournaments/{id}/results` | **Un lot d'ingestion** : résultats déjà joués + `expected_matches` |
| GET | `/rankings?date=&limit=&player_id=` | Dernier classement publié à la date demandée |
| GET | `/players/{id}/history?as_of=` | Fiche, matchs et classements d'un joueur ; `as_of` obligatoire |

Toutes les réponses portent `X-Simulated-Now` et `X-Dataset-Version` (et `X-Chaos-Mode` pendant un incident).

| Situation | Réponse |
|---|---|
| Date (`until`, `date`, `as_of`) postérieure à l'horloge | `400` : le futur n'est jamais accessible |
| Paramètre obligatoire manquant ou mal formé | `422` |
| Période sans match (pause 2020, intersaison) | `200`, liste vide et `reason` explicite |
| Joueur inconnu | `404` ; joueur connu sans match avant `as_of` → `is_debutant: true` |
| Tournoi inconnu ou pas encore annoncé | `404` |

## Règles du rejeu

**Date des matchs** — les fichiers Sackmann ne donnent que la date de début du tournoi. Le jour de chaque match est
estimé en répartissant les tours sur la durée typique du tournoi (Grand Chelem 14 j, Masters à 96+ joueurs 12 j,
ATP 250/500 7 j, Coupe Davis 3 j…), poules (`RR`) avant phase finale. Exemple Grand Chelem : R128 J0, R64 J2,
R32 J4, R16 J6, QF J9, SF J11, F J13.

**Visibilité** (granularité : le jour ; à la date D, tout ce qui est daté avant D est terminé) :

| Donnée | Visible quand |
|---|---|
| Résultat d'un match | `match_date < now` |
| Match à venir | ses **deux joueurs sont connus** : 1er tour 2 jours avant le tournoi, tours suivants le lendemain du match précédent de chaque joueur. Un quart de finale n'est jamais annoncé avant la fin des huitièmes |
| Classement | `ranking_date <= now` (les classements gelés de mars à août 2020 le restent) |
| Tournoi | 14 jours avant son début |

**Ordre neutre des joueurs** — dans `/fixtures`, `player1` est le vainqueur si la parité du SHA-256 du `match_id`
est paire. Sur les vraies données, `player1` gagne dans 49,96 % des cas : l'ordre ne trahit pas le résultat.

**`match_id`** = `{tourney_id}-{match_num}`, unique et stable : c'est la clé qui relie prédiction et feedback.

**`status`** : `completed`, `retired`, `walkover`, `defaulted` ou `abandoned`, déduit du score.

**Déterminisme** — même horloge ⇒ mêmes réponses, octet pour octet (aucun aléatoire, y compris dans le chaos).

## Incidents contrôlés

```sh
make chaos MODE=503 FAILURES=2   # les 2 prochains appels échouent, le 3e passe : un retry réussi
make chaos MODE=slow             # réponses retardées de 5 s → timeouts côté client
make chaos MODE=corrupt          # lot invalide → doit être bloqué par Great Expectations
make chaos MODE=schema           # contrat JSON rompu → doit être bloqué par le contrôle de schéma
make chaos MODE=off              # retour à la normale
docker compose stop mock_source  # source totalement indisponible → fallback / alerte
```

| Mode | Effet (endpoints de données uniquement, jamais `/admin/*` ni `/health`) |
|---|---|
| `503` | `503 Service Unavailable` + `Retry-After: 1` |
| `slow` | Réponse retardée de `delay_s` secondes (5 par défaut) |
| `corrupt` | ~60 % des matchs altérés, toujours les mêmes : vainqueur = perdant, score illisible (`6-4 7-`), durée négative, date dans le futur, surface `Moon`, lignes supprimées (volume ≠ `expected_matches`) |
| `schema` | `match_id` → `matchId`, `score` supprimé, `minutes` en texte (`"90 min"`), `winner.id` → `winner.player_id` |

`failures` (facultatif) limite le nombre de requêtes affectées, après quoi le mode repasse à `off` tout seul.

## Garanties testées

`tests/unit/mock_source/test_no_future_leak.py` règle l'horloge jour par jour sur un mini-dataset, puis à 30 dates
de 2015 à 2026 sur les vraies données, et vérifie à chaque fois qu'aucune réponse ne contient de résultat futur,
de joueur déjà éliminé annoncé au tour suivant, ni de classement postérieur à l'horloge.

Contrôle complémentaire sur les 46 964 matchs réels : `known_date <= match_date` partout, aucun joueur avec deux
matchs à élimination directe le même jour, aucun match révélé avant la fin du précédent.
