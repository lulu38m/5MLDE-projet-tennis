# Déroulé de la démo

La démonstration de la soutenance (20 minutes), commande par commande, avec ce qu'on doit voir à l'écran. Les
commandes `make` sont celles de macOS et Linux ; sous Windows, voir la dernière section.

## Préparer (la veille, puis 30 minutes avant)

La démo part d'un **état connu**, toujours le même : v1 en production, puis v2 promue par la chaîne après cinq mois
de 2016. Tout est déterministe : chaque préparation donne les mêmes versions et les mêmes chiffres.

```sh
uv run tennis reset          # repartir de zéro (garde les données téléchargées)
uv run tennis up             # stack + v1 en production (5 à 10 min)
make simulate DAYS=154       # 2016-01-01 → 2016-06-03 : 64 tournois ingérés, réentraînement par événement
make registry                # attendre ici que v2 soit en production (1 à 3 min), previous = v1
make simulate DAYS=7         # une semaine de plus : les matchs prédits au pas précédent ont leur feedback
uv run tennis urls           # tout doit répondre (●)
```

Vérifier aussi, la veille :

- `make alert-test` : une carte « 🧪 TEST » arrive dans le canal Teams (le webhook de `.env` marche) ;
- `make grafana-check` : chaque panneau répond ;
- `make check` : vert ;
- la vidéo de secours est sur le poste (voir plus bas).

Ouvrir à l'avance, un onglet chacun : l'interface (`make ui-open`), Prefect (http://127.0.0.1:4201), MLflow
(`make mlflow`), Grafana (`make grafana`), le canal Teams, et ce dépôt sur GitHub (README, diagramme).

Juste avant de commencer : `make demo`, pour que des pas de simulation arrivent pendant qu'on parle (une semaine
simulée par minute). Avec la simulation en marche, d'autres campagnes peuvent partir pendant la présentation, et
une nouvelle version peut être promue ou refusée : c'est la chaîne qui vit, `make registry` dira où elle en est.

## Le déroulé

| Temps | Partie | À l'écran |
|---|---|---|
| 0 – 2 min | Sujet | README |
| 2 – 4 | Architecture | diagramme du README |
| 4 – 7 | Données | `mock_source`, Great Expectations |
| 7 – 10 | Prefect et MLflow | runs, tasks parallèles, campagne |
| 10 – 12 | Registre et mise en service | `make registry`, `make trace` |
| 12 – 15 | Service | interface, API, dashboards |
| 15 – 17 | **Incident en direct** | lot corrompu bloqué, alerte, rollback |
| 17 – 19 | Tests de charge | `SYNTHESE_OPS.md` |
| 19 – 20 | Limites | `SYNTHESE_OPS.md` |

### 0 – 2 min · Le sujet

Prédire la probabilité de victoire d'un joueur dans un match ATP. **Ce qui est évalué, c'est la chaîne, pas le
score** : des données qui arrivent dans le temps, contrôlées, un modèle réentraîné et mis en service par une règle,
servi, surveillé, et qui se juge sur les vrais résultats.

### 2 – 4 min · L'architecture

Le diagramme du README : `mock_source` → ingestion (Prefect + Great Expectations) → PostgreSQL → entraînement
(MLflow) → registre → API → interface ; Prometheus et Grafana autour. Insister sur deux choses : **tout se lance en
deux commandes**, et **la boucle est fermée** (les vrais résultats jugent les prédictions).

### 4 – 7 min · Les données

```sh
make clock                   # {"now": "2016-06-…"} : l'horloge simulée de la source
```

- http://127.0.0.1:8001/docs : la source rejoue l'historique ; `GET /fixtures` donne les matchs à venir, **sans
  résultat**. Aucune réponse ne révèle le futur (testé jour par jour sur les vraies données).
- `make lots` : un lot = un tournoi, statut `valid`, volumes reçus et attendus.
- `make gx-docs` : le rapport Great Expectations d'un lot ; montrer les **expectations custom** (score de tennis
  valide, vainqueur ≠ perdant, aucun match après l'horloge, volume conforme au tableau).
- `configs/data.lock.json` : la version des données figée par SHA-256, comme `uv.lock` pour les dépendances.

### 7 – 10 min · Prefect et MLflow

- **Prefect** : les runs `simulation-atp` s'appellent par leur semaine simulée (`2016-06-10 → 2016-06-17`) ; en
  ouvrir un : ingestion, feedback, scoring en sous-flows. Ouvrir un run `entrainement-atp` : lancé par
  l'**automation** (événement `tennis.training.requested`, ≥ 200 nouveaux matchs), les 8 candidats en parallèle.
- `make demo-status` : le planning, le worker, les derniers runs.
- **MLflow** : l'expérience `atp-winner`, une campagne dépliée et ses 8 candidats ; dans les artefacts du candidat
  retenu, `rapport.html` (chiffres contre l'Elo pur, log loss mois par mois, segments, plus grosses surprises).

### 10 – 12 min · Le registre et la règle de mise en service

```sh
make registry                # les versions, leurs alias et la décision de la règle pour chacune
make trace                   # de la version jusqu'aux SHA-256 des fichiers sources
```

La règle : données validées, **Elo pur et classement battus**, **au moins aussi bon que la production sur le même
test**, aucune surface nettement dégradée, dérive bornée. Deux anecdotes vraies : le premier modèle sur 2010 → 2014
était refusé (accuracy sous « le mieux classé gagne ») ; une version entraînée sur cinq semaines de plus aussi. Les
alias : `production` (ce que l'API sert), `previous` (la cible du rollback), `staging` (le dernier refusé).

### 12 – 15 min · Le service

- **Interface** (http://127.0.0.1:8501), page Match : Djokovic contre Nadal sur terre → probabilité de chacun, la
  version qui l'a calculée, ce que le modèle a vu. Page Matchs à venir : prédits à l'avance par le scoring, puis le
  vrai résultat et « le modèle avait-il raison ? ».
- `make predict` : la même chose par l'API ; `make predictions` : la trace de chaque prédiction en base.
- **Grafana** : Technique (débit, latence, erreurs, conteneurs), **Métier** (accuracy et log loss en ligne sur l'axe
  du temps simulé, calibration), **Croisé** (latence et qualité par version). Les annotations : promotions, lots
  refusés, incidents.

### 15 – 17 min · L'incident en direct

```sh
make demo-stop                          # la simulation en pause : on pilote l'incident à la main
make chaos MODE=corrupt                 # la source corrompt ses réponses
make advance DAYS=7 && make ingest      # → « Error 1 » : l'ingestion échoue
make lots                               # les lots de la semaine en quarantined, et les contrôles en échec
```

À montrer, dans l'ordre :

1. **Prefect** : le run `ingestion-atp` en `Failed`, les tasks « quarantaine » ;
2. **Teams** : la carte « 🚫 Lot refusé par Great Expectations » (tournoi, contrôles en échec, lien vers le run) ;
3. **Grafana** : l'annotation rouge sur les dashboards ;
4. aucun entraînement n'est parti, la production n'a pas bougé ; et si on en force un (`make train`), il est
   **interrompu** : « entraînement interrompu : N lot(s) en quarantaine non résolu(s) », carte « échec » dans Teams.

```sh
make chaos MODE=off && make ingest      # reprise : les mêmes lots passent
make rollback REASON="démo : retour à la version précédente"
make api-status                         # dans les 30 s : l'API sert la version previous, sans redémarrage
```

Montrer la carte « ↩️ Rollback » dans Teams, et la version affichée par l'interface (page Modèle).

### 17 – 19 min · Les tests de charge

`docs/SYNTHESE_OPS.md`, section 3 : **50 utilisateurs simultanés par réplica d'API à 1 CPU** (p95 51 ms), 100 avec
deux réplicas derrière nginx ; le CPU de l'API est le goulot ; pic ×5 sans erreur, retour en 60 s ; pas de fuite
mémoire sur 15 minutes. Recommandation : deux réplicas par défaut, un de plus par tranche de 50 utilisateurs.

### 19 – 20 min · Limites et pistes

`docs/SYNTHESE_OPS.md`, sections 4 et 5 : un modèle qui bat les baselines de peu, une seule machine, une sécurité de
démonstration ; et les pistes (features plus riches, scoring shadow avant promotion, réplicas).

## Si quelque chose casse

| Problème | Réponse |
|---|---|
| Une interface ne répond pas | `uv run tennis urls`, puis `docker compose ps` ; relancer : `uv run tennis up` (idempotent) |
| Pas de carte Teams | `make alerts` : l'alerte est dans l'historique (statut et raison) ; la montrer là |
| La démo en direct est impossible | **la vidéo de secours** (enregistrée la veille sur la même préparation), puis le rapport des tests d'acceptation (`uv run tennis report`) comme preuve |
| Il reste 2 minutes, pas le temps de l'incident | `uv run tennis scenarios --only S2` : le même incident, vérifié automatiquement en 15 secondes |

**Enregistrer la vidéo de secours** : refaire la préparation, lancer l'enregistrement d'écran de macOS
(Maj ⇧ + Cmd ⌘ + 5), dérouler les parties 4 à 7 min, 7 à 10, 12 à 15 et l'incident.

## Sous Windows (sans make)

| Commande `make` | Équivalent |
|---|---|
| `make simulate DAYS=7` | `uv run python -m tennis_mlops.flows.simulation --days 7` |
| `make ingest` | `uv run python -m tennis_mlops.flows.ingest` |
| `make registry` / `make trace` | `uv run python -m tennis_mlops.registry status` / `… trace` |
| `make rollback REASON="…"` | `uv run python -m tennis_mlops.registry rollback --reason "…"` |
| `make demo` / `make demo-stop` | `uv run tennis demo` / `uv run tennis demo-stop` |
| `make chaos MODE=corrupt`, `make advance DAYS=7`, `make clock` | depuis http://127.0.0.1:8001/docs : `POST /admin/chaos`, `POST /admin/clock/advance`, `GET /admin/clock` |
