# Tests de charge et dimensionnement

Combien d'utilisateurs simultanés l'API de prédiction tient-elle sur notre machine, dans quelle configuration, et
que se passe-t-il au-delà ? Les réponses chiffrées sont dans `loadtest/results/<campagne>/report.md` et reprises
dans [`SYNTHESE_OPS.md`](SYNTHESE_OPS.md) : **B tient 50 utilisateurs, C (2 × B) 100, A 10**.

```sh
make demo-stop    # la simulation en pause : ses runs ne doivent pas se mêler aux mesures
make load         # campagne courte (~6 min, config B) : de quoi vérifier que tout marche
make load-full    # campagne complète (~75 min) : configs A, B, C, rampe, pic, endurance
make load-ui      # Locust avec son interface (http://127.0.0.1:8089), pour une démo
```

## Machine de référence

| | |
|---|---|
| Machine | MacBook Pro (Mac17,2), Apple M5, 10 cœurs, 16 Go |
| Docker | Docker Desktop, moteur 29.9.2, **10 CPU et 7,7 Go alloués** à la VM |
| Locust | sur la même machine, 4 processus (`configs/load.yaml`) |

Le rapport de chaque campagne relit et note ces informations (`sysctl`, `docker info`).

## Ce qu'on appelle un utilisateur, et l'objectif

Un **utilisateur** Locust envoie une requête, attend 0,5 à 1,5 s, recommence : environ **une requête par seconde**.
Un palier est **tenu** si, une fois la montée passée :

- 95 % des requêtes unitaires répondent en moins de **200 ms** (p95) ;
- 95 % des batchs de 50 matchs répondent en moins de **1 s** ;
- moins de **1 %** des requêtes échouent (code hors 2xx, ou pas de réponse après 10 s).

Le **nombre maximal d'utilisateurs simultanés** d'une configuration est le dernier palier tenu, tous les
précédents l'étant aussi.

## Scénarios (`loadtest/locustfile.py`)

| Utilisateur | Ce qu'il fait |
|---|---|
| `PredictUser` | `POST /predict`, deux joueurs tirés parmi les 200 classés de l'année, surface tirée au hasard |
| `BatchUser` | `POST /predict/batch` de 50 matchs (un tableau à prédire d'un coup) |
| `MixUser` | usage réaliste : 12 prédictions pour 3 recherches de joueur, 2 fiches joueur, 2 face-à-face, 1 batch |

Chaque requête porte `X-Client: loadtest` : elle est tracée dans `predictions` et comptée à part dans Prometheus,
sans se mêler au scoring ni à l'interface. Elle compte en revanche dans la fenêtre de dérive de l'API
([MONITORING.md](MONITORING.md)), et c'est normal : des paires tirées au hasard ne ressemblent pas à un tableau.

## Protocole (`loadtest/run_campaign.py`, `configs/load.yaml`)

| Scénario | Configs | Profil | Ce qu'on en tire |
|---|---|---|---|
| **Rampe** (mix) | A, B, C | 10 → 50 → 100 → 200 → 500 → 1 000 utilisateurs, 2 min par palier | max d'utilisateurs, débit, CPU et mémoire au max |
| Rampe `predict` | B | 10 → 500, 1 min par palier | coût d'une prédiction seule |
| Rampe `batch` | B | 1 → 50 utilisateurs (50 → 2 500 matchs/s demandés) | débit de prédictions en batch |
| **Pic** | B | 50 utilisateurs, ×5 en 10 s pendant 90 s, retour à 50 | dégradation, erreurs, temps de retour sous l'objectif |
| **Endurance** | B | 50 utilisateurs (une charge que B tient) pendant 15 min | pente de la mémoire de l'API (fuite ?), stabilité du p95 |

Pour chaque scénario, la campagne :

1. remet l'API dans la configuration voulue (`docker compose -f compose.yml -f loadtest/compose.<X>.yml up`) et
   attend `GET /ready` ;
2. lance Locust sans interface (`--headless`, 4 processus) sur le profil de paliers (`loadtest/shape.py`) ; chaque
   requête est journalisée (heure, endpoint, durée, succès) ;
3. calcule par palier, **après sa montée** (20 s), le débit, le taux d'erreurs et les percentiles **exacts**
   (p50, p95, p99) de toutes ses requêtes, et le p95 des batchs à part ;
4. lit dans Prometheus le **CPU moyen et la mémoire maximale** de l'API (tous réplicas), de PostgreSQL et de
   l'équilibreur pendant le palier (exporteur de la phase 9, comme `docker stats`), et mesure le CPU de Locust ;
5. écrit `stages.csv`, `endpoints.csv` (par endpoint), `timeline.csv` (par tranche de 10 s), `summary.json`, les
   graphiques et `report.md`.

À la fin, l'API revient à sa configuration de `compose.yml`, même si la campagne est interrompue.

Sur macOS, la campagne empêche la mise en veille tant qu'elle tourne (`caffeinate`) : une veille fige Locust et la
VM de Docker. Un scénario où plus aucune requête ne part pendant plus de 10 s est marqué **interrompu** dans le
rapport, avec la commande pour le rejouer seul dans le même dossier ; les autres scénarios du dossier sont alors
réanalysés pour que le rapport reste complet :

```sh
uv run --locked python -m loadtest.run_campaign full --only B-ramp --out loadtest/results/<campagne>
uv run --locked python -m loadtest.run_campaign full --report-only --out loadtest/results/<campagne>  # rapport seul
```

## Les configurations comparées

| | Réplicas | CPU | Mémoire | Workers uvicorn | Fichier |
|---|---|---|---|---|---|
| **A** | 1 | 0,5 | 512 Mo | 1 | `loadtest/compose.A.yml` |
| **B** | 1 | 1 | 1,5 Go | 2 | `loadtest/compose.B.yml` |
| **C** | 2 × B, derrière nginx | 2 (+ 0,5 nginx) | 2 × 1,5 Go | 2 par réplica | `loadtest/compose.C.yml`, `loadtest/nginx.conf` |

- Chaque worker charge son modèle et rejoue l'historique : la mémoire de B est prévue pour deux.
- En C, nginx prend le port publié de l'API et répartit entre les réplicas ; le scoring et l'interface joignent
  toujours `http://api:8000`, que Docker répartit aussi.
- Avec plusieurs workers ou réplicas, `/metrics` ne montre que le processus qui répond à Prometheus : les mesures de
  la campagne viennent de Locust et des conteneurs, pas de `/metrics`.

## Résultats

`loadtest/results/<date>-<campagne>/` : `report.md` (le résumé), `comparison.png` (p95 et débit des trois
configurations), un dossier par scénario (`B-ramp/`, `B-spike/`…). Le journal brut des requêtes (`raw/`) reste en
local. La synthèse et la recommandation de dimensionnement sont dans [`SYNTHESE_OPS.md`](SYNTHESE_OPS.md).

## Biais et limites

- **Locust tourne sur la même machine** que la stack et lui prend du CPU : sa consommation est mesurée et donnée
  dans le rapport. Les capacités mesurées sont un minimum.
- Docker Desktop exécute les conteneurs dans une VM (10 CPU, 7,7 Go) : la limite `cpus` d'un conteneur est une part
  de cette VM, pas d'un cœur physique dédié.
- Le reste de la stack tourne pendant les mesures (PostgreSQL reçoit les traces de chaque prédiction, Prometheus
  collecte) : c'est voulu, c'est l'API telle qu'elle est servie. La simulation, elle, est en pause.

## Tests

`tests/loadtest/test_loadtest.py` : profils (rampe, pic, fin du test), journal des requêtes, percentiles exacts
après la montée, objectif séparé pour les batchs, erreurs, nombre maximal d'utilisateurs, temps de retour après un
pic, détection d'une fuite mémoire, et rapport complet reconstruit depuis des journaux enregistrés. Locust lui-même
n'est pas importé par les tests (il remplace le réseau de Python à l'import).
