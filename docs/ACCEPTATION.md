# Parcours correcteur et tests d'acceptation

Objectif : cloner le dépôt et rejouer **toute** la chaîne en trois commandes, sans secret ni configuration.

```sh
git clone https://github.com/lulu38m/5MLDE-projet-tennis.git && cd 5MLDE-projet-tennis
uv run tennis up          # stack + données + premier modèle en production (~5 à 10 min la première fois)
uv run tennis scenarios   # les 9 tests d'acceptation, ~10 min ; rapport : uv run tennis report
```

Prérequis : **Docker Desktop** (Compose v2, 8 Go de mémoire conseillés), **uv** et **git**. Python 3.12 est installé
par uv. Mêmes commandes sous macOS, Linux et Windows (PowerShell) ; sous macOS et Linux, `make up`, `make scenarios`…
font la même chose. Les ports utilisés sont tous liés à `127.0.0.1` (`uv run tennis urls` les liste).

## Ce que fait `uv run tennis up` (ou `make up`)

1. crée `.env` depuis `.env.example` s'il n'existe pas : rien à remplir (sans webhook Teams, les alertes restent dans
   l'historique de `make alerts`) ;
2. télécharge les données à la version verrouillée (`make data`, ~45 Mo, vérifiées par SHA-256) ;
3. construit l'image et démarre la stack (`docker compose up --build --wait`) ;
4. **premier modèle, par la chaîne elle-même** (`python -m tennis_mlops.acceptance bootstrap`) : une ingestion
   amorce l'historique 2010 → 2015 contrôlé par Great Expectations ; son volume émet l'événement
   `tennis.training.requested` ; l'automation Prefect lance la campagne sur le worker ; la règle de mise en service
   place le meilleur candidat en `production` ; on attend que l'API le serve.

Idempotent : relancé sur une stack qui a déjà un modèle en production, il vérifie seulement que l'API le sert.

> Pourquoi l'historique va jusqu'à fin 2015 : avec 2010 → 2014 seulement, le meilleur candidat ne bat pas « le mieux
> classé gagne » en accuracy (68,9 % contre 69,3 %) et la règle de mise en service le refuse, à raison. Le rejeu
> commence donc le 1er janvier 2016 (`simulation.replay_start`, `MOCK_START_DATE`).

## `uv run tennis scenarios` (ou `make scenarios`)

| # | Scénario | Ce qui est fait | Vérifié automatiquement |
|---|---|---|---|
| S1 | Nominal | 2 pas de simulation : 1er janvier → 3 juin 2016, puis une semaine | lots validés par GX → réentraînement demandé → campagne sur le worker → nouvelle version, règle passée → alias `production` déplacé (`previous` = l'ancienne) → l'API sert la nouvelle version sans redémarrage → matchs à venir prédits → feedbacks enregistrés |
| S2 | Lot corrompu | `corrupt` pendant une semaine d'ingestion | ingestion en échec, lots en quarantaine, au moins une **expectation custom** en échec, aucun match stocké, aucun entraînement, alerte « quarantaine » reçue par `alert_sink`, production inchangée ; puis reprise : les mêmes lots passent |
| S3 | Source en 503 / lente | `503` ×2, `slow` ×1, puis `503` sans fin | retries visibles dans Prefect (task jouée 3 fois, puis 2 fois), succès final ; après épuisement (4 essais) état « Dégradé » et alerte |
| S4 | Source arrêtée | `docker compose stop mock_source` | ingestion « Dégradé », watermark inchangé, incident dans `pipeline_events`, alerte ; l'API sert toujours le dernier état validé ; source relancée, horloge remise à l'heure |
| S5 | Schéma modifié | `schema` pendant une semaine | contrôle de schéma (`expect_column_to_exist`) en échec, quarantaine, rien de stocké, alerte ; puis reprise |
| S6 | Modèle moins bon | `make train-weak` | version enregistrée avec `gate_status=failed`, `lifecycle=rejected`, gardée en `staging` ; `production` et version servie inchangées |
| S7 | Rollback | `make rollback REASON=…` | `production` = la version `previous`, l'ancienne marquée `rolled_back`, l'API la sert sans redémarrage (relecture de l'alias), alerte « rollback » |
| S8 | MLflow arrêté + entrée invalide | `docker compose stop mlflow` | `/ready` passe à `degraded`, prédiction servie depuis le cache, entrée invalide → `422` tracé dans `errors` ; MLflow relancé, `/ready` de nouveau `ready` |
| S9 | Interface de test | `make ui-smoke` | l'interface répond, la page Match prédit, la prédiction est dans `predictions` avec le client `ui` |

Chaque vérification est une ligne ✅ ou ❌ dans le terminal et dans le rapport. Code de sortie 0 si tout est vert.

```sh
uv run tennis scenarios --only S2,S5   # quelques scénarios seulement      (make scenarios ONLY=S2,S5)
uv run tennis scenarios --teams        # laisser partir les cartes Teams    (make scenarios TEAMS=1)
uv run tennis report                   # state/acceptance/rapport.html     (make scenarios-report)
```

### Point de départ connu

Les scénarios s'enchaînent sur la même stack (S7 annule la promotion faite par S1) et partent de l'état laissé par
`make up` : **une seule version, en production, horloge au 1er janvier 2016**. Toutes les données et tous les
entraînements étant déterministes, le résultat est le même sur toutes les machines. Sur une stack qui a déjà servi
(démo, entraînements), `scenarios` refuse de démarrer et donne la commande :

```sh
uv run tennis reset
uv run tennis up
uv run tennis scenarios
```

`reset` supprime les volumes (bases, MLflow, Grafana) et l'état local (`state/`), pas les données de `data/raw`.
`--force` (`FORCE=1` avec make) joue quand même, sans garantie sur S1 (la règle de mise en service peut alors refuser la nouvelle version).

Pourquoi S1 avance de cinq mois d'un coup : la règle ne met en production qu'une version **au moins aussi bonne** que
celle en service, mesurée sur le même test. Mesuré sur nos données, avec un, deux ou trois mois de 2016 en plus, la
nouvelle version reste légèrement moins bonne que v1 (log loss + 0,002 environ), et la règle la refuse, à raison ;
à partir de mai, elle la bat. S1 s'arrête donc le 3 juin 2016. La règle elle-même n'a pas été touchée.

### Ce que `make scenarios` touche, et remet en ordre

- **Pas de carte Teams** : le service d'alertes passe en **mode silence** le temps des tests (`POST /silence`) ; les
  alertes sont reçues et gardées dans l'historique avec le statut `silenced`, rien n'est envoyé. Le silence est levé
  à la fin, et s'arrête de lui-même au bout de 90 min si la campagne est interrompue.
- La simulation planifiée est mise en pause (`make demo` pour la relancer).
- Entre deux scénarios : incident de la source arrêté, services relancés s'ils ont été arrêtés, entraînements
  déclenchés par l'ingestion menés à leur terme. Un scénario raté n'empêche pas les suivants.
- Les flows sont lancés depuis la machine hôte, comme `make ingest` ou `make simulate` : ils apparaissent dans l'UI
  Prefect, et chaque section du rapport donne le lien du run ou de la version MLflow concernés.

## Code

| Fichier | Rôle |
|---|---|
| `src/tennis_mlops/cli.py` | `uv run tennis …` : le parcours, sans `make`, identique sous macOS, Linux et Windows |
| `src/tennis_mlops/acceptance/scenarios.py` | les 9 scénarios et leurs vérifications |
| `src/tennis_mlops/acceptance/runner.py` | vérifications préalables, silence, enchaînement, remise en ordre |
| `src/tennis_mlops/acceptance/stack.py` | la stack vue par ses interfaces publiques (API d'admin de la source, API, alertes, MLflow, Prefect, base en lecture) |
| `src/tennis_mlops/acceptance/bootstrap.py` | premier modèle à la fin de `make up` |
| `src/tennis_mlops/acceptance/report.py` | rapport HTML autonome et JSON |
| `tests/unit/acceptance/` | chaque scénario et le déroulé contre une stack en mémoire (sans Docker) |
