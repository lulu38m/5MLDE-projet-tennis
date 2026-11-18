# Alertes Teams

Un incident de la chaîne produit **une** carte dans un canal Teams : un flow en échec après ses retries, un lot
refusé par Great Expectations, une source injoignable, un rollback. La carte dit ce qui s'est passé, pourquoi, ce
qui reste utilisable, ce qu'il faut faire, et donne un bouton vers le run Prefect (ou MLflow).

Code : `src/tennis_mlops/alerts/` (`cards.py` mise en page, `builders.py` contenu, `sink.py` service,
`teams.py` envoi, `prefect_reader.py`, `client.py`). Service : `alert_sink` dans `compose.yml`.

## Le webhook est un secret

L'URL du workflow Teams permet à n'importe qui de poster dans le canal (partagé avec la classe). Elle ne va que dans
`.env`, ignoré par Git :

```sh
TENNIS_TEAMS_WEBHOOK_URL=https://…   # jamais dans Git, une issue, un log ou un message
TENNIS_ALERT_OWNER=Lucas             # prénom affiché sur les cartes (facultatif)
```

Elle n'apparaît nulle part ailleurs : ni dans les logs du service, ni dans les messages d'erreur (en cas d'échec
d'envoi, seul le type d'erreur est gardé), ni dans `pipeline_events`, ni dans Prefect. Sans webhook, rien n'est
envoyé : les cartes sont construites et gardées dans l'historique (`make alerts`, statut « sans webhook »).

Avant tout envoi réel : `make alert-preview` écrit le JSON des cartes d'exemple dans `state/alerts/`, et
`make alert-test` envoie **une seule** carte marquée « 🧪 TEST » pour vérifier le webhook.

## Les cartes

| Type (`teams_events`) | Titre | Déclencheur | Envoyé par défaut |
|---|---|---|---|
| `echec` | ❌ Échec de … / 💥 Arrêt brutal de … | état final `Failed` ou `Crashed` d'un flow (après ses retries) | oui |
| `quarantaine` | 🚫 Lot refusé par Great Expectations | ingestion en `Failed` avec des lots en quarantaine | oui |
| `degrade` | ⚠️ Ingestion dégradée : source injoignable | ingestion à l'état `Dégradé` | oui |
| `rollback` | ↩️ Rollback : production vX → vY | `make rollback` | oui |
| `promotion` | ✅ Nouveau modèle en production | règle de mise en service passée, ou `make promote` | non |

Une version **refusée** par la règle n'envoie jamais de carte : c'est une décision normale, visible dans MLflow
et dans `make registry`. La promotion est désactivée par défaut pour ne pas remplir un canal partagé avec de bonnes
nouvelles ; elle s'active en ajoutant `promotion` à `teams_events` (`configs/project.yaml`).

Structure commune : bandeau coloré (rouge : agir ; orange : dégradé ou retour arrière ; vert : bonne nouvelle) avec
le titre — **seul le titre porte un emoji** —, la cause, trois chiffres clés, de quoi retrouver l'exécution (run,
événement, poste), la marche à suivre, et les boutons. Les liens pointent vers `127.0.0.1` : ils ne s'ouvrent que
sur le poste qui fait tourner la stack, ce que rappelle le pied de carte.

Contenu, par carte :

- **Échec** : la cause (message de l'exception, sans l'en-tête de Prefect), la première task en échec, la version
  en production (« v9 inchangée » : l'API n'est pas touchée), la campagne MLflow si elle a été ouverte.
- **Lot refusé** : le tournoi, les contrôles en échec en clair, la cause d'abord (« Colonnes absentes : … », puis
  le nombre de contrôles en échec par conséquence), des exemples de valeurs, le dossier de quarantaine.
- **Dégradé** : jusqu'où les données en base sont utilisables, jusqu'où l'ingestion est arrivée.
- **Rollback** : la raison donnée, la version servie, et si un deuxième rollback reste possible.

## Le service `alert_sink`

```
automation Prefect (Failed, Crashed, Dégradé) ─► POST /prefect   ─┐
registre (rollback, promotion)                 ─► POST /registry ─┼─► carte ─► filtre ─► Teams
                                                                   └─► pipeline_events (kind = alert)
```

- **Enrichissement** : l'automation ne transmet que trois identifiants (événement, nom de l'événement, flow run) ;
  le service relit le flow run dans l'API Prefect, les lots refusés dans `pipeline_events`, la campagne dans
  MLflow (tag `prefect.flow_run_id`), la version servie et l'état des données dans la base.
- **Un événement n'est traité qu'une fois** (identifiant de l'événement Prefect, ou de la décision du registre).
- **Pas de rafale** : la simulation repasse chaque minute. Un incident qui persiste (le même lot refusé à
  chaque pas, la source toujours en panne) ne renvoie une carte qu'après `alert_cooldown_min` (30 min) ; entre-temps
  l'alerte est gardée avec le statut « répétée ».
- **Jamais bloquant** : une lecture qui échoue (MLflow arrêté, base injoignable) donne une carte moins riche, pas
  d'erreur. Côté registre et entraînement, un service d'alertes injoignable est signalé en une ligne ; la décision,
  elle, est déjà appliquée.

- **Mode silence** (phase 11) : pendant `make scenarios`, le service ne poste rien sur Teams (le canal est partagé
  avec la classe) ; chaque alerte est construite et gardée avec le statut « en silence ». `POST /silence`
  (`{"minutes": 90, "reason": "…"}`) le pose, `DELETE /silence` le lève ; il s'arrête seul à l'heure dite.
  `make scenarios TEAMS=1` laisse partir les cartes.

API (`http://127.0.0.1:8090/docs`) : `POST /prefect`, `POST /registry`, `GET /alerts?limit=20`, `GET /health`,
`POST /silence`, `DELETE /silence`.

```sh
make alerts
# 12/11 13:11:31  répétée       ❌ Échec de l'entraînement
# 12/11 13:11:08  envoyée       ❌ Échec de l'entraînement
# 12/11 13:10:44  envoyée       ↩️ Rollback : production v2 → v1
# 12/11 13:10:39  filtrée       ✅ Nouveau modèle en production : atp-winner v2
```

## Démontrer chaque carte

Avec la démo lancée (`make demo`), sans rien envoyer de réel tant que le webhook n'est pas dans `.env` :

| Carte | Commande |
|---|---|
| Lot refusé | `make chaos MODE=schema` (puis `make chaos MODE=off`) |
| Dégradé | `make chaos MODE=503 FAILURES=20` |
| Échec de l'entraînement | `docker compose stop mlflow`, lancer `entrainement-atp/sur-evenement` depuis l'UI, puis `docker compose start mlflow` |
| Rollback | `make rollback REASON="…"` (il faut une version `previous`) |
