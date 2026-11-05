"""Simulateur de source de données ATP : rejoue l'historique Sackmann selon une horloge simulée.

Service séparé de la chaîne MLOps (il joue le rôle d'un fournisseur externe), mais construit dans le même dépôt :
il lit exactement les fichiers verrouillés par `configs/data.lock.json` et expose leur `dataset_version`.
"""
