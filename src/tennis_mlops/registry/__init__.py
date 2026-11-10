"""Registre des modèles : stratégie de tagging, règle de mise en service, rollback, journal des décisions.

Une campagne enregistre une version (`training`) ; ce paquet décide si elle est servie (alias `production`), garde
la trace de chaque décision et permet de revenir en arrière en une commande. Détail : docs/TAGGING.md.
"""
