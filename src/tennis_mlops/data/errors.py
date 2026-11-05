"""Erreurs du domaine données. Chacune correspond à une décision opérationnelle différente."""


class DataError(Exception):
    """Racine des erreurs liées aux données."""


class SourceUnavailableError(DataError):
    """La source est injoignable : on peut basculer sur la source suivante (fallback)."""


class DataIntegrityError(DataError):
    """Le contenu ne correspond pas au fichier de verrouillage : bloquant, jamais de fallback silencieux."""


class DataAccessError(DataError):
    """Les fichiers existent mais ne sont pas lisibles par le processus (droits, utilisateur du conteneur)."""
