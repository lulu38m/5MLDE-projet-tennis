"""Erreurs du domaine données. Chacune correspond à une décision opérationnelle différente."""


class DataError(Exception):
    """Racine des erreurs liées aux données."""


class SourceUnavailableError(DataError):
    """La source est injoignable : on peut basculer sur la source suivante (fallback)."""


class DataIntegrityError(DataError):
    """Le contenu ne correspond pas au fichier de verrouillage : bloquant, jamais de fallback silencieux."""


class DataAccessError(DataError):
    """Les fichiers existent mais ne sont pas lisibles par le processus (droits, utilisateur du conteneur)."""


class SourceRequestError(DataError):
    """La source a répondu mais refuse la requête (4xx) ou rompt son contrat : réessayer ne changera rien."""


class DataQualityError(DataError):
    """Un lot ne respecte pas les contrôles bloquants : il est mis en quarantaine et l'entraînement n'a pas lieu."""
