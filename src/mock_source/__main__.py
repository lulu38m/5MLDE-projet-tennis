"""Lance le simulateur : `python -m mock_source` (variables `MOCK_*` pour l'hôte, le port, la date de départ)."""

import uvicorn

from mock_source.service import MockSettings


def main() -> None:
    """Démarre uvicorn avec la fabrique d'application."""
    settings = MockSettings()
    uvicorn.run("mock_source.service:build_app", factory=True, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
