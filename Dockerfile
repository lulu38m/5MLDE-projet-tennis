# Image unique du projet : mock_source aujourd'hui, puis l'API, le worker Prefect et l'interface.
# Dépendances installées depuis uv.lock (--locked) : l'image est reproductible.
FROM python:3.12.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Dépendances seules : couche mise en cache tant que uv.lock ne change pas.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# 2) Code et configuration du projet.
#    COPY conserve les droits de la machine hôte : un fichier en 600 y serait illisible par l'utilisateur non-root
#    (« No module named mock_source.__main__ »). On rend donc le code lisible par tous, quel que soit l'umask local.
COPY src ./src
COPY configs ./configs
RUN chmod -R a+rX /app/src /app/configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Les données ne sont jamais dans l'image (licence) : montées en lecture seule dans /app/data/raw.
RUN useradd --create-home --uid 10001 app && mkdir -p /app/data/raw && chown -R app /app/data
USER app

EXPOSE 8001
CMD ["python", "-m", "mock_source"]
