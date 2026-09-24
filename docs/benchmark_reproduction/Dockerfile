FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml /app/
RUN uv sync --no-dev --extra grading
RUN uv run --no-dev --extra grading python -m nltk.downloader -q \
    -d /app/nltk_data punkt punkt_tab
COPY common.py grade_one.py /app/
ENV NLTK_DATA=/app/nltk_data PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib
USER 65534:65534
ENTRYPOINT ["/app/.venv/bin/python", "/app/grade_one.py"]
