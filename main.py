"""Entrypoint for the Egyptian Laws Embedding Service API."""
from __future__ import annotations

import logging
import os


def main() -> None:
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "7860"))
    reload = os.getenv("API_RELOAD", "false").strip().lower() in {"1", "true", "yes", "on"}

    # Production log level.
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    print(f"Starting API server on {host}:{port} ...")
    uvicorn.run(
        "app.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        workers=1,  # Single worker for local Qdrant safety.
    )


if __name__ == "__main__":
    main()
