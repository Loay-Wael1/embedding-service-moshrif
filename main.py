"""Entrypoint for the Egyptian Laws Embedding Service API."""
from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").strip().lower() in {"1", "true", "yes", "on"}

    print(f"Starting API server on {host}:{port} ...")
    uvicorn.run("app.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
