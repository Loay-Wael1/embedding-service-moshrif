"""List all FastAPI routes with method, path, name, and tags."""
from __future__ import annotations

from app.api.app import app


def main() -> None:
    print(f"{'Method':<8} {'Path':<25} {'Name':<25} {'Tags'}")
    print("-" * 85)
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", getattr(route, "path_format", "?"))
        name = getattr(route, "name", "?")
        tags = getattr(route, "tags", [])
        endpoint = getattr(route, "endpoint", None)
        func_name = getattr(endpoint, "__name__", "?") if endpoint else "?"
        if methods:
            for m in sorted(methods):
                print(f"{m:<8} {path:<25} {name:<25} {tags or ''}")
        else:
            print(f"{'?':<8} {path:<25} {name:<25} {tags or ''}")


if __name__ == "__main__":
    main()
