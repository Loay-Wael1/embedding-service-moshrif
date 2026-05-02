from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from app.settings import Settings, settings


logger = logging.getLogger("runtime_assets")


class RuntimeAssetError(RuntimeError):
    pass


def ensure_runtime_assets(config: Settings | None = None) -> None:
    """Ensure local model and Qdrant runtime assets exist.

    This function only checks/downloads files. It does not instantiate the
    embedding model, Qdrant client, or retriever.
    """
    active = config or settings
    model_path = Path(active.model_name)
    qdrant_path = Path(active.qdrant_path)

    if not _uses_managed_runtime_layout(model_path, qdrant_path):
        logger.info("Runtime asset download skipped for custom asset paths")
        return

    if _dir_has_files(model_path) and _dir_has_files(qdrant_path):
        logger.info("Runtime assets found locally")
        return

    if not active.hf_assets_download_enabled:
        missing = [str(path) for path in (model_path, qdrant_path) if not _dir_has_files(path)]
        raise RuntimeAssetError(
            "Runtime assets are missing and HF_ASSETS_DOWNLOAD_ENABLED=false: " + ", ".join(missing)
        )

    logger.info("Downloading runtime assets from HF repo %s ...", active.hf_assets_repo_id)
    download_root = Path(active.hf_assets_cache_dir)
    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=active.hf_assets_repo_id,
                repo_type=active.hf_assets_repo_type,
                revision=active.hf_assets_revision,
                local_dir=str(download_root),
                token=os.getenv("HF_TOKEN") or None,
            )
        )
        _copy_asset_dir(snapshot_path / "model" / "bge-m3", model_path)
        _copy_asset_dir(snapshot_path / "qdrant_db_legal", qdrant_path)
    except Exception as exc:
        raise RuntimeAssetError(
            f"Failed to download runtime assets from {active.hf_assets_repo_id}: {exc}"
        ) from exc

    if not _dir_has_files(model_path) or not _dir_has_files(qdrant_path):
        raise RuntimeAssetError(
            "Runtime asset download completed, but expected folders were not found at "
            f"{model_path} and {qdrant_path}."
        )

    logger.info("Runtime assets are ready")


def _dir_has_files(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def _uses_managed_runtime_layout(model_path: Path, qdrant_path: Path) -> bool:
    expected_model = (Path.cwd() / "model" / "bge-m3").resolve()
    expected_qdrant = (Path.cwd() / "qdrant_db_legal").resolve()
    try:
        return model_path.resolve() == expected_model and qdrant_path.resolve() == expected_qdrant
    except OSError:
        return False


def _copy_asset_dir(source: Path, target: Path) -> None:
    if not source.exists() or not source.is_dir():
        raise RuntimeAssetError(f"Expected asset folder not found in downloaded snapshot: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True, ignore=_ignore_runtime_files)


def _ignore_runtime_files(directory: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if name.endswith(".lock") or name == "__pycache__"}
    return ignored
