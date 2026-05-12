from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from . import DEFAULT_CONFIG_FILENAME, DEFAULT_REPO_ID, DEFAULT_WEIGHTS_FILENAME
from .config import read_json
from .runtime import default_model_cache_dir


@dataclass(frozen=True)
class ModelFiles:
    model_dir: Path
    weights_path: Path
    config_path: Path
    repo_id: str


def local_model_dir(repo_id: str = DEFAULT_REPO_ID, cache_dir: str | Path | None = None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else default_model_cache_dir()
    return base / repo_id.replace("/", "--")


def has_model_files(
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: str | Path | None = None,
    weights_filename: str = DEFAULT_WEIGHTS_FILENAME,
    config_filename: str = DEFAULT_CONFIG_FILENAME,
) -> bool:
    model_dir = local_model_dir(repo_id, cache_dir)
    config_path = model_dir / config_filename
    if not config_path.exists():
        return False
    try:
        config = read_json(config_path)
        weights_filename = str(config.get("weights", weights_filename))
    except Exception:
        pass
    return (model_dir / weights_filename).exists()


def get_cached_model_files(
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: str | Path | None = None,
    weights_filename: str = DEFAULT_WEIGHTS_FILENAME,
    config_filename: str = DEFAULT_CONFIG_FILENAME,
) -> ModelFiles:
    model_dir = local_model_dir(repo_id, cache_dir)
    return ModelFiles(
        model_dir=model_dir,
        weights_path=model_dir / weights_filename,
        config_path=model_dir / config_filename,
        repo_id=repo_id,
    )


def model_files_from_dir(
    model_dir: str | Path,
    weights_filename: str | None = None,
    config_filename: str = DEFAULT_CONFIG_FILENAME,
    repo_id: str = "local",
) -> ModelFiles:
    root = Path(model_dir)
    config_path = root / config_filename
    if weights_filename is None:
        config = read_json(config_path)
        weights_filename = str(config.get("weights", DEFAULT_WEIGHTS_FILENAME))
    return ModelFiles(
        model_dir=root,
        weights_path=root / weights_filename,
        config_path=config_path,
        repo_id=repo_id,
    )


def ensure_model_files(
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: str | Path | None = None,
    weights_filename: str = DEFAULT_WEIGHTS_FILENAME,
    config_filename: str = DEFAULT_CONFIG_FILENAME,
    force_download: bool = False,
) -> ModelFiles:
    files = get_cached_model_files(repo_id, cache_dir, weights_filename, config_filename)
    files.model_dir.mkdir(parents=True, exist_ok=True)

    if files.config_path.exists():
        config = read_json(files.config_path)
        configured_weights = str(config.get("weights", weights_filename))
        files = get_cached_model_files(repo_id, cache_dir, configured_weights, config_filename)
        if not force_download and files.weights_path.exists():
            return files

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=config_filename,
            local_dir=files.model_dir,
            force_download=force_download,
        )
        config = read_json(files.config_path)
        configured_weights = str(config.get("weights", weights_filename))
        files = get_cached_model_files(repo_id, cache_dir, configured_weights, config_filename)
        hf_hub_download(
            repo_id=repo_id,
            filename=configured_weights,
            local_dir=files.model_dir,
            force_download=force_download,
        )
    except Exception:
        if files.weights_path.exists() and files.config_path.exists():
            return files
        raise

    return files
