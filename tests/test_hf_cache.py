from pathlib import Path

from rainmap_nowcasting.hf import get_cached_model_files, has_model_files


def test_hf_cache_detection_uses_expected_filenames(tmp_path: Path) -> None:
    repo_id = "TechieMoon/rainmap-nowcasting"
    files = get_cached_model_files(repo_id=repo_id, cache_dir=tmp_path)

    assert not has_model_files(repo_id=repo_id, cache_dir=tmp_path)

    files.model_dir.mkdir(parents=True)
    files.config_path.write_text("{}", encoding="utf-8")
    files.weights_path.write_bytes(b"weights")

    assert has_model_files(repo_id=repo_id, cache_dir=tmp_path)
