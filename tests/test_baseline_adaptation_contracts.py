import pytest

from root_teller.baselines.llm import credentials, InferenceError, validate_ranking


def test_baseline_credentials_do_not_parse_private_workspace_files(tmp_path, monkeypatch):
    for name in ["ROOTTELLER_API_KEY", "OPENAI_API_KEY"]:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(InferenceError, match="environment|Set ROOTTELLER"):
        credentials(tmp_path)


def test_baseline_service_ranking_is_not_padded_or_repaired_from_labels():
    with pytest.raises(ValueError):
        validate_ranking({"ranking": ["a", "a"]}, ["a", "b"])
    with pytest.raises(ValueError):
        validate_ranking({"ranking": ["a", "unknown"]}, ["a", "b"])
    assert validate_ranking({"ranking": ["b", "a"]}, ["a", "b"]) == ["b", "a"]
