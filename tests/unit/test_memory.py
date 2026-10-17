import pytest

from memory.embedding import HashingEmbedder
from memory.store import MemoryStore, config_hash
from memory.vector import VectorIndex


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.db")


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_record_is_idempotent(store):
    args = ("ablation", "drop_usage", {"groups": ["customer"]}, {"val_pr_auc": 0.17})
    first_id, created = store.record_experiment(*args, dataset_sha256="d1")
    second_id, created_again = store.record_experiment(*args, dataset_sha256="d1")
    assert created and not created_again
    assert first_id == second_id


def test_same_experiment_on_new_data_is_a_new_record(store):
    args = ("training", "xgboost", {"depth": 3}, {"test_pr_auc": 0.27})
    a, _ = store.record_experiment(*args, dataset_sha256="d1")
    b, created = store.record_experiment(*args, dataset_sha256="d2")
    assert created and a != b


def test_already_tried(store):
    config = {"groups": ["usage"]}
    assert store.already_tried("ablation", "only_usage", config, "d1") is None
    store.record_experiment("ablation", "only_usage", config, {"x": 1}, conclusion="ok", dataset_sha256="d1")
    found = store.already_tried("ablation", "only_usage", config, "d1")
    assert found["conclusion"] == "ok"
    assert found["metrics"] == {"x": 1}


def test_search_finds_experiments_and_lessons(store):
    exp_id, _ = store.record_experiment(
        "ablation", "drop_usage", {"groups": ["customer"]}, {"v": 1},
        hypothesis="Does removing the usage feature group hurt?", verdict="group helps",
    )
    store.add_lesson("Usage features carry most of the signal", experiment_id=exp_id, tags="usage")
    store.record_experiment("ablation", "drop_risk", {"groups": ["usage"]}, {"v": 2}, hypothesis="risk group")

    results = store.search("usage signal")
    kinds = {r["type"] for r in results}
    assert "lessons" in kinds
    assert any(r["type"] == "experiments" and r["item"]["name"] == "drop_usage" for r in results)
    assert store.search("") == []
    assert store.search("zzzz-not-there") == []


def test_search_handles_punctuation(store):
    store.record_experiment("training", "xgboost", {}, {}, conclusion="PR-AUC 0.27 [0.22, 0.32]")
    assert store.search('PR-AUC "0.27" (xgboost)!')


@pytest.fixture
def hybrid(tmp_path):
    index = VectorIndex(HashingEmbedder(), tmp_path / "chroma")
    return MemoryStore(tmp_path / "memory.db", vectors=index)


def test_hybrid_search_marks_how_each_result_was_found(hybrid):
    hybrid.record_experiment(
        "ablation", "drop_usage", {"g": 1}, {}, hypothesis="removing usage features hurts the score"
    )
    hybrid.record_experiment("ablation", "drop_risk", {"g": 2}, {}, hypothesis="removing risk features")
    results = hybrid.search("usage features hurts")
    assert results[0]["item"]["name"] == "drop_usage"
    assert set(results[0]["found_by"]) == {"keyword", "vector"}


def test_vector_index_is_filled_on_write(hybrid):
    exp_id, _ = hybrid.record_experiment("training", "xgboost", {}, {}, conclusion="best model")
    hybrid.add_lesson("keep xgboost", experiment_id=exp_id)
    assert hybrid.vectors.count() == 2


def test_reindex_rebuilds_vectors_from_sqlite(tmp_path):
    plain = MemoryStore(tmp_path / "memory.db")
    plain.record_experiment("training", "xgboost", {}, {}, conclusion="best model")
    plain.add_lesson("keep xgboost")
    plain.close()

    index = VectorIndex(HashingEmbedder(), tmp_path / "chroma")
    store = MemoryStore(tmp_path / "memory.db", vectors=index)
    assert index.count() == 0
    assert store.reindex() == 2
    assert index.count() == 2


def test_each_embedder_gets_its_own_collection(tmp_path):
    class Other(HashingEmbedder):
        name = "other-embedder"

    a = VectorIndex(HashingEmbedder(), tmp_path / "chroma")
    b = VectorIndex(Other(), tmp_path / "chroma")
    a.upsert("lessons:1", "some text")
    assert a.count() == 1
    assert b.count() == 0