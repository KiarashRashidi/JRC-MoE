import yaml
import pytest

from jrc_moe.multidataset_baseline import (
    DATASET_ORDER,
    stable_sample,
    validate_config,
)


def test_config_declares_balanced_reflex_only_40_case_protocol():
    with open("configs/reflex_multidataset_40.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    validate_config(config)
    assert config["experiment"]["mode"] == "reflex"
    assert tuple(config["datasets"]) == DATASET_ORDER
    assert sum(item["count"] for item in config["datasets"].values()) == 40


def test_stable_sample_is_independent_of_source_order():
    rows = [{"id": str(i), "value": i} for i in range(30)]
    forward = stable_sample(rows, 10, 19, lambda row: row["id"])
    backward = stable_sample(list(reversed(rows)), 10, 19, lambda row: row["id"])

    assert [example_id for example_id, _ in forward] == [
        example_id for example_id, _ in backward
    ]


def test_stable_sample_rejects_duplicate_ids_and_oversampling():
    with pytest.raises(ValueError, match="duplicate stable IDs"):
        stable_sample([{"id": 1}, {"id": 1}], 1, 1, lambda row: row["id"])
    with pytest.raises(ValueError, match="only 1 records"):
        stable_sample([{"id": 1}], 2, 1, lambda row: row["id"])


def test_config_rejects_generation_lengths_not_aligned_to_decoder_blocks():
    with open("configs/reflex_multidataset_40.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["datasets"]["humaneval"]["generation_length"] = 1000

    with pytest.raises(ValueError, match="positive multiple of 32"):
        validate_config(config)


def test_legacy_reflex_controller_k_is_expanded_for_per_layer_trace_accounting():
    pytest.importorskip("transformers")
    import torch
    from jrc_moe.generation import _effective_k_by_layer

    shared_k = torch.tensor([8, 8, 6, 6])
    assert _effective_k_by_layer({}, shared_k, 2) == ((8, 8, 6, 6), (8, 8, 6, 6))
    layerwise = ((8, 7, 6), (8, 6, 5))
    assert _effective_k_by_layer({"k_by_layer": layerwise}, shared_k, 2) == layerwise

