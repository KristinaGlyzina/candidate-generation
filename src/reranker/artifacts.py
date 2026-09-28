import numpy as np
from catboost import CatBoostRanker

from artifact_utils import read_json, validate_npy
from reranker.data import ROOT

ARTIFACTS_DIR = ROOT / "outputs" / "reranker_v3_full_scores"

N_FEATURES = 44
N_DEV_CONTEXTS = 5309
N_TRAIN_CONTEXTS = 21240
MAX_SAMPLE_ROWS = 4_779_000


def completed_feature_specs():
    config = read_json(ARTIFACTS_DIR / "config.json")
    assert config is not None

    dev_offsets = validate_npy(
        ARTIFACTS_DIR / "dev_offsets.npy",
        (N_DEV_CONTEXTS + 1,),
        np.int64,
    )
    dev_rows = int(dev_offsets[-1])

    return config, dev_rows, {
        "X_dev.npy": ((dev_rows, N_FEATURES), np.float32),
        "dev_candidates.npy": ((dev_rows,), np.int32),
        "X_train.npy": (
            (int(config["training_rows"]), N_FEATURES),
            np.float32,
        ),
        "sample_label.npy": ((MAX_SAMPLE_ROWS,), np.float32),
        "sample_group.npy": ((MAX_SAMPLE_ROWS,), np.int32),
        "sample_validation_row.npy": ((MAX_SAMPLE_ROWS,), np.int32),
        "sample_item_idx.npy": ((MAX_SAMPLE_ROWS,), np.int32),
        "sample_offsets.npy": ((N_TRAIN_CONTEXTS + 1,), np.int64),
    }


def validate_completed_feature_caches():
    complete = read_json(ARTIFACTS_DIR / "features_complete.json")
    dev_progress = read_json(ARTIFACTS_DIR / "dev_features_progress.json")
    train_progress = read_json(ARTIFACTS_DIR / "train_features_progress.json")
    sampling = read_json(ARTIFACTS_DIR / "sampling_complete.json")

    config, _, specs = completed_feature_specs()

    assert complete == {
        "rows": int(config["training_rows"]),
        "features": N_FEATURES,
    }
    assert dev_progress["next_context"] == N_DEV_CONTEXTS
    assert train_progress["next_context"] == N_TRAIN_CONTEXTS
    assert sampling["next_context"] == N_TRAIN_CONTEXTS
    assert sampling["cursor"] == int(config["training_rows"])

    for name, (shape, dtype) in specs.items():
        validate_npy(ARTIFACTS_DIR / name, shape, dtype)

    with np.load(
            ARTIFACTS_DIR / "training_rows.npz",
            allow_pickle=False,
    ) as archive:
        expected = {
            "y": ((int(config["training_rows"]),), np.float32),
            "group": ((int(config["training_rows"]),), np.int32),
            "validation_row": (
                (int(config["training_rows"]),),
                np.int32,
            ),
            "item_idx": (
                (int(config["training_rows"]),),
                np.int32,
            ),
        }

        assert set(archive.files) == set(expected)

        for name, (shape, dtype) in expected.items():
            assert archive[name].shape == shape
            assert archive[name].dtype == np.dtype(dtype)

    model_path = ARTIFACTS_DIR / "catboost_ranker.cbm"

    if model_path.exists():
        model = CatBoostRanker()
        model.load_model(model_path)

        assert model.tree_count_ == 600
        assert model.feature_names_ == config["features"]
        assert len(model.feature_names_) == N_FEATURES

    return True
