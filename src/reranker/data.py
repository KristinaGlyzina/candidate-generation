import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRanker

from text_features import (
    DEFAULT_VECTORIZER_CONFIG,
    FieldFeatureBuilder,
    clean_field,
)

from .candidates import candidate_union
from .config import SOURCE_NAMES
from .features import build_text_enriched_features
from .metrics import macro_metrics

ROOT = Path(__file__).resolve().parents[2]

FIELD_FEATURES_DIR = ROOT / "outputs" / "reranker_v2_field"
RETRIEVAL_DIR = ROOT / "outputs" / "hybrid_retrieval"

SOURCES = tuple(SOURCE_NAMES)

BASE_MODEL_CONFIG = {
    "loss_function": "YetiRank",
    "iterations": 600,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": 42,
}


def clean_text(value):
    if value is None or value is np.nan:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return " ".join(
        str(value)
        .lower()
        .replace("ё", "е")
        .split()
    )


def load_data():
    validation = pd.read_parquet(
        RETRIEVAL_DIR / "validation_queries.parquet"
    )

    assert isinstance(validation.index, pd.RangeIndex)
    assert validation.index.start == 0
    assert validation.index.step == 1

    validation["normalized_query"] = (
        validation["normalized_query"]
        .astype(str)
        .map(clean_text)
    )

    validation["normalized_params"] = (
        validation["search_infm_params_text"]
        .map(clean_text)
    )

    items = pd.read_parquet(
        ROOT / "data" / "benchmark_items.parquet"
    )

    assert len(validation) == 26549
    assert len(items) == 189212
    assert items["item_id"].astype(str).is_unique

    item_locations = items["item_location_id"].to_numpy()

    best_weights = (
        pd.read_csv(
            FIELD_FEATURES_DIR / "best_rrf_weights.csv"
        )
        .set_index("source")
        .weight
        .to_dict()
    )

    assert best_weights == {
        "word": 1.0,
        "char": 0.75,
        "e5_q0": 1.0,
        "e5_q1": 0.25,
    }

    indices = {
        source: np.load(
            RETRIEVAL_DIR / f"{source}_indices.npy",
            mmap_mode="r",
        )
        for source in SOURCES
    }

    scores = {
        source: np.load(
            RETRIEVAL_DIR / f"{source}_scores.npy",
            mmap_mode="r",
        )
        for source in SOURCES
    }

    for source in SOURCES:
        assert (
                indices[source].shape
                == scores[source].shape
                == (26549, 2000)
        )
        assert indices[source].dtype == np.int32
        assert scores[source].dtype == np.float32

    model = CatBoostRanker()
    model.load_model(
        str(FIELD_FEATURES_DIR / "catboost_ranker.cbm")
    )

    feature_names = model.feature_names_
    assert len(feature_names) == 40

    params = model.get_all_params()

    for key in (
            "loss_function",
            "iterations",
            "depth",
            "random_seed",
    ):
        assert params[key] == BASE_MODEL_CONFIG[key], (
            key,
            params[key],
            BASE_MODEL_CONFIG[key],
        )

    assert np.isclose(
        params["learning_rate"],
        BASE_MODEL_CONFIG["learning_rate"],
    )

    namespace = {
        "validation": validation,
        "item_locations": item_locations,
        "best_weights": best_weights,
        "indices": indices,
        "scores": scores,
        "FEATURE_NAMES": feature_names,
        "clean_text": clean_text,
        "macro_metrics": macro_metrics,
    }

    namespace["candidate_union"] = (
        lambda row_idx: candidate_union(
            namespace["indices"],
            row_idx,
        )
    )

    namespace["build_context_features"] = (
        lambda row_idx, candidate_items: build_text_enriched_features(
            row_idx,
            candidate_items,
            indices=namespace["indices"],
            scores=namespace["scores"],
            best_weights=namespace["best_weights"],
            validation=namespace["validation"],
            item_locations=namespace["item_locations"],
            field_builder=namespace["field_builder"],
        )
    )

    return namespace, items, model


def load_field_builder(items, validation):
    fields = (
        ("title", "item_title_raw"),
        ("params", "item_infm_params_text"),
        ("desc", "item_description_raw"),
    )

    texts = {
        field: [
            clean_field(
                value,
                max_chars=1000 if field == "desc" else None,
            )
            for value in items[column]
        ]
        for field, column in fields
    }

    cache_path = (
            FIELD_FEATURES_DIR
            / "field_tfidf_cache.joblib"
    )

    state = joblib.load(
        cache_path,
        mmap_mode="r",
    )

    assert state["n_items"] == len(items)
    assert state["use_desc_char"] is True
    assert state["desc_char_max_chars"] == 300
    assert state["config"] == DEFAULT_VECTORIZER_CONFIG, (
        "Field TF-IDF cache configuration mismatch"
    )

    del state
    gc.collect()

    builder = FieldFeatureBuilder(
        texts,
        use_desc_char=True,
        desc_char_max_chars=300,
        cache_path=cache_path,
    )

    builder.set_queries(
        validation["normalized_query"].tolist(),
        validation["normalized_params"].tolist(),
    )

    return builder
