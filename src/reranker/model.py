from __future__ import annotations

import gc
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from rapidfuzz.fuzz import ratio as fuzz_ratio
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import context_features
from reranker.config import SOURCE_NAMES
from reranker.data import load_data, load_field_builder
from reranker.features import build_stage1_base_features

BASE_FEATURES_DIR = ROOT / "outputs/reranker_v3_full_scores"
STAGE1_DIR = ROOT / "outputs/reranker_v4_context"
TITLE_RETRIEVAL_DIR = ROOT / "outputs/title_e5_ablation"
LOCATION_RETRIEVAL_DIR = ROOT / "outputs/location_e5_ablation"
FINAL_RANKER_DIR = ROOT / "outputs/reranker_v5_light"

ITEM_EMBEDDINGS_PATH = ROOT / "outputs/e5_small_item_embeddings.npy"

TITLE_TOP_K = 500
LOCATION_TOP_K = 50

SEED = 42
N_HARD_PER_SOURCE = 50
N_RANDOM = 50
EPS = 1e-9

FINAL_RANKER_DIR.mkdir(parents=True, exist_ok=True)

FINAL_FEATURE_NAMES = [
    "v4_score",
    "v4_score_missing",
    "is_old_union",
    "title_e5_score",
    "title_e5_rank",
    "title_e5_present",
    "title_e5_score_z",
    "location_e5_score",
    "location_e5_rank",
    "location_e5_present",
    "location_e5_score_z",
    "same_location",
    "query_title_e5_cosine",
    "query_item_text_e5_cosine",
    "word_overlap_query_title",
    "char_similarity_query_title",
    "num_sources",
    "new_source_rrf",
]


def clean_text(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).lower().replace("ё", "е"),
    ).strip()


def factorize_pair(item_values, query_values):
    item_codes, uniques = pd.factorize(pd.Series(item_values).astype(str))
    query_codes = pd.Index(uniques).get_indexer(pd.Series(query_values).astype(str))

    return item_codes.astype(np.int32), query_codes.astype(np.int32)


def detect_category_column(items):
    columns = [column for column in items.columns if "categ" in column.lower()]
    return columns[0] if columns else None


def build_stage1_feature_builder():
    namespace, items, _ = load_data()

    validation = namespace["validation"]
    normalize = namespace["clean_text"]

    candidates = np.load(BASE_FEATURES_DIR / "union_items.npy", mmap_mode="r")
    offsets = np.load(BASE_FEATURES_DIR / "union_offsets.npy")

    full_scores = {
        source: np.load(
            BASE_FEATURES_DIR / f"{source}_full_score.npy",
            mmap_mode="r",
        )
        for source in context_features.SOURCES
    }

    item_locations, query_locations = factorize_pair(
        items.item_location_id,
        validation.search_location_id,
    )

    location_coverage = float(np.mean(query_locations >= 0))
    assert location_coverage > 0.5

    category_column = detect_category_column(items)

    item_categories = None
    query_categories = None

    if category_column is not None:
        item_categories, query_categories = factorize_pair(
            items[category_column],
            validation.search_category,
        )

    query_groups = pd.factorize(
        validation.search_query.map(normalize)
    )[0]

    grouped_rows = pd.Series(
        np.arange(len(validation))
    ).groupby(query_groups)

    group_rows = {
        int(group): rows.to_numpy()
        for group, rows in grouped_rows
    }

    query_arrays = context_features.make_query_arrays(
        validation.normalized_query.tolist(),
        validation.normalized_params.tolist(),
        validation.search_is_delivery_search.astype(np.float32).to_numpy(),
        query_locations,
        query_categories,
        query_groups,
    )

    item_static = context_features.item_static_matrix(
        items.item_title_raw.fillna("").astype(str),
        items.item_infm_params_text.fillna("").astype(str),
        items.item_description_raw.fillna("").astype(str),
    )

    hub_path = STAGE1_DIR / "hub_stats.npz"

    if hub_path.exists():
        with np.load(hub_path) as archive:
            hub_count = archive["count"]

            hub_sums = {
                source: archive[f"sum_{source}"]
                for source in context_features.HUB_SOURCES
            }
    else:
        hub_count, hub_sums = context_features.hub_stats(
            candidates,
            full_scores,
            len(items),
        )

        tmp_path = STAGE1_DIR / "hub_stats.tmp.npz"

        np.savez(
            tmp_path,
            count=hub_count,
            **{
                f"sum_{source}": values
                for source, values in hub_sums.items()
            },
        )

        tmp_path.replace(hub_path)

    embeddings = np.load(ITEM_EMBEDDINGS_PATH, mmap_mode="r")

    assert embeddings.shape == (len(items), 384)

    builder = context_features.ContextFeatureBuilder(
        candidates=candidates,
        offsets=offsets,
        full=full_scores,
        embeddings=embeddings,
        item_static=item_static,
        item_loc=item_locations,
        item_cat=item_categories,
        q=query_arrays,
        group_rows=group_rows,
        hub_count=hub_count,
        hub_sums=hub_sums,
    )

    return builder, namespace, query_groups


def macro_metrics(ranked, positives, rows):
    recalls = []
    query_hits = []

    hits = 0
    total = 0

    for prediction, row in zip(ranked, rows):
        positive_items = positives[int(row)]

        n_hits = len(
            set(map(int, prediction[:50]))
            & positive_items
        )

        recalls.append(n_hits / len(positive_items))
        query_hits.append(float(n_hits > 0))

        hits += n_hits
        total += len(positive_items)

    return {
        "macro_recall": float(np.mean(recalls)),
        "query_hit": float(np.mean(query_hits)),
        "pair_hit": float(hits / total),
    }


def source_map(indices, scores, k, valid_nonnegative=False):
    indices = np.asarray(indices[:k], dtype=np.int32)
    scores = np.asarray(scores[:k], dtype=np.float32)

    if valid_nonnegative:
        mask = indices >= 0
        indices = indices[mask]
        scores = scores[mask]

    return {
        int(item): (float(score), rank)
        for rank, (item, score) in enumerate(
            zip(indices, scores),
            start=1,
        )
    }


def z_params(scores):
    values = np.asarray(scores, dtype=np.float32)
    values = values[np.isfinite(values)]

    if not len(values):
        return 0.0, 1.0

    return float(values.mean()), float(values.std() + EPS)


def build_expanded_pool(
        row,
        stage1_candidates,
        title_indices,
        location_indices,
):
    location_candidates = np.asarray(
        location_indices[row, :LOCATION_TOP_K],
        dtype=np.int32,
    )

    location_candidates = location_candidates[
        location_candidates >= 0
        ]

    return np.unique(
        np.concatenate(
            [
                np.asarray(stage1_candidates, dtype=np.int32),
                np.asarray(title_indices[row, :TITLE_TOP_K], dtype=np.int32),
                location_candidates,
            ]
        )
    ).astype(np.int32)


def lexical_features(query, titles):
    query = clean_text(query)
    query_tokens = set(query.split())

    denominator = max(len(query_tokens), 1)

    overlap = np.empty(len(titles), dtype=np.float32)
    similarity = np.empty(len(titles), dtype=np.float32)

    for index, title in enumerate(titles):
        title = clean_text(title)

        overlap[index] = (
                len(query_tokens & set(title.split()))
                / denominator
        )

        similarity[index] = (
            fuzz_ratio(query, title) / 100.0
            if query or title
            else 0.0
        )

    return overlap, similarity


def build_final_features(
        row,
        pool_items,
        stage1_candidates,
        stage1_scores,
        *,
        validation,
        items,
        title_idx,
        title_scores,
        loc_idx,
        loc_scores,
        qemb,
        item_emb,
        title_emb,
        original_indices,
):
    pool_items = np.asarray(pool_items, dtype=np.int32)
    n_items = len(pool_items)

    item_to_position = {
        int(item): position
        for position, item in enumerate(pool_items)
    }

    stage1_feature = np.full(n_items, np.nan, dtype=np.float32)

    stage1_positions = np.searchsorted(
        pool_items,
        stage1_candidates,
    )

    np.testing.assert_array_equal(
        pool_items[stage1_positions],
        stage1_candidates,
    )

    stage1_feature[stage1_positions] = stage1_scores

    is_stage1_candidate = np.zeros(n_items, dtype=np.float32)
    is_stage1_candidate[stage1_positions] = 1.0

    title_map = source_map(
        title_idx[row],
        title_scores[row],
        TITLE_TOP_K,
    )

    location_map = source_map(
        loc_idx[row],
        loc_scores[row],
        LOCATION_TOP_K,
        valid_nonnegative=True,
    )

    title_score = np.full(n_items, np.nan, dtype=np.float32)
    title_rank = np.full(n_items, TITLE_TOP_K + 1, dtype=np.float32)

    location_score = np.full(n_items, np.nan, dtype=np.float32)
    location_rank = np.full(n_items, LOCATION_TOP_K + 1, dtype=np.float32)

    for item, (score, rank) in title_map.items():
        position = item_to_position.get(item)

        if position is not None:
            title_score[position] = score
            title_rank[position] = rank

    for item, (score, rank) in location_map.items():
        position = item_to_position.get(item)

        if position is not None:
            location_score[position] = score
            location_rank[position] = rank

    title_present = np.isfinite(title_score).astype(np.float32)
    location_present = np.isfinite(location_score).astype(np.float32)

    title_mean, title_std = z_params(
        [score for score, _ in title_map.values()]
    )

    location_mean, location_std = z_params(
        [score for score, _ in location_map.values()]
    )

    title_z = np.where(
        np.isfinite(title_score),
        (title_score - title_mean) / title_std,
        np.nan,
    ).astype(np.float32)

    location_z = np.where(
        np.isfinite(location_score),
        (location_score - location_mean) / location_std,
        np.nan,
    ).astype(np.float32)

    query_location = str(
        validation.iloc[row]["search_location_id"]
    )

    item_locations = (
        items.iloc[pool_items]["item_location_id"]
        .fillna("__NA__")
        .astype(str)
        .to_numpy()
    )

    same_location = (
            item_locations == query_location
    ).astype(np.float32)

    query_embedding = np.asarray(
        qemb[row],
        dtype=np.float32,
    )

    query_title_cosine = np.asarray(
        title_emb[pool_items] @ query_embedding,
        dtype=np.float32,
    )

    query_item_cosine = np.asarray(
        item_emb[pool_items] @ query_embedding,
        dtype=np.float32,
    )

    overlap, char_similarity = lexical_features(
        validation.iloc[row]["normalized_query"],
        items.iloc[pool_items]["item_title_raw"]
        .fillna("")
        .astype(str)
        .tolist(),
    )

    original_source_count = np.zeros(n_items, dtype=np.float32)

    for source_indices in original_indices.values():
        source_items = set(
            map(
                int,
                np.asarray(source_indices[row, :500]),
            )
        )

        original_source_count += np.fromiter(
            (
                float(int(item) in source_items)
                for item in pool_items
            ),
            dtype=np.float32,
            count=n_items,
        )

    num_sources = (
            original_source_count
            + title_present
            + location_present
    )

    new_source_rrf = (
            np.where(
                title_present > 0,
                1.0 / (60.0 + title_rank),
                0.0,
            )
            + np.where(
        location_present > 0,
        1.0 / (60.0 + location_rank),
        0.0,
    )
    )

    features = np.column_stack(
        [
            stage1_feature,
            np.isnan(stage1_feature).astype(np.float32),
            is_stage1_candidate,
            title_score,
            title_rank,
            title_present,
            title_z,
            location_score,
            location_rank,
            location_present,
            location_z,
            same_location,
            query_title_cosine,
            query_item_cosine,
            overlap,
            char_similarity,
            num_sources,
            new_source_rrf.astype(np.float32),
        ]
    ).astype(np.float32)

    assert features.shape == (
        n_items,
        len(FINAL_FEATURE_NAMES),
    )

    assert not np.isinf(features).any()

    return features


def predict_stage1(
        row,
        builder,
        namespace,
        model,
):
    row = int(row)

    candidates, context_matrix = builder.build(row)

    base_matrix = build_stage1_base_features(
        row,
        candidates,
        indices=namespace["indices"],
        scores=namespace["scores"],
        best_weights=namespace["best_weights"],
        validation=namespace["validation"],
        item_locations=namespace["item_locations"],
        field_builder=namespace["field_builder"],
        union_items=builder.candidates,
        union_offsets=builder.offsets,
        full_scores=builder.full,
    )

    features = np.hstack(
        [
            base_matrix,
            context_matrix,
        ]
    ).astype(np.float32, copy=False)

    scores = np.asarray(
        model.predict(
            features,
            thread_count=6,
        ),
        dtype=np.float32,
    )

    return np.asarray(candidates, dtype=np.int32), scores


def stratified_sample(
        pool_items,
        features,
        positives,
        rng,
):
    labels = np.fromiter(
        (
            float(int(item) in positives)
            for item in pool_items
        ),
        dtype=np.float32,
        count=len(pool_items),
    )

    positive_rows = np.flatnonzero(labels > 0)

    if not len(positive_rows):
        return None

    selected = set(map(int, positive_rows))

    hard_columns = (
        FINAL_FEATURE_NAMES.index("v4_score"),
        FINAL_FEATURE_NAMES.index("title_e5_score"),
        FINAL_FEATURE_NAMES.index("location_e5_score"),
    )

    for column in hard_columns:
        scores = features[:, column]

        order = np.argsort(
            -np.nan_to_num(
                scores,
                nan=-np.inf,
            ),
            kind="stable",
        )

        added = 0

        for position in order:
            position = int(position)

            if (
                    labels[position] > 0
                    or position in selected
                    or not np.isfinite(scores[position])
            ):
                continue

            selected.add(position)
            added += 1

            if added >= N_HARD_PER_SOURCE:
                break

    remaining = np.array(
        [
            position
            for position in range(len(pool_items))
            if labels[position] == 0 and position not in selected
        ],
        dtype=np.int32,
    )

    if len(remaining):
        n_random = min(N_RANDOM, len(remaining))

        selected.update(
            map(
                int,
                rng.choice(
                    remaining,
                    size=n_random,
                    replace=False,
                ),
            )
        )

    selected = np.array(sorted(selected), dtype=np.int32)

    return selected, labels[selected]


def build_sampled_dataset(
        rows,
        seed,
        *,
        builder,
        namespace,
        model,
        positives,
        validation,
        items,
        title_idx,
        title_scores,
        loc_idx,
        loc_scores,
        query_embeddings,
        item_embeddings,
        title_embeddings,
        original_indices,
):
    rng = np.random.default_rng(seed)

    feature_parts = []
    label_parts = []
    group_parts = []

    group_id = 0

    for row in rows:
        stage1_candidates, stage1_scores = predict_stage1(
            row,
            builder,
            namespace,
            model,
        )

        pool = build_expanded_pool(
            row,
            stage1_candidates,
            title_idx,
            loc_idx,
        )

        features = build_final_features(
            row,
            pool,
            stage1_candidates,
            stage1_scores,
            validation=validation,
            items=items,
            title_idx=title_idx,
            title_scores=title_scores,
            loc_idx=loc_idx,
            loc_scores=loc_scores,
            qemb=query_embeddings,
            item_emb=item_embeddings,
            title_emb=title_embeddings,
            original_indices=original_indices,
        )

        sampled = stratified_sample(
            pool,
            features,
            positives[int(row)],
            rng,
        )

        if sampled is None:
            continue

        selected, labels = sampled

        feature_parts.append(features[selected])
        label_parts.append(labels)

        group_parts.append(
            np.full(
                len(selected),
                group_id,
                dtype=np.int32,
            )
        )

        group_id += 1

    assert feature_parts
    assert group_id > 0

    return (
        np.vstack(feature_parts),
        np.concatenate(label_parts),
        np.concatenate(group_parts),
    )


def main():
    required = [
        STAGE1_DIR / "catboost_v4_seed42.cbm",
        STAGE1_DIR / "features_complete.json",
        TITLE_RETRIEVAL_DIR / "title_e5_indices.npy",
        TITLE_RETRIEVAL_DIR / "title_e5_scores.npy",
        TITLE_RETRIEVAL_DIR / "title_e5_item_embeddings.npy",
        LOCATION_RETRIEVAL_DIR / "location_e5_indices.npy",
        LOCATION_RETRIEVAL_DIR / "location_e5_scores.npy",
        BASE_FEATURES_DIR / "audited_split.npz",
        BASE_FEATURES_DIR / "audited_positives.joblib",
        BASE_FEATURES_DIR / "training_rows.npz",
        BASE_FEATURES_DIR / "e5_q0_query_embeddings.npy",
        ITEM_EMBEDDINGS_PATH,
    ]

    missing = [str(path) for path in required if not path.exists()]

    assert not missing, (
            "Missing artifacts:\n"
            + "\n".join(missing)
    )

    builder, namespace, _ = build_stage1_feature_builder()

    validation = namespace["validation"]

    items = pd.read_parquet(
        ROOT / "data/benchmark_items.parquet"
    )

    namespace["field_builder"] = load_field_builder(
        items,
        validation,
    )

    positives = joblib.load(
        BASE_FEATURES_DIR / "audited_positives.joblib"
    )

    split = np.load(
        BASE_FEATURES_DIR / "audited_split.npz"
    )

    train_rows = split["train"]
    dev_rows = split["dev"]

    title_idx = np.load(
        TITLE_RETRIEVAL_DIR / "title_e5_indices.npy",
        mmap_mode="r",
    )

    title_scores = np.load(
        TITLE_RETRIEVAL_DIR / "title_e5_scores.npy",
        mmap_mode="r",
    )

    loc_idx = np.load(
        LOCATION_RETRIEVAL_DIR / "location_e5_indices.npy",
        mmap_mode="r",
    )

    loc_scores = np.load(
        LOCATION_RETRIEVAL_DIR / "location_e5_scores.npy",
        mmap_mode="r",
    )

    query_embeddings = np.load(
        BASE_FEATURES_DIR / "e5_q0_query_embeddings.npy",
        mmap_mode="r",
    )

    item_embeddings = np.load(
        ITEM_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    title_embeddings = np.load(
        TITLE_RETRIEVAL_DIR / "title_e5_item_embeddings.npy",
        mmap_mode="r",
    )

    original_indices = {
        source: namespace["indices"][source]
        for source in SOURCE_NAMES
    }

    stage1_model = CatBoostRanker()

    stage1_model.load_model(
        str(STAGE1_DIR / "catboost_v4_seed42.cbm")
    )

    manifest = json.loads(
        (STAGE1_DIR / "features_complete.json").read_text()
    )

    assert stage1_model.feature_names_ == (
            manifest["v3_features"]
            + manifest["features"]
    )

    with np.load(
            BASE_FEATURES_DIR / "training_rows.npz"
    ) as archive:
        sampled_rows = archive["validation_row"]

    query_codes = pd.factorize(
        validation.search_query.map(clean_text)
    )[0]

    sampled_query_codes = query_codes[sampled_rows]

    rng = np.random.default_rng(2024)

    unique_codes = np.unique(sampled_query_codes)

    holdout_codes = set(
        rng.choice(
            unique_codes,
            size=int(0.08 * len(unique_codes)),
            replace=False,
        ).tolist()
    )

    stage2_rows = np.array(
        [
            int(row)
            for row in train_rows
            if int(query_codes[int(row)]) in holdout_codes
        ],
        dtype=np.int32,
    )

    assert len(stage2_rows) > 0
    assert not np.intersect1d(stage2_rows, dev_rows).size

    unique_stage2_codes = np.unique(
        query_codes[stage2_rows]
    )

    rng = np.random.default_rng(505)

    validation_codes = set(
        rng.choice(
            unique_stage2_codes,
            size=max(
                1,
                int(0.15 * len(unique_stage2_codes)),
            ),
            replace=False,
        ).tolist()
    )

    validation_mask = np.isin(
        query_codes[stage2_rows],
        list(validation_codes),
    )

    fit_rows = stage2_rows[~validation_mask]
    early_stop_rows = stage2_rows[validation_mask]

    X_train, y_train, groups_train = build_sampled_dataset(
        fit_rows,
        42,
        builder=builder,
        namespace=namespace,
        model=stage1_model,
        positives=positives,
        validation=validation,
        items=items,
        title_idx=title_idx,
        title_scores=title_scores,
        loc_idx=loc_idx,
        loc_scores=loc_scores,
        query_embeddings=query_embeddings,
        item_embeddings=item_embeddings,
        title_embeddings=title_embeddings,
        original_indices=original_indices,
    )

    X_valid, y_valid, groups_valid = build_sampled_dataset(
        early_stop_rows,
        43,
        builder=builder,
        namespace=namespace,
        model=stage1_model,
        positives=positives,
        validation=validation,
        items=items,
        title_idx=title_idx,
        title_scores=title_scores,
        loc_idx=loc_idx,
        loc_scores=loc_scores,
        query_embeddings=query_embeddings,
        item_embeddings=item_embeddings,
        title_embeddings=title_embeddings,
        original_indices=original_indices,
    )

    train_pool = Pool(
        X_train,
        label=y_train,
        group_id=groups_train,
        feature_names=FINAL_FEATURE_NAMES,
    )

    valid_pool = Pool(
        X_valid,
        label=y_valid,
        group_id=groups_valid,
        feature_names=FINAL_FEATURE_NAMES,
    )

    model = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=50",
        iterations=3000,
        depth=8,
        learning_rate=0.06,
        l2_leaf_reg=5.0,
        random_seed=SEED,
        thread_count=6,
        verbose=False,
        allow_writing_files=False,
        early_stopping_rounds=200,
        use_best_model=True,
    )

    model.fit(
        train_pool,
        eval_set=valid_pool,
    )

    model.save_model(
        str(FINAL_RANKER_DIR / "catboost_v5_seed42.cbm")
    )

    del X_train, X_valid, train_pool, valid_pool
    gc.collect()

    ranked = []
    ceilings = []

    for row in dev_rows:
        stage1_candidates, stage1_scores = predict_stage1(
            row,
            builder,
            namespace,
            stage1_model,
        )

        pool = build_expanded_pool(
            row,
            stage1_candidates,
            title_idx,
            loc_idx,
        )

        features = build_final_features(
            row,
            pool,
            stage1_candidates,
            stage1_scores,
            validation=validation,
            items=items,
            title_idx=title_idx,
            title_scores=title_scores,
            loc_idx=loc_idx,
            loc_scores=loc_scores,
            qemb=query_embeddings,
            item_emb=item_embeddings,
            title_emb=title_embeddings,
            original_indices=original_indices,
        )

        scores = np.asarray(
            model.predict(
                features,
                thread_count=6,
            )
        )

        top_items = pool[
            np.argsort(
                -scores,
                kind="stable",
            )[:50]
        ]

        ranked.append(top_items)

        positive_items = positives[int(row)]

        ceilings.append(
            len(set(map(int, pool)) & positive_items)
            / len(positive_items)
        )

    metrics = macro_metrics(
        ranked,
        positives,
        dev_rows,
    )

    metrics.update(
        {
            "candidate_ceiling_macro": float(np.mean(ceilings)),
            "v4_reference_macro": 0.722491,
            "delta_vs_v4_reference": metrics["macro_recall"] - 0.722491,
            "v5_train_contexts": int(len(fit_rows)),
            "v5_early_stop_contexts": int(len(early_stop_rows)),
            "best_iteration": int(model.get_best_iteration()),
        }
    )

    (FINAL_RANKER_DIR / "metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    pd.DataFrame(
        {
            "validation_row": dev_rows,
            "top50_item_idx": [
                items.tolist()
                for items in ranked
            ],
        }
    ).to_parquet(
        FINAL_RANKER_DIR / "dev_top50.parquet",
        index=False,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=6):
        main()
