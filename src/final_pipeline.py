from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "src" else Path.cwd()
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import context_features
import retrieval as hr

from reranker import model as reranker
from reranker.data import load_data, load_field_builder
from reranker.features import build_stage1_base_features

ITEMS_PATH = ROOT / "data/benchmark_items.parquet"
QUERIES_PATH = ROOT / "data/benchmark_queries.parquet"
ITEM_EMBEDDINGS_PATH = ROOT / "outputs/e5_small_item_embeddings.npy"
TITLE_EMBEDDINGS_PATH = ROOT / "outputs/title_e5_ablation/title_e5_item_embeddings.npy"
STAGE1_MODEL_PATH = ROOT / "outputs/reranker_v4_context/catboost_v4_seed42.cbm"
STAGE1_MANIFEST_PATH = ROOT / "outputs/reranker_v4_context/features_complete.json"
STAGE1_HUB_STATS_PATH = ROOT / "outputs/reranker_v4_context/hub_stats.npz"
FINAL_RANKER_MODEL_PATH = ROOT / "outputs/reranker_v5_light/catboost_v5_seed42.cbm"

OUTPUT_DIR = ROOT / "outputs/final_v5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = ("word", "char", "e5_q0", "e5_q1")

RETRIEVAL_TOP_K = 500
FINAL_K = 50
RETRIEVAL_BATCH_SIZE = 16


def topk(scores, k=RETRIEVAL_TOP_K):
    return hr.topk_from_dense_scores(np.asarray(scores, dtype=np.float32), k)


def save_or_load(name):
    index_path = OUTPUT_DIR / f"{name}_indices.npy"
    score_path = OUTPUT_DIR / f"{name}_scores.npy"

    if not (index_path.exists() and score_path.exists()):
        return None

    return np.load(index_path, mmap_mode="r"), np.load(score_path, mmap_mode="r")


def lexical_texts(items, queries, source):
    if source == "word":
        item_texts = items.apply(hr.make_word_item_text, axis=1).tolist()
        query_texts = queries.apply(hr.make_word_query_text, axis=1).tolist()

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=hr.WORD_MAX_FEATURES,
            sublinear_tf=True,
            dtype=np.float32,
            norm="l2",
        )
    else:
        item_texts = items.apply(hr.make_char_item_text, axis=1).tolist()
        query_texts = queries.apply(hr.make_char_query_text, axis=1).tolist()

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=hr.CHAR_MAX_FEATURES,
            sublinear_tf=True,
            dtype=np.float32,
            norm="l2",
        )

    item_matrix = vectorizer.fit_transform(item_texts).tocsr()
    query_matrix = vectorizer.transform(query_texts).tocsr()

    return item_matrix, query_matrix


def retrieve_sparse(item_matrix, query_matrix, name):
    cached = save_or_load(name)

    if cached is not None:
        return cached

    n_queries = query_matrix.shape[0]

    indices = np.empty((n_queries, RETRIEVAL_TOP_K), dtype=np.int32)
    scores = np.empty((n_queries, RETRIEVAL_TOP_K), dtype=np.float32)

    item_matrix_t = item_matrix.T.tocsc()

    for start in range(0, n_queries, RETRIEVAL_BATCH_SIZE):
        end = min(start + RETRIEVAL_BATCH_SIZE, n_queries)

        block = (query_matrix[start:end] @ item_matrix_t).toarray().astype(
            np.float32, copy=False
        )

        batch_indices, batch_scores = topk(block)

        indices[start:end] = batch_indices
        scores[start:end] = batch_scores

    index_path = OUTPUT_DIR / f"{name}_indices.npy"
    score_path = OUTPUT_DIR / f"{name}_scores.npy"

    np.save(index_path, indices)
    np.save(score_path, scores)

    return np.load(index_path, mmap_mode="r"), np.load(score_path, mmap_mode="r")


def retrieve_dense(query_embeddings, item_embeddings, name):
    cached = save_or_load(name)

    if cached is not None:
        return cached

    n_queries = query_embeddings.shape[0]

    indices = np.empty((n_queries, RETRIEVAL_TOP_K), dtype=np.int32)
    scores = np.empty((n_queries, RETRIEVAL_TOP_K), dtype=np.float32)

    for start in range(0, n_queries, RETRIEVAL_BATCH_SIZE):
        end = min(start + RETRIEVAL_BATCH_SIZE, n_queries)

        batch_indices, batch_scores = topk(
            query_embeddings[start:end] @ item_embeddings.T
        )

        indices[start:end] = batch_indices
        scores[start:end] = batch_scores

    index_path = OUTPUT_DIR / f"{name}_indices.npy"
    score_path = OUTPUT_DIR / f"{name}_scores.npy"

    np.save(index_path, indices)
    np.save(score_path, scores)

    return np.load(index_path, mmap_mode="r"), np.load(score_path, mmap_mode="r")


def build_union(indices, n_queries):
    offsets_path = OUTPUT_DIR / "union_offsets.npy"
    items_path = OUTPUT_DIR / "union_items.npy"

    if offsets_path.exists() and items_path.exists():
        return np.load(items_path, mmap_mode="r"), np.load(offsets_path)

    offsets = np.zeros(n_queries + 1, dtype=np.int64)
    parts = []

    for query_idx in range(n_queries):
        union = np.unique(
            np.concatenate(
                [
                    np.asarray(
                        indices[source][query_idx, :RETRIEVAL_TOP_K],
                        dtype=np.int32,
                    )
                    for source in SOURCES
                ]
            )
        ).astype(np.int32)

        parts.append(union)
        offsets[query_idx + 1] = offsets[query_idx] + len(union)

    candidates = np.concatenate(parts)

    np.save(offsets_path, offsets)
    np.save(items_path, candidates)

    return np.load(items_path, mmap_mode="r"), offsets


def full_scores_from_matrices(
        name,
        query_matrix,
        item_matrix,
        candidates,
        offsets,
        dense=False,
):
    path = OUTPUT_DIR / f"{name}_full_score.npy"

    if path.exists():
        scores = np.load(path, mmap_mode="r")
        assert scores.shape == (len(candidates),)
        return scores

    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(candidates),),
    )

    item_matrix_t = item_matrix.T if dense else item_matrix.T.tocsc()
    n_queries = query_matrix.shape[0]

    for start in range(0, n_queries, RETRIEVAL_BATCH_SIZE):
        end = min(start + RETRIEVAL_BATCH_SIZE, n_queries)

        block = query_matrix[start:end] @ item_matrix_t

        if not dense:
            block = block.toarray()

        block = np.asarray(block, dtype=np.float32)

        for query_idx in range(start, end):
            lo, hi = offsets[query_idx: query_idx + 2]
            query_candidates = np.asarray(candidates[lo:hi])

            output[lo:hi] = block[
                query_idx - start,
                query_candidates,
            ]

    output.flush()
    del output

    return np.load(path, mmap_mode="r")


def title_retrieval(query_embeddings, title_embeddings):
    cached = save_or_load("title_e5")

    if cached is not None:
        return cached

    return retrieve_dense(query_embeddings, title_embeddings, "title_e5")


def location_retrieval(
        queries,
        query_embeddings,
        item_embeddings,
        items,
):
    name = "location_e5"
    cached = save_or_load(name)

    if cached is not None:
        return cached

    indices = np.full(
        (len(queries), RETRIEVAL_TOP_K),
        -1,
        dtype=np.int32,
    )

    scores = np.full(
        (len(queries), RETRIEVAL_TOP_K),
        -np.inf,
        dtype=np.float32,
    )

    item_locations = (
        items.item_location_id.fillna("__NA__").astype(str).to_numpy()
    )

    location_groups = {}

    for item_idx, location in enumerate(item_locations):
        location_groups.setdefault(location, []).append(item_idx)

    location_groups = {
        location: np.asarray(item_indices, dtype=np.int32)
        for location, item_indices in location_groups.items()
    }

    query_locations = (
        queries.search_location_id.fillna("__NA__").astype(str).to_numpy()
    )

    for query_idx in range(len(queries)):
        candidates = location_groups.get(query_locations[query_idx])

        if candidates is not None and len(candidates):
            candidate_scores = np.asarray(
                item_embeddings[candidates] @ query_embeddings[query_idx],
                dtype=np.float32,
            )

            k = min(RETRIEVAL_TOP_K, len(candidates))

            if k == len(candidates):
                order = np.argsort(-candidate_scores)[:k]
            else:
                partition = np.argpartition(-candidate_scores, k - 1)[:k]
                order = partition[np.argsort(-candidate_scores[partition])]

            indices[query_idx, :k] = candidates[order]
            scores[query_idx, :k] = candidate_scores[order]

    index_path = OUTPUT_DIR / f"{name}_indices.npy"
    score_path = OUTPUT_DIR / f"{name}_scores.npy"

    np.save(index_path, indices)
    np.save(score_path, scores)

    return np.load(index_path, mmap_mode="r"), np.load(score_path, mmap_mode="r")


class BenchmarkContextBuilder(context_features.ContextFeatureBuilder):
    def _hub_block(self, row, U, S):
        remaining = self.hub_count[U]

        columns = {
            "hub_count_log": np.log1p(np.maximum(remaining, 0)).astype(np.float32)
        }

        for source in context_features.HUB_SOURCES:
            mean = np.where(
                remaining > 0,
                self.hub_sums[source][U] / np.maximum(remaining, 1),
                np.nan,
            )

            columns[f"hub_mean_{source}"] = mean.astype(np.float32)
            columns[f"{source}_csls"] = (S[source] - mean).astype(np.float32)

        return columns


def make_stage1_builder(
        items,
        queries,
        candidates,
        offsets,
        full_scores,
        item_embeddings,
):
    item_location, query_location = reranker.factorize_pair(
        items.item_location_id,
        queries.search_location_id,
    )

    category_column = reranker.detect_category_column(items)

    item_category = None
    query_category = None

    if category_column is not None:
        item_category, query_category = reranker.factorize_pair(
            items[category_column],
            queries.search_category,
        )

    query_group = pd.factorize(
        queries.search_query.map(reranker.clean_text)
    )[0]

    query_arrays = context_features.make_query_arrays(
        queries.normalized_query.tolist(),
        queries.normalized_params.tolist(),
        queries.search_is_delivery_search.astype(np.float32).to_numpy(),
        query_location,
        query_category,
        query_group,
    )

    item_static = context_features.item_static_matrix(
        items.item_title_raw.fillna("").astype(str),
        items.item_infm_params_text.fillna("").astype(str),
        items.item_description_raw.fillna("").astype(str),
    )

    with np.load(STAGE1_HUB_STATS_PATH) as hub_stats:
        hub_count = hub_stats["count"]

        hub_sums = {
            source: hub_stats[f"sum_{source}"]
            for source in context_features.HUB_SOURCES
        }

    return BenchmarkContextBuilder(
        candidates=candidates,
        offsets=offsets,
        full=full_scores,
        embeddings=item_embeddings,
        item_static=item_static,
        item_loc=item_location,
        item_cat=item_category,
        q=query_arrays,
        group_rows={},
        hub_count=hub_count,
        hub_sums=hub_sums,
    )


def load_query_embeddings(queries):
    q0_path = OUTPUT_DIR / "e5_q0_query_embeddings.npy"
    q1_path = OUTPUT_DIR / "e5_q1_query_embeddings.npy"

    if q0_path.exists() and q1_path.exists():
        return (
            np.load(q0_path, mmap_mode="r"),
            np.load(q1_path, mmap_mode="r"),
        )

    encoder = SentenceTransformer(
        hr.E5_MODEL_NAME,
        local_files_only=True,
    )

    q0 = encoder.encode(
        ["query: " + hr.clean_text(query) for query in queries.search_query],
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    q1 = encoder.encode(
        ["query: " + hr.make_query_q1(row) for _, row in queries.iterrows()],
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    np.save(q0_path, q0)
    np.save(q1_path, q1)

    del encoder, q0, q1
    gc.collect()

    return (
        np.load(q0_path, mmap_mode="r"),
        np.load(q1_path, mmap_mode="r"),
    )


def validate_required_files():
    required = [
        ITEMS_PATH,
        QUERIES_PATH,
        ITEM_EMBEDDINGS_PATH,
        TITLE_EMBEDDINGS_PATH,
        STAGE1_MODEL_PATH,
        STAGE1_MANIFEST_PATH,
        STAGE1_HUB_STATS_PATH,
        FINAL_RANKER_MODEL_PATH,
    ]

    missing = [str(path) for path in required if not path.exists()]

    assert not missing, "Missing:\n" + "\n".join(missing)


def build_original_retrieval(items, queries):
    indices = {}
    scores = {}

    for source in ("word", "char"):
        cached = save_or_load(source)

        if cached is not None:
            indices[source], scores[source] = cached
            continue

        item_matrix, query_matrix = lexical_texts(
            items,
            queries,
            source,
        )

        indices[source], scores[source] = retrieve_sparse(
            item_matrix,
            query_matrix,
            source,
        )

        del item_matrix, query_matrix
        gc.collect()

    return indices, scores


def build_full_scores(
        items,
        queries,
        candidates,
        offsets,
        q0,
        q1,
        item_embeddings,
):
    full_scores = {}

    for source in ("word", "char"):
        path = OUTPUT_DIR / f"{source}_full_score.npy"

        if path.exists():
            full_scores[source] = np.load(path, mmap_mode="r")
            continue

        item_matrix, query_matrix = lexical_texts(
            items,
            queries,
            source,
        )

        full_scores[source] = full_scores_from_matrices(
            source,
            query_matrix,
            item_matrix,
            candidates,
            offsets,
        )

        del item_matrix, query_matrix
        gc.collect()

    full_scores["e5_q0"] = full_scores_from_matrices(
        "e5_q0",
        q0,
        item_embeddings,
        candidates,
        offsets,
        dense=True,
    )

    full_scores["e5_q1"] = full_scores_from_matrices(
        "e5_q1",
        q1,
        item_embeddings,
        candidates,
        offsets,
        dense=True,
    )

    return full_scores


def save_results(rows, queries, items):
    result = pd.DataFrame(
        rows,
        columns=["query_id", "item_id", "rank"],
    )

    query_ids = queries.query_id.astype(str)

    assert len(result) == len(queries) * FINAL_K
    assert result.groupby("query_id").size().eq(FINAL_K).all()
    assert not result.duplicated(["query_id", "item_id"]).any()
    assert set(result.query_id) == set(query_ids)
    assert set(result.item_id) <= set(items.item_id.astype(str))
    assert result[["query_id", "item_id"]].notna().all().all()

    long_path = OUTPUT_DIR / "top50_long.csv"
    wide_path = OUTPUT_DIR / "top50_wide.csv"
    answer_path = ROOT / "answer.csv"

    result.to_csv(long_path, index=False)

    wide = (
        result
        .pivot(
            index="query_id",
            columns="rank",
            values="item_id",
        )
        .reindex(query_ids)
        .reset_index()
    )

    wide.columns = ["query_id"] + [
        f"item_{rank}"
        for rank in range(1, FINAL_K + 1)
    ]

    wide.to_csv(wide_path, index=False)

    answer = (
        result
        .sort_values(
            ["query_id", "rank"],
            kind="stable",
        )
        .groupby(
            "query_id",
            sort=False,
        )["item_id"]
        .agg(lambda values: " ".join(map(str, values)))
        .reindex(query_ids)
        .reset_index(name="answer")
    )

    assert list(answer.columns) == ["query_id", "answer"]
    assert len(answer) == len(queries)
    assert answer.query_id.is_unique
    assert answer.query_id.tolist() == query_ids.tolist()
    assert answer.answer.notna().all()
    assert answer.answer.str.split().map(len).eq(FINAL_K).all()

    answer.to_csv(
        answer_path,
        index=False,
        encoding="utf-8",
    )

    return long_path, wide_path, answer_path


def main():
    validate_required_files()

    items = pd.read_parquet(ITEMS_PATH)
    queries = pd.read_parquet(QUERIES_PATH).reset_index(drop=True)

    assert len(items) == 189212
    assert len(queries) == 2452

    queries["normalized_query"] = queries.search_query.map(reranker.clean_text)
    queries["normalized_params"] = queries.search_infm_params_text.map(
        reranker.clean_text
    )

    indices, scores = build_original_retrieval(
        items,
        queries,
    )

    item_embeddings = np.load(
        ITEM_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    title_embeddings = np.load(
        TITLE_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    q0, q1 = load_query_embeddings(queries)

    indices["e5_q0"], scores["e5_q0"] = retrieve_dense(
        q0,
        item_embeddings,
        "e5_q0",
    )

    indices["e5_q1"], scores["e5_q1"] = retrieve_dense(
        q1,
        item_embeddings,
        "e5_q1",
    )

    candidates, offsets = build_union(
        indices,
        len(queries),
    )

    full_scores = build_full_scores(
        items,
        queries,
        candidates,
        offsets,
        q0,
        q1,
        item_embeddings,
    )

    title_indices, title_scores = title_retrieval(
        q0,
        title_embeddings,
    )

    location_indices, location_scores = location_retrieval(
        queries,
        q0,
        item_embeddings,
        items,
    )

    namespace, _, _ = load_data()

    namespace.update(
        validation=queries,
        indices=indices,
        scores=scores,
        item_locations=items.item_location_id.to_numpy(),
    )

    namespace["field_builder"] = load_field_builder(items, queries)

    stage1_builder = make_stage1_builder(
        items,
        queries,
        candidates,
        offsets,
        full_scores,
        item_embeddings,
    )

    stage1_model = CatBoostRanker()
    stage1_model.load_model(str(STAGE1_MODEL_PATH))

    manifest = json.loads(STAGE1_MANIFEST_PATH.read_text())

    assert stage1_model.feature_names_ == (manifest["v3_features"] + manifest["features"])

    final_model = CatBoostRanker()
    final_model.load_model(str(FINAL_RANKER_MODEL_PATH))

    assert final_model.feature_names_ == reranker.FINAL_FEATURE_NAMES

    rows = []

    for row in range(len(queries)):
        stage1_candidates, context_feature_matrix = stage1_builder.build(row)

        base_features = build_stage1_base_features(
            row,
            stage1_candidates,
            indices=namespace["indices"],
            scores=namespace["scores"],
            best_weights=namespace["best_weights"],
            validation=namespace["validation"],
            item_locations=namespace["item_locations"],
            field_builder=namespace["field_builder"],
            union_items=candidates,
            union_offsets=offsets,
            full_scores=full_scores,
        )

        stage1_features = np.hstack(
            [
                base_features,
                context_feature_matrix,
            ]
        ).astype(np.float32)

        stage1_scores = np.asarray(
            stage1_model.predict(
                stage1_features,
                thread_count=6,
            ),
            dtype=np.float32,
        )

        pool = reranker.build_expanded_pool(
            row,
            stage1_candidates,
            title_indices,
            location_indices,
        )

        final_features = reranker.build_final_features(
            row,
            pool,
            stage1_candidates,
            stage1_scores,
            validation=queries,
            items=items,
            title_idx=title_indices,
            title_scores=title_scores,
            loc_idx=location_indices,
            loc_scores=location_scores,
            qemb=q0,
            item_emb=item_embeddings,
            title_emb=title_embeddings,
            original_indices=indices,
        )

        predictions = np.asarray(
            final_model.predict(
                final_features,
                thread_count=6,
            )
        )

        top_items = pool[
            np.argsort(
                -predictions,
                kind="stable",
            )[:FINAL_K]
        ]

        query_id = str(queries.iloc[row].query_id)

        rows.extend(
            (
                query_id,
                str(items.iloc[int(item_idx)].item_id),
                rank,
            )
            for rank, item_idx in enumerate(
                top_items,
                1,
            )
        )

    save_results(rows, queries, items)


if __name__ == "__main__":
    with threadpool_limits(limits=6):
        main()
