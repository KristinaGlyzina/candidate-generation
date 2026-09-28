from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data/train.parquet"
ITEMS_PATH = ROOT / "data/benchmark_items.parquet"

E5_ITEM_EMBEDDINGS_PATH = ROOT / "outputs/e5_small_item_embeddings.npy"

OUTPUT_DIR = ROOT / "outputs/hybrid_retrieval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

E5_MODEL_NAME = "intfloat/multilingual-e5-small"

TOP_K = 2000
DENSE_BATCH_SIZE = 16
SPARSE_BATCH_SIZE = 16
E5_ENCODE_BATCH_SIZE = 64

WORD_MAX_FEATURES = 300_000
CHAR_MAX_FEATURES = 300_000

RRF_K = 60.0

RRF_WEIGHTS = {
    "word": 1.00,
    "char": 0.75,
    "e5_q0": 1.00,
    "e5_q1": 0.25,
}

EVAL_KS = [50, 100, 500, 1000, 2000]

SPACE_RE = re.compile(r"\s+")


def clean_text(x) -> str:
    if x is None or pd.isna(x):
        return ""

    text = str(x)
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = SPACE_RE.sub(" ", text).strip()

    return text


def make_query_q0(row) -> str:
    return clean_text(row["search_query"])


def make_query_q1(row) -> str:
    query = clean_text(row["search_query"])
    params = clean_text(row["search_infm_params_text"])

    if not params:
        return query

    return f"{query}. {params}"


def make_word_item_text(row) -> str:
    title = clean_text(row["item_title_raw"])
    params = clean_text(row["item_infm_params_text"])
    description = clean_text(row["item_description_raw"])

    return f"{title} {title} {title} {params} {description}"


def make_char_item_text(row) -> str:
    title = clean_text(row["item_title_raw"])
    params = clean_text(row["item_infm_params_text"])

    return f"{title} {params}"


def make_word_query_text(row) -> str:
    query = clean_text(row["search_query"])
    params = clean_text(row["search_infm_params_text"])

    if not params:
        return query

    return f"{query} {params}"


def make_char_query_text(row) -> str:
    return clean_text(row["search_query"])


def topk_from_dense_scores(
        scores: np.ndarray,
        k: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[1])

    partition = np.argpartition(
        scores,
        kth=scores.shape[1] - k,
        axis=1,
    )[:, -k:]

    values = np.take_along_axis(scores, partition, axis=1)
    order = np.argsort(-values, axis=1)

    indices = np.take_along_axis(partition, order, axis=1)
    values = np.take_along_axis(values, order, axis=1)

    return (
        indices.astype(np.int32, copy=False),
        values.astype(np.float32, copy=False),
    )


def sparse_retrieve(
        query_matrix: sparse.csr_matrix,
        item_matrix: sparse.csr_matrix,
        top_k: int,
        batch_size: int,
        name: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_queries = query_matrix.shape[0]

    all_indices = np.empty((n_queries, top_k), dtype=np.int32)
    all_scores = np.empty((n_queries, top_k), dtype=np.float32)

    item_t = item_matrix.T.tocsc()

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)

        batch_scores = (
                query_matrix[start:end] @ item_t
        ).toarray().astype(np.float32, copy=False)

        indices, scores = topk_from_dense_scores(
            batch_scores,
            top_k,
        )

        all_indices[start:end] = indices
        all_scores[start:end] = scores

        del batch_scores, indices, scores

    return all_indices, all_scores


def dense_retrieve(
        query_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        top_k: int,
        batch_size: int,
        name: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_queries = query_embeddings.shape[0]

    all_indices = np.empty((n_queries, top_k), dtype=np.int32)
    all_scores = np.empty((n_queries, top_k), dtype=np.float32)

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)

        scores = query_embeddings[start:end] @ item_embeddings.T
        scores = np.asarray(scores, dtype=np.float32)

        indices, values = topk_from_dense_scores(
            scores,
            top_k,
        )

        all_indices[start:end] = indices
        all_scores[start:end] = values

        del scores, indices, values

    return all_indices, all_scores


def build_validation(
        train: pd.DataFrame,
        item_ids: np.ndarray,
) -> tuple[pd.DataFrame, list[set[str]]]:
    corpus_ids = set(item_ids.tolist())

    validation = train[
        train["item_id"].isin(corpus_ids)
    ].copy()

    validation["search_query_clean"] = validation["search_query"].map(clean_text)
    validation["search_params_clean"] = validation["search_infm_params_text"].map(clean_text)

    context_columns = [
        "search_query_clean",
        "search_location_id",
        "search_is_delivery_search",
        "search_params_clean",
        "search_category",
    ]

    grouped_rows = []

    for _, group in validation.groupby(
            context_columns,
            dropna=False,
            sort=False,
    ):
        positives = set(
            group["item_id"].astype(str).tolist()
        )

        grouped_rows.append(
            {
                "search_query": group["search_query"].iloc[0],
                "search_location_id": group["search_location_id"].iloc[0],
                "search_is_delivery_search": group["search_is_delivery_search"].iloc[0],
                "search_infm_params_text": group["search_infm_params_text"].iloc[0],
                "search_category": group["search_category"].iloc[0],
                "normalized_query": group["search_query_clean"].iloc[0],
                "positives": positives,
            }
        )

    queries = pd.DataFrame(grouped_rows)
    positive_sets = queries.pop("positives").tolist()

    return queries, positive_sets


def evaluate_ranking(
        ranking_indices: np.ndarray,
        positive_sets: list[set[str]],
        item_ids: np.ndarray,
        ks: list[int],
        name: str,
) -> list[dict]:
    rows = []

    for k in ks:
        macro_recalls = []
        query_hits = []

        pair_hits = 0
        total_pairs = 0

        for query_index, positives in enumerate(positive_sets):
            retrieved = set(
                item_ids[
                    ranking_indices[
                    query_index,
                    :k,
                    ]
                ].tolist()
            )

            hits = len(
                positives.intersection(retrieved)
            )

            macro_recalls.append(
                hits / len(positives)
            )

            query_hits.append(
                1.0 if hits > 0 else 0.0
            )

            pair_hits += hits
            total_pairs += len(positives)

        rows.append(
            {
                "retriever": name,
                "k": k,
                "macro_recall": float(np.mean(macro_recalls)),
                "query_hit": float(np.mean(query_hits)),
                "pair_hit": pair_hits / total_pairs,
                "positive_hits": pair_hits,
                "positive_total": total_pairs,
            }
        )

    return rows


def positive_hit_sets(
        ranking_indices: np.ndarray,
        positive_sets: list[set[str]],
        item_ids: np.ndarray,
        k: int,
) -> set[tuple[int, str]]:
    hits = set()

    for query_index, positives in enumerate(positive_sets):
        retrieved = set(
            item_ids[
                ranking_indices[
                query_index,
                :k,
                ]
            ].tolist()
        )

        for item_id in positives:
            if item_id in retrieved:
                hits.add((query_index, item_id))

    return hits


def evaluate_union(
        rankings: dict[str, np.ndarray],
        positive_sets: list[set[str]],
        item_ids: np.ndarray,
        source_k: int,
        name: str,
) -> dict:
    macro_recalls = []
    query_hits = []

    pair_hits = 0
    total_pairs = 0

    union_sizes = []

    for query_index, positives in enumerate(positive_sets):
        union = set()

        for ranking in rankings.values():
            union.update(
                item_ids[
                    ranking[
                    query_index,
                    :source_k,
                    ]
                ].tolist()
            )

        hits = len(
            positives.intersection(union)
        )

        macro_recalls.append(
            hits / len(positives)
        )

        query_hits.append(
            1.0 if hits else 0.0
        )

        pair_hits += hits
        total_pairs += len(positives)

        union_sizes.append(len(union))

    return {
        "retriever": name,
        "source_k": source_k,
        "mean_union_size": float(np.mean(union_sizes)),
        "macro_recall": float(np.mean(macro_recalls)),
        "query_hit": float(np.mean(query_hits)),
        "pair_hit": pair_hits / total_pairs,
    }


def rrf_fusion(
        rankings: dict[str, np.ndarray],
        weights: dict[str, float],
        output_k: int,
        rrf_k: float = 60.0,
) -> np.ndarray:
    names = list(rankings.keys())
    n_queries = rankings[names[0]].shape[0]

    result = np.empty(
        (n_queries, output_k),
        dtype=np.int32,
    )

    for query_index in range(n_queries):
        scores: dict[int, float] = {}

        for name, ranking in rankings.items():
            weight = weights[name]

            for rank_zero, item_index in enumerate(ranking[query_index]):
                rank = rank_zero + 1
                item_index = int(item_index)

                scores[item_index] = (
                        scores.get(item_index, 0.0)
                        + weight / (rrf_k + rank)
                )

        best = sorted(
            scores.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )[:output_k]

        result[query_index] = np.fromiter(
            (
                item_index
                for item_index, _ in best
            ),
            dtype=np.int32,
            count=output_k,
        )

    return result


def save_ranking(
        name: str,
        indices: np.ndarray,
        scores: np.ndarray | None = None,
) -> None:
    np.save(
        OUTPUT_DIR / f"{name}_indices.npy",
        indices,
    )

    if scores is not None:
        np.save(
            OUTPUT_DIR / f"{name}_scores.npy",
            scores,
        )


def main():
    train = pd.read_parquet(TRAIN_PATH)
    items = pd.read_parquet(ITEMS_PATH)

    train["item_id"] = train["item_id"].astype(str)
    items["item_id"] = items["item_id"].astype(str)

    item_ids = items["item_id"].to_numpy()

    queries, positive_sets = build_validation(
        train,
        item_ids,
    )

    query_word = queries.apply(
        make_word_query_text,
        axis=1,
    ).tolist()

    query_char = queries.apply(
        make_char_query_text,
        axis=1,
    ).tolist()

    query_q0 = queries.apply(
        make_query_q0,
        axis=1,
    ).tolist()

    query_q1 = queries.apply(
        make_query_q1,
        axis=1,
    ).tolist()

    metrics = []
    rankings: dict[str, np.ndarray] = {}

    word_item_texts = items.apply(
        make_word_item_text,
        axis=1,
    ).tolist()

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=WORD_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
        norm="l2",
    )

    word_items = word_vectorizer.fit_transform(
        word_item_texts
    ).tocsr()

    word_queries = word_vectorizer.transform(
        query_word
    ).tocsr()

    word_indices, word_scores = sparse_retrieve(
        word_queries,
        word_items,
        TOP_K,
        SPARSE_BATCH_SIZE,
        "word",
    )

    rankings["word"] = word_indices

    save_ranking(
        "word",
        word_indices,
        word_scores,
    )

    metrics.extend(
        evaluate_ranking(
            word_indices,
            positive_sets,
            item_ids,
            EVAL_KS,
            "word",
        )
    )

    del word_items, word_queries, word_vectorizer, word_item_texts
    gc.collect()

    char_item_texts = items.apply(
        make_char_item_text,
        axis=1,
    ).tolist()

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=CHAR_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
        norm="l2",
    )

    char_items = char_vectorizer.fit_transform(
        char_item_texts
    ).tocsr()

    char_queries = char_vectorizer.transform(
        query_char
    ).tocsr()

    char_indices, char_scores = sparse_retrieve(
        char_queries,
        char_items,
        TOP_K,
        SPARSE_BATCH_SIZE,
        "char",
    )

    rankings["char"] = char_indices

    save_ranking(
        "char",
        char_indices,
        char_scores,
    )

    metrics.extend(
        evaluate_ranking(
            char_indices,
            positive_sets,
            item_ids,
            EVAL_KS,
            "char",
        )
    )

    del char_items, char_queries, char_vectorizer, char_item_texts
    gc.collect()

    item_embeddings = np.load(
        E5_ITEM_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    if item_embeddings.shape[0] != len(items):
        raise ValueError(
            "E5 embedding row count does not match benchmark_items row count. "
            "Item alignment may be wrong."
        )

    model = SentenceTransformer(
        E5_MODEL_NAME
    )

    e5_q0_texts = [
        f"query: {text}"
        for text in query_q0
    ]

    e5_q1_texts = [
        f"query: {text}"
        for text in query_q1
    ]

    q0_embeddings = model.encode(
        e5_q0_texts,
        batch_size=E5_ENCODE_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    q1_embeddings = model.encode(
        e5_q1_texts,
        batch_size=E5_ENCODE_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    e5_q0_indices, e5_q0_scores = dense_retrieve(
        q0_embeddings,
        item_embeddings,
        TOP_K,
        DENSE_BATCH_SIZE,
        "e5_q0",
    )

    rankings["e5_q0"] = e5_q0_indices

    save_ranking(
        "e5_q0",
        e5_q0_indices,
        e5_q0_scores,
    )

    metrics.extend(
        evaluate_ranking(
            e5_q0_indices,
            positive_sets,
            item_ids,
            EVAL_KS,
            "e5_q0",
        )
    )

    e5_q1_indices, e5_q1_scores = dense_retrieve(
        q1_embeddings,
        item_embeddings,
        TOP_K,
        DENSE_BATCH_SIZE,
        "e5_q1",
    )

    rankings["e5_q1"] = e5_q1_indices

    save_ranking(
        "e5_q1",
        e5_q1_indices,
        e5_q1_scores,
    )

    metrics.extend(
        evaluate_ranking(
            e5_q1_indices,
            positive_sets,
            item_ids,
            EVAL_KS,
            "e5_q1",
        )
    )

    del model, q0_embeddings, q1_embeddings
    gc.collect()

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(
        OUTPUT_DIR / "source_metrics.csv",
        index=False,
    )

    hit_sets = {
        name: positive_hit_sets(
            ranking,
            positive_sets,
            item_ids,
            50,
        )
        for name, ranking in rankings.items()
    }

    unique_rows = []
    all_names = list(hit_sets)

    for name in all_names:
        other_hits = set()

        for other_name in all_names:
            if other_name != name:
                other_hits |= hit_sets[other_name]

        unique = hit_sets[name] - other_hits

        unique_rows.append(
            {
                "retriever": name,
                "hits_at_50": len(hit_sets[name]),
                "unique_hits_at_50": len(unique),
            }
        )

    pd.DataFrame(unique_rows).to_csv(
        OUTPUT_DIR / "unique_hits_at_50.csv",
        index=False,
    )

    union_rows = []

    union_configs = {
        "word+char": {
            "word": rankings["word"],
            "char": rankings["char"],
        },
        "word+e5_q0": {
            "word": rankings["word"],
            "e5_q0": rankings["e5_q0"],
        },
        "word+char+e5_q0": {
            "word": rankings["word"],
            "char": rankings["char"],
            "e5_q0": rankings["e5_q0"],
        },
        "all_sources": rankings,
    }

    for source_k in EVAL_KS:
        for name, subset in union_configs.items():
            union_rows.append(
                evaluate_union(
                    subset,
                    positive_sets,
                    item_ids,
                    source_k,
                    name,
                )
            )

    pd.DataFrame(union_rows).to_csv(
        OUTPUT_DIR / "union_metrics.csv",
        index=False,
    )

    rrf_indices = rrf_fusion(
        rankings=rankings,
        weights=RRF_WEIGHTS,
        output_k=TOP_K,
        rrf_k=RRF_K,
    )

    save_ranking(
        "rrf",
        rrf_indices,
    )

    rrf_metrics = evaluate_ranking(
        rrf_indices,
        positive_sets,
        item_ids,
        EVAL_KS,
        "weighted_rrf",
    )

    pd.DataFrame(rrf_metrics).to_csv(
        OUTPUT_DIR / "rrf_metrics.csv",
        index=False,
    )

    queries.to_parquet(
        OUTPUT_DIR / "validation_queries.parquet",
        index=False,
    )


if __name__ == "__main__":
    main()
