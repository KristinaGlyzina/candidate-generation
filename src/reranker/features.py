import numpy as np

from .config import (
    BASE_FEATURE_NAMES,
    MISSING_RANK,
    RRF_K,
    SOURCE_NAMES,
    SOURCE_TOP_K,
)


def build_base_features(
        row_idx,
        candidate_items,
        *,
        indices,
        scores,
        best_weights,
        validation,
        item_locations,
):
    candidate_items = np.asarray(candidate_items, dtype=np.int32).reshape(-1)
    n = len(candidate_items)

    candidate_ids: list[int] = candidate_items.tolist()
    candidate_to_pos = {
        item_idx: position
        for position, item_idx in enumerate(candidate_ids)
    }

    source_score_features = {}
    source_rank_features = {}

    num_sources = np.zeros(n, dtype=np.float32)
    best_rank = np.full(n, MISSING_RANK, dtype=np.float32)
    rrf_score = np.zeros(n, dtype=np.float32)

    for source in SOURCE_NAMES:
        ranks = np.full(n, MISSING_RANK, dtype=np.float32)
        source_scores = np.zeros(n, dtype=np.float32)

        source_items = indices[source][row_idx, :SOURCE_TOP_K]
        source_values = scores[source][row_idx, :SOURCE_TOP_K]

        weight = best_weights[source]

        for rank0, (item_idx, score) in enumerate(zip(source_items, source_values)):
            item_id = int(item_idx)
            position = candidate_to_pos.get(item_id)

            if position is None:
                continue

            rank = rank0 + 1

            ranks[position] = rank
            source_scores[position] = float(score)

            num_sources[position] += 1.0

            if rank < best_rank[position]:
                best_rank[position] = rank

            rrf_score[position] += weight / (RRF_K + rank)

        source_score_features[source] = source_scores
        source_rank_features[source] = ranks

    columns = []

    for source in SOURCE_NAMES:
        ranks = source_rank_features[source]

        reciprocal_rank = np.where(
            ranks <= SOURCE_TOP_K,
            1.0 / ranks,
            0.0,
        ).astype(np.float32)

        columns.extend(
            (
                source_score_features[source],
                ranks,
                reciprocal_rank,
            )
        )

    columns.extend(
        (
            num_sources,
            best_rank,
            rrf_score,
        )
    )

    if "search_location_id" in validation.columns:
        query_location = validation.iloc[row_idx]["search_location_id"]

        item_locations_array = np.asarray(item_locations)
        selected_locations = np.asarray(item_locations_array[candidate_items])

        same_location = np.asarray(
            selected_locations == query_location,
            dtype=np.float32,
        )
    else:
        same_location = np.zeros(n, dtype=np.float32)

    columns.append(same_location)

    features = np.column_stack(columns).astype(np.float32, copy=False)

    assert features.shape == (n, len(BASE_FEATURE_NAMES))
    assert features.shape[1] == 16

    return features


def build_text_enriched_features(
        row_idx,
        candidate_items,
        *,
        indices,
        scores,
        best_weights,
        validation,
        item_locations,
        field_builder,
):
    base_features = build_base_features(
        row_idx,
        candidate_items,
        indices=indices,
        scores=scores,
        best_weights=best_weights,
        validation=validation,
        item_locations=item_locations,
    )

    assert base_features.shape[1] == 16

    field_features = field_builder.build(
        q_idx=row_idx,
        p_idx=row_idx,
        cand=candidate_items,
    )

    assert field_features.shape[0] == base_features.shape[0]
    assert field_features.shape[1] == 24

    features = np.column_stack(
        [
            base_features,
            field_features,
        ]
    ).astype(np.float32, copy=False)

    assert features.shape[1] == 40

    return features


FULL_SCORE_SOURCES = (
    "word",
    "char",
    "e5_q0",
    "e5_q1",
)

FULL_SCORE_FEATURE_NAMES = tuple(
    f"{source}_full_score"
    for source in FULL_SCORE_SOURCES
)


def build_stage1_base_features(
        row_idx,
        candidate_items,
        *,
        indices,
        scores,
        best_weights,
        validation,
        item_locations,
        field_builder,
        union_items,
        union_offsets,
        full_scores,
):
    candidate_items = np.asarray(candidate_items, dtype=np.int32).reshape(-1)

    lo, hi = union_offsets[row_idx: row_idx + 2]
    union = union_items[lo:hi]

    positions = np.searchsorted(
        union,
        candidate_items,
    )

    assert np.all(positions < len(union))

    np.testing.assert_array_equal(
        union[positions],
        candidate_items,
    )

    enriched_features = build_text_enriched_features(
        row_idx,
        candidate_items,
        indices=indices,
        scores=scores,
        best_weights=best_weights,
        validation=validation,
        item_locations=item_locations,
        field_builder=field_builder,
    )

    assert enriched_features.dtype == np.float32
    assert enriched_features.shape == (len(candidate_items), 40)

    full_score_features = np.column_stack(
        [
            full_scores[source][lo + positions]
            for source in FULL_SCORE_SOURCES
        ]
    )

    assert np.isfinite(full_score_features).all()

    features = np.column_stack(
        [
            enriched_features,
            full_score_features,
        ]
    ).astype(np.float32, copy=False)

    np.testing.assert_array_equal(
        features[:, :40],
        enriched_features,
    )

    assert features.shape == (len(candidate_items), 44)
    assert not np.isinf(features).any()

    return features
