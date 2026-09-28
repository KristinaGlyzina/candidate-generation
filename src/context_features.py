import re

import numpy as np
from scipy.stats import rankdata

SOURCES = ("word", "char", "e5_q0", "e5_q1")

RRF_WEIGHTS = {
    "word": 1.0,
    "char": 0.75,
    "e5_q0": 1.0,
    "e5_q1": 0.25,
}

RRF_K = 60
TOP_K = 50
CONSENSUS_K = 30

HUB_SOURCES = ("word", "e5_q0")

EPS = 1e-6

TOKEN_RE = re.compile(r"(?u)\b\w+\b")

ITEM_STATIC_NAMES = (
    "title_chars",
    "title_tokens",
    "title_digits",
    "params_chars",
    "params_tokens",
    "desc_chars",
    "desc_tokens",
)


def feature_names(with_category):
    names = []

    for source in SOURCES:
        names.extend(
            f"{source}_{suffix}"
            for suffix in ("urank", "z", "gap_max", "gap_top50", "pct_max")
        )

    names.extend((
        "union_rrf",
        "union_rrf_rank",
        "z_mean",
        "z_min",
        "z_max",
        "z_std",
        "urank_mean",
        "urank_min",
        "n_src_top50",
        "n_src_top200",
        "q_n_tokens",
        "q_n_chars",
        "q_has_params",
        "q_param_tokens",
        "q_is_delivery",
        "q_has_digit",
        "q_union_size",
    ))

    names.extend(f"q_max_{source}" for source in SOURCES)

    names.extend((
        "q_margin_e5_q0",
        "q_margin_word",
        "loc_share_top30",
        "q_loc_rate_top30",
        "q_loc_rate_union",
    ))

    if with_category:
        names.extend((
            "cat_share_top30",
            "cat_equals_query",
        ))

    names.extend((
        "e5_top10_mean_sim",
        "e5_top30_mean_sim",
    ))

    names.extend(f"item_{name}" for name in ITEM_STATIC_NAMES)

    names.extend((
        "q_over_title_tokens",
        "hub_count_log",
        "hub_mean_word",
        "hub_mean_e5_q0",
        "word_csls",
        "e5_q0_csls",
    ))

    assert len(names) == len(set(names))

    return names


def feature_group(name):
    if name.startswith("hub_") or name.endswith("_csls"):
        return "hub"

    if name.startswith(("loc_", "q_loc_")):
        return "location"

    if name.startswith("cat_"):
        return "category"

    if name.startswith("e5_top"):
        return "e5_neighbors"

    if name.startswith("item_") or name == "q_over_title_tokens":
        return "item_static"

    if name.startswith("q_"):
        return "query"

    if name.startswith(("union_", "z_", "urank_", "n_src")):
        return "fusion"

    return "relative"


def item_static_matrix(title, params, desc):
    def chars(series):
        return series.str.len().to_numpy(np.float32)

    def tokens(series):
        return series.str.count(r"\w+").to_numpy(np.float32)

    return np.column_stack((
        chars(title),
        tokens(title),
        title.str.count(r"\d").to_numpy(np.float32),
        chars(params),
        tokens(params),
        chars(desc),
        tokens(desc),
    )).astype(np.float32)


def make_query_arrays(
        query_text,
        params_text,
        is_delivery,
        loc_codes,
        cat_codes,
        group_codes,
):
    tokens = [TOKEN_RE.findall(text) for text in query_text]

    return {
        "tokens": np.array([len(set(values)) for values in tokens], dtype=np.float32),
        "chars": np.array([len(text) for text in query_text], dtype=np.float32),
        "has_params": np.array([float(bool(text)) for text in params_text], dtype=np.float32),
        "param_tokens": np.array(
            [len(set(TOKEN_RE.findall(text))) for text in params_text],
            dtype=np.float32,
        ),
        "delivery": np.asarray(is_delivery, dtype=np.float32),
        "has_digit": np.array(
            [float(any(char.isdigit() for char in text)) for text in query_text],
            dtype=np.float32,
        ),
        "loc": np.asarray(loc_codes, dtype=np.int32),
        "cat": None if cat_codes is None else np.asarray(cat_codes, dtype=np.int32),
        "group": np.asarray(group_codes, dtype=np.int64),
    }


def hub_stats(candidates, full, n_items, chunk=4_000_000):
    count = np.zeros(n_items, dtype=np.int64)

    sums = {
        source: np.zeros(n_items, dtype=np.float64)
        for source in HUB_SOURCES
    }

    for start in range(0, len(candidates), chunk):
        end = start + chunk
        items = np.asarray(candidates[start:end])

        count += np.bincount(items, minlength=n_items)

        for source in HUB_SOURCES:
            sums[source] += np.bincount(
                items,
                weights=np.asarray(full[source][start:end], dtype=np.float64),
                minlength=n_items,
            )

    return count, sums


def share_in_top(codes, top, weights=None):
    weights = (
        np.ones(len(top), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )

    unique, inverse = np.unique(codes[top], return_inverse=True)

    mass = np.bincount(inverse, weights=weights, minlength=len(unique))

    positions = np.minimum(
        np.searchsorted(unique, codes),
        len(unique) - 1,
    )

    return np.where(
        unique[positions] == codes,
        mass[positions] / weights.sum(),
        0.0,
    ).astype(np.float32)


def leave_self_out_mean_sim(embeddings, top):
    total = embeddings[top].sum(axis=0)

    member = np.zeros(len(embeddings), dtype=np.float32)
    member[top] = 1.0

    self_dot = member * np.einsum("ij,ij->i", embeddings, embeddings)

    return (
            (embeddings @ total - self_dot)
            / (len(top) - member)
    ).astype(np.float32)


class ContextFeatureBuilder:
    def __init__(
            self,
            *,
            candidates,
            offsets,
            full,
            embeddings,
            item_static,
            item_loc,
            item_cat,
            q,
            group_rows,
            hub_count,
            hub_sums,
    ):
        self.candidates = candidates
        self.offsets = offsets
        self.full = full

        self.E = embeddings
        self.item_static = item_static
        self.item_loc = item_loc
        self.item_cat = item_cat

        self.q = q
        self.group_rows = group_rows

        self.hub_count = hub_count
        self.hub_sums = hub_sums

        self.names = feature_names(item_cat is not None)

    def _union(self, row):
        lo, hi = self.offsets[row: row + 2]

        return lo, hi, np.asarray(self.candidates[lo:hi])

    def _hub_block(self, row, union, scores):
        n = len(union)

        sibling_items = []
        sibling_scores = {source: [] for source in HUB_SOURCES}

        for sibling in self.group_rows[self.q["group"][row]]:
            lo, hi, items = self._union(sibling)

            sibling_items.append(items)

            for source in HUB_SOURCES:
                sibling_scores[source].append(
                    np.asarray(self.full[source][lo:hi], dtype=np.float64)
                )

        sibling_items = np.concatenate(sibling_items)

        positions = np.minimum(
            np.searchsorted(union, sibling_items),
            n - 1,
        )

        valid = union[positions] == sibling_items

        remaining = (
                self.hub_count[union]
                - np.bincount(positions[valid], minlength=n)
        )

        columns = {
            "hub_count_log": np.log1p(
                np.maximum(remaining, 0)
            ).astype(np.float32)
        }

        for source in HUB_SOURCES:
            removed = np.bincount(
                positions[valid],
                weights=np.concatenate(sibling_scores[source])[valid],
                minlength=n,
            )

            mean = np.where(
                remaining > 0,
                (self.hub_sums[source][union] - removed)
                / np.maximum(remaining, 1),
                np.nan,
            )

            columns[f"hub_mean_{source}"] = mean.astype(np.float32)

            columns[f"{source}_csls"] = (
                    scores[source] - mean
            ).astype(np.float32)

        return columns

    def build(self, row):
        lo, hi, union = self._union(row)

        n = len(union)

        assert n > TOP_K

        scores = {
            source: np.asarray(self.full[source][lo:hi], dtype=np.float32)
            for source in SOURCES
        }

        columns = {}
        ranks = {}

        for source in SOURCES:
            values = scores[source]

            ranks[source] = rankdata(-values, method="average")

            maximum = values.max()

            top50_threshold = np.partition(
                values,
                n - TOP_K,
            )[n - TOP_K]

            columns[f"{source}_urank"] = ranks[source]

            columns[f"{source}_z"] = (
                                             values - values.mean()
                                     ) / (
                                             values.std() + EPS
                                     )

            columns[f"{source}_gap_max"] = values - maximum

            columns[f"{source}_gap_top50"] = values - top50_threshold

            columns[f"{source}_pct_max"] = (
                values / maximum
                if maximum > 1e-9
                else np.zeros(n, dtype=np.float32)
            )

        rank_matrix = np.stack([ranks[source] for source in SOURCES])

        z_matrix = np.stack([
            columns[f"{source}_z"]
            for source in SOURCES
        ])

        rrf = sum(
            RRF_WEIGHTS[source] / (RRF_K + ranks[source])
            for source in SOURCES
        )

        columns.update({
            "union_rrf": rrf,
            "union_rrf_rank": rankdata(-rrf, method="average"),
            "z_mean": z_matrix.mean(axis=0),
            "z_min": z_matrix.min(axis=0),
            "z_max": z_matrix.max(axis=0),
            "z_std": z_matrix.std(axis=0),
            "urank_mean": rank_matrix.mean(axis=0),
            "urank_min": rank_matrix.min(axis=0),
            "n_src_top50": (rank_matrix <= 50).sum(axis=0),
            "n_src_top200": (rank_matrix <= 200).sum(axis=0),
        })

        order = np.argsort(-rrf, kind="stable")

        top30 = order[:CONSENSUS_K]

        query = self.q
        ones = np.ones(n, dtype=np.float32)

        title_tokens = self.item_static[union, 1]

        columns.update({
            "q_n_tokens": query["tokens"][row] * ones,
            "q_n_chars": query["chars"][row] * ones,
            "q_has_params": query["has_params"][row] * ones,
            "q_param_tokens": query["param_tokens"][row] * ones,
            "q_is_delivery": query["delivery"][row] * ones,
            "q_has_digit": query["has_digit"][row] * ones,
            "q_union_size": float(n) * ones,
        })

        for source in SOURCES:
            columns[f"q_max_{source}"] = scores[source].max() * ones

        for key, source in (
                ("e5_q0", "e5_q0"),
                ("word", "word"),
        ):
            top_two = np.partition(
                scores[source],
                n - 2,
            )[n - 2:]

            columns[f"q_margin_{key}"] = (
                                                 top_two.max() - top_two.min()
                                         ) * ones

        locations = self.item_loc[union]
        query_location = query["loc"][row]

        columns["loc_share_top30"] = share_in_top(
            locations,
            top30,
        )

        columns["q_loc_rate_top30"] = (
                float(
                    query_location >= 0
                    and np.mean(locations[top30] == query_location)
                )
                * ones
        )

        columns["q_loc_rate_union"] = (
                float(
                    query_location >= 0
                    and np.mean(locations == query_location)
                )
                * ones
        )

        if self.item_cat is not None:
            categories = self.item_cat[union]

            columns["cat_share_top30"] = share_in_top(
                categories,
                top30,
                rrf[top30],
            )

            columns["cat_equals_query"] = (
                    categories == query["cat"][row]
            ).astype(np.float32)

        embeddings = np.asarray(
            self.E[union],
            dtype=np.float32,
        )

        columns["e5_top10_mean_sim"] = leave_self_out_mean_sim(
            embeddings,
            order[:10],
        )

        columns["e5_top30_mean_sim"] = leave_self_out_mean_sim(
            embeddings,
            top30,
        )

        static = self.item_static[union]

        for index, name in enumerate(ITEM_STATIC_NAMES):
            columns[f"item_{name}"] = static[:, index]

        columns["q_over_title_tokens"] = (
                query["tokens"][row]
                / np.maximum(title_tokens, 1.0)
        )

        columns.update(
            self._hub_block(
                row,
                union,
                scores,
            )
        )

        assert set(columns) == set(self.names)

        features = np.column_stack([
            np.asarray(columns[name], dtype=np.float32)
            for name in self.names
        ])

        assert features.shape == (n, len(self.names))
        assert not np.isinf(features).any()

        return union, features.astype(np.float32, copy=False)
