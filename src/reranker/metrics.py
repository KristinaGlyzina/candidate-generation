import numpy as np


def macro_metrics(top_indices, positives, k=50):
    recalls = []
    query_hits = []
    total_hits = 0
    total_positives = 0

    for ranked, pos in zip(top_indices, positives):
        ranked = ranked[:k]

        if not pos:
            continue

        hits = len(set(map(int, ranked)) & pos)

        recalls.append(hits / len(pos))
        query_hits.append(float(hits > 0))

        total_hits += hits
        total_positives += len(pos)

    return {
        "macro_recall": float(np.mean(recalls)),
        "query_hit": float(np.mean(query_hits)),
        "pair_hit": (
            float(total_hits / total_positives)
            if total_positives
            else 0.0
        ),
    }
