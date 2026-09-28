SOURCE_NAMES = ("word", "char", "e5_q0", "e5_q1")

SOURCE_TOP_K = 500
FINAL_K = 50
RRF_K = 60
MISSING_RANK = SOURCE_TOP_K + 1

RANDOM_STATE = 42
N_HARD_NEGATIVES = 150
N_RANDOM_NEGATIVES = 50

EXPECTED_POSITIVE_PAIRS = 29_571

BASE_FEATURE_NAMES = tuple(
    feature
    for source in SOURCE_NAMES
    for feature in (
        f"{source}_score",
        f"{source}_rank",
        f"{source}_reciprocal_rank",
    )
) + (
    "num_sources",
    "best_rank",
    "rrf_score",
    "same_location",
)

assert len(BASE_FEATURE_NAMES) == 16
