import re
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN_PATTERN = r"(?u)\b\w+\b"
TOKEN_RE = re.compile(TOKEN_PATTERN)

ITEM_FIELDS = ("title", "params", "desc")

DEFAULT_VECTORIZER_CONFIG = {
    "char_ngram_range": (3, 5),
    "word": {
        "title": {"min_df": 1, "max_features": None},
        "params": {"min_df": 1, "max_features": None},
        "desc": {"min_df": 2, "max_features": 300_000},
    },
    "char": {
        "title": {"min_df": 2, "max_features": 500_000},
        "params": {"min_df": 2, "max_features": 500_000},
        "desc": {"min_df": 3, "max_features": 300_000},
    },
}

DEFAULT_MEMORY_WARN_MB = 6000


def clean_field(x, max_chars=None):
    if x is None or x is np.nan or (isinstance(x, float) and x != x):
        return ""
    x = str(x)
    if max_chars:
        x = x[:max_chars]
    x = x.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", x).strip()


class _QueryBlock:
    def __init__(self, texts, word_vec, char_vec, idf):
        self.texts = list(texts)
        self.word = word_vec.transform(self.texts).tocsr()
        self.char = (
            char_vec.transform(self.texts).tocsr()
            if char_vec is not None
            else None
        )

        binary = self.word.copy()
        binary.data = np.ones_like(binary.data)
        self.binary = binary.tocsr()
        self.idf_binary = (
                self.binary @ sparse.diags(idf.astype(np.float32))
        ).tocsr()

        self.n_tokens = np.array(
            [len(set(TOKEN_RE.findall(t))) for t in self.texts],
            dtype=np.float32,
        )
        self.idf_total = np.asarray(
            self.idf_binary.sum(axis=1)
        ).ravel().astype(np.float32)


class FieldFeatureBuilder:
    def __init__(
            self,
            item_texts,
            use_desc_char=True,
            desc_char_max_chars=300,
            cache_path=None,
            vectorizer_config=None,
            memory_warn_mb=DEFAULT_MEMORY_WARN_MB,
    ):
        self.use_desc_char = use_desc_char
        self.desc_char_max_chars = desc_char_max_chars
        self.config = vectorizer_config or DEFAULT_VECTORIZER_CONFIG
        self.memory_warn_mb = memory_warn_mb
        self.n_items = len(item_texts["title"])

        self.texts = {
            f: np.array(item_texts[f], dtype=object) for f in ITEM_FIELDS
        }
        self.nonempty = {
            f: np.array([len(t) > 0 for t in item_texts[f]], dtype=bool)
            for f in ITEM_FIELDS
        }

        self.word_vec, self.char_vec = {}, {}
        self.Mw, self.Mc, self.idf = {}, {}, {}

        if cache_path is not None and Path(cache_path).exists():
            state = joblib.load(cache_path)
            if (
                    state["n_items"] == self.n_items
                    and state["use_desc_char"] == use_desc_char
                    and state.get("desc_char_max_chars") == desc_char_max_chars
                    and state.get("config") == self.config
            ):
                print(f"Loaded field TF-IDF cache: {cache_path}")
                self.word_vec = state["word_vec"]
                self.char_vec = state["char_vec"]
                self.Mw = state["Mw"]
                self.Mc = state["Mc"]
                self.idf = state["idf"]
                self._print_sizes()
                return
            print("Field TF-IDF cache is stale, refitting.")

        self._fit(item_texts)

        if cache_path is not None:
            joblib.dump(
                {
                    "n_items": self.n_items,
                    "use_desc_char": use_desc_char,
                    "desc_char_max_chars": desc_char_max_chars,
                    "config": self.config,
                    "word_vec": self.word_vec,
                    "char_vec": self.char_vec,
                    "Mw": self.Mw,
                    "Mc": self.Mc,
                    "idf": self.idf,
                },
                cache_path,
            )
            print(f"Saved field TF-IDF cache: {cache_path}")

    def _fit(self, item_texts):
        for f in ITEM_FIELDS:
            print(f"Fitting TF-IDF for field '{f}'...")
            docs = item_texts[f]

            wcfg = self.config["word"][f]
            wv = TfidfVectorizer(
                token_pattern=TOKEN_PATTERN,
                lowercase=False,
                sublinear_tf=True,
                min_df=wcfg["min_df"],
                max_features=wcfg["max_features"],
                dtype=np.float32,
            )
            self.Mw[f] = wv.fit_transform(docs).tocsr()
            self.word_vec[f] = wv
            self.idf[f] = wv.idf_.astype(np.float32)

            if f == "desc" and not self.use_desc_char:
                self.char_vec[f] = None
                self.Mc[f] = None
                continue

            ccfg = self.config["char"][f]
            cv = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=tuple(self.config["char_ngram_range"]),
                lowercase=False,
                sublinear_tf=True,
                min_df=ccfg["min_df"],
                max_features=ccfg["max_features"],
                dtype=np.float32,
            )
            char_docs = (
                [d[: self.desc_char_max_chars] for d in docs]
                if f == "desc"
                else docs
            )
            self.Mc[f] = cv.fit_transform(char_docs).tocsr()
            self.char_vec[f] = cv

        self._print_sizes()

    @staticmethod
    def _mb(m):
        return (m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / 1024 ** 2

    def _print_sizes(self):
        total = 0.0
        print("Field TF-IDF matrices:")
        for f in ITEM_FIELDS:
            mw = self.Mw[f]
            mb_w = self._mb(mw)
            total += mb_w
            msg = (
                f"  {f:<7} word: nnz={mw.nnz:,} vocab={mw.shape[1]:,} "
                f"avg_nnz/item={mw.nnz / max(mw.shape[0], 1):.1f} "
                f"{mb_w:.0f} MB"
            )
            mc = self.Mc[f]
            if mc is not None:
                mb_c = self._mb(mc)
                total += mb_c
                msg += (
                    f" | char: nnz={mc.nnz:,} vocab={mc.shape[1]:,} "
                    f"avg_nnz/item={mc.nnz / max(mc.shape[0], 1):.1f} "
                    f"{mb_c:.0f} MB"
                )
            print(msg)

        text_mb = sum(
            sum(len(t) for t in self.texts[f]) for f in ITEM_FIELDS
        ) / 1024 ** 2
        print(
            f"  matrices total: {total:.0f} MB | "
            f"item texts (~chars): {text_mb:.0f} MB"
        )
        if total + text_mb > self.memory_warn_mb:
            print(
                f"  WARNING: field features need ~{total + text_mb:.0f} MB "
                f"(> {self.memory_warn_mb} MB). Reduce max_features / "
                f"DESC_MAX_CHARS / desc_char_max_chars, or set "
                f"use_desc_char=False."
            )

    def set_queries(self, queries, query_params):
        self.q_strings = list(queries)
        self.p_strings = list(query_params)

        self.qblock = {
            f: self._block(f, self.q_strings) for f in ITEM_FIELDS
        }
        self.pblock = self._block("params", self.p_strings)

    def _block(self, field, texts):
        return _QueryBlock(
            texts,
            self.word_vec[field],
            self.char_vec[field],  # None, если use_desc_char=False
            self.idf[field],
        )

    @property
    def feature_names(self):
        names = []
        for f in ITEM_FIELDS:
            names += [
                f"q_{f}_word_cos",
                f"q_{f}_char_cos",
                f"q_{f}_coverage",
                f"q_{f}_idf_coverage",
                f"q_{f}_phrase",
            ]
            if f == "title":
                names += ["q_title_startswith", "q_title_equals"]
            if f == "params":
                names += ["q_params_equals"]
        names += [
            "qp_params_word_cos",
            "qp_params_char_cos",
            "qp_params_coverage",
            "qp_params_idf_coverage",
            "qp_params_phrase",
            "qp_params_equals",
        ]
        return names

    def _sim_block(self, field, cand, blk, qi):
        n = len(cand)

        Mw = self.Mw[field][cand]
        qw = blk.word[qi]
        cos_w = (Mw @ qw.T).toarray().ravel()

        if self.Mc[field] is not None:
            Mc = self.Mc[field][cand]
            qc = blk.char[qi]
            cos_c = (Mc @ qc.T).toarray().ravel()
        else:
            cos_c = np.full(n, np.nan, dtype=np.float32)

        Bc = Mw.copy()
        Bc.data = np.ones_like(Bc.data)

        matched = (Bc @ blk.binary[qi].T).toarray().ravel()
        idf_matched = (Bc @ blk.idf_binary[qi].T).toarray().ravel()

        n_tok = blk.n_tokens[qi]
        idf_tot = blk.idf_total[qi]

        if n_tok > 0:
            cov = matched / n_tok
        else:
            cov = np.full(n, np.nan, dtype=np.float32)

        if idf_tot > 0:
            idf_cov = idf_matched / idf_tot
        else:
            idf_cov = np.full(n, np.nan, dtype=np.float32)

        return [
            cos_w.astype(np.float32),
            np.asarray(cos_c, dtype=np.float32),
            cov.astype(np.float32),
            idf_cov.astype(np.float32),
        ]

    @staticmethod
    def _string_flag(strings, fn):
        return np.fromiter(
            (fn(s) for s in strings),
            dtype=np.float32,
            count=len(strings),
        )

    def build(self, q_idx, p_idx, cand):
        cand = np.asarray(cand, dtype=np.int64)
        n = len(cand)
        nan = np.full(n, np.nan, dtype=np.float32)

        q = self.q_strings[q_idx]
        qp = self.p_strings[p_idx]

        cols = []

        for f in ITEM_FIELDS:
            strings = self.texts[f][cand]
            empty = ~self.nonempty[f][cand]

            if q:
                sim = self._sim_block(f, cand, self.qblock[f], q_idx)
                phrase = self._string_flag(strings, lambda s: float(q in s))
                extra = []
                if f == "title":
                    extra = [
                        self._string_flag(
                            strings, lambda s: float(s.startswith(q))
                        ),
                        self._string_flag(strings, lambda s: float(s == q)),
                    ]
                if f == "params":
                    extra = [
                        self._string_flag(strings, lambda s: float(s == q))
                    ]
                block = sim + [phrase] + extra
            else:
                k = 5 + (2 if f == "title" else 0) + (1 if f == "params" else 0)
                block = [nan.copy() for _ in range(k)]

            for col in block:
                col = col.copy()
                col[empty] = np.nan
                cols.append(col)

        strings = self.texts["params"][cand]
        empty = ~self.nonempty["params"][cand]

        if qp:
            sim = self._sim_block("params", cand, self.pblock, p_idx)
            phrase = self._string_flag(strings, lambda s: float(qp in s))
            equals = self._string_flag(strings, lambda s: float(s == qp))
            block = sim + [phrase, equals]
        else:
            block = [nan.copy() for _ in range(6)]

        for col in block:
            col = col.copy()
            col[empty] = np.nan
            cols.append(col)

        X = np.column_stack(cols).astype(np.float32, copy=False)
        assert X.shape[1] == len(self.feature_names), (
            X.shape,
            len(self.feature_names),
        )
        return X
