import hashlib
import json
import os
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.tmp")

    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())

    temporary.replace(path)


def read_json(path, default=None):
    path = Path(path)

    if not path.exists():
        return default

    return json.loads(path.read_text())


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.tmp")

    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())

    temporary.replace(path)


def validate_npy(path, shape, dtype, mmap_mode="r"):
    path = Path(path)
    expected_shape = tuple(map(int, shape))
    expected_dtype = np.dtype(dtype)

    array = np.load(
        path,
        mmap_mode=mmap_mode,
        allow_pickle=False,
    )

    assert array.shape == expected_shape, (
        path,
        array.shape,
        expected_shape,
    )

    assert array.dtype == expected_dtype, (
        path,
        array.dtype,
        expected_dtype,
    )

    expected_size = array.offset + array.nbytes

    assert path.stat().st_size == expected_size, (
        path,
        path.stat().st_size,
        expected_size,
    )

    return array


def check_manifest(path, paths, root):
    path = Path(path)

    state = {
        str(Path(item).relative_to(root)): file_sha256(item)
        for item in paths
    }

    previous = read_json(path)

    if previous is None:
        atomic_json(path, state)
    else:
        assert previous == state, (
            f"Inputs changed: refusing stale cache {path}"
        )

    return state
