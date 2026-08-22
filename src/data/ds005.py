"""Canonical loader for frozen DS-005 zebrafish behavioral data.

This module provides a leakage-safe, read-only interface to the frozen
DS-005 primary dataset used by the zebrafish-behavior-ssl project.

Design principles
-----------------
1. Fish are the split unit. Bouts inherit their fish partition.
2. MetaData/lengths_data is the authoritative valid-bout mask.
3. The frozen split is never regenerated here.
4. HDF5 access is lazy; the full dataset is never loaded into memory.
5. Dataset invariants are checked against the frozen DS-005 specification.
6. QC flags are exposed separately from primary-analysis exclusion rules.

Expected repo layout
--------------------
data/
├── raw/
│   └── DS-005/
│       └── DS-005-v1/
│           └── Datasets/
│               └── JM_data/
│                   └── filtered_jmpool_kin.h5
├── metadata/
│   └── DS-005/
│       └── DS-005-fish-map.csv
└── splits/
    └── DS-005-fish-split-v1.csv

The loader does not modify source files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import csv
import hashlib

import h5py
import numpy as np


Partition = Literal["train", "validation", "test"]


# ---------------------------------------------------------------------------
# Frozen DS-005 constants
# ---------------------------------------------------------------------------

DATASET_ID = "DS-005"

EXPECTED_FISH = 463
EXPECTED_VALID_BOUTS = 1_203_409
EXPECTED_CONTEXTS = 14
EXPECTED_FRAME_RATE_HZ = 700.0
EXPECTED_TEMPORAL_SAMPLES = 175
EXPECTED_PADDED_BOUT_AXIS = 11_651

FISH_ID_PREFIX = "DS005-JM-F"
SESSION_ID_PREFIX = "DS005-JM-S"

FROZEN_SPLIT_SHA256 = (
    "19c1c7589e046337ec51b66b8fec7632029084d59905ca45b2ce751b3268c935"
)

DEFAULT_H5_RELATIVE_PATH = Path(
    "data/raw/DS-005/DS-005-v1/Datasets/JM_data/filtered_jmpool_kin.h5"
)
DEFAULT_FISH_MAP_RELATIVE_PATH = Path(
    "data/metadata/DS-005/DS-005-fish-map.csv"
)
DEFAULT_SPLIT_RELATIVE_PATH = Path(
    "data/splits/DS-005-fish-split-v1.csv"
)

REQUIRED_H5_DATASETS = {
    "bout_types",
    "converge_bouts",
    "eye_convergence",
    "eye_convergence_state",
    "head_pos",
    "orientation_smooth",
    "speed_head",
    "stims",
    "times_bouts",
}

REQUIRED_METADATA_DATASETS = {
    "errmask",
    "frameRate",
    "lengths_data",
    "t0_bout",
}

REQUIRED_FISH_MAP_COLUMNS = {
    "dataset_id",
    "cohort",
    "canonical_fish_id",
    "source_fish_index",
    "valid_bout_count",
    "context_id",
    "context_name",
}

REQUIRED_SPLIT_COLUMNS = {
    "dataset_id",
    "canonical_fish_id",
    "source_fish_index",
    "context_id",
    "context_name",
    "partition",
    "split_seed",
    "split_version",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FishRecord:
    """Canonical metadata for one fish."""

    dataset_id: str
    cohort: str
    canonical_fish_id: str
    canonical_session_id: str
    source_fish_index: int
    valid_bout_count: int
    context_id: str
    context_name: str
    partition: Partition


@dataclass(frozen=True)
class BoutKey:
    """Stable identifier for one valid bout."""

    fish_id: str
    fish_index: int
    bout_index: int
    partition: Partition
    context_id: str
    context_name: str


@dataclass(frozen=True)
class BoutQC:
    """QC indicators for a single bout.

    These flags do not automatically imply primary-analysis exclusion.
    The frozen project policy currently:
      - excludes non-finite valid bouts;
      - retains all-zero-speed-only bouts in primary analysis;
      - retains max-speed > 100 bouts in primary analysis;
      - uses the latter two as sensitivity-analysis flags.
    """

    contains_nonfinite: bool
    all_zero_speed: bool
    extreme_speed_gt_100: bool
    max_abs_speed: float

    @property
    def primary_exclude(self) -> bool:
        """Whether this bout fails the frozen primary structural QC rule."""
        return self.contains_nonfinite

    @property
    def sensitivity_flag(self) -> bool:
        """Whether this bout belongs to a frozen sensitivity-analysis set."""
        return self.all_zero_speed or self.extreme_speed_gt_100


@dataclass
class BoutData:
    """Data for one valid bout."""

    key: BoutKey
    qc: BoutQC
    head_pos: np.ndarray
    orientation_smooth: np.ndarray
    speed_head: np.ndarray
    times_bouts: np.ndarray
    bout_type: float
    stimulus_code: float
    converge_bout: Optional[np.ndarray] = None
    eye_convergence: Optional[float] = None
    eye_convergence_state: Optional[float] = None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def find_repo_root(start: Optional[Union[str, Path]] = None) -> Path:
    """Find repository root by searching upward for a ``data`` directory.

    Parameters
    ----------
    start:
        Starting directory. Defaults to current working directory.

    Returns
    -------
    pathlib.Path
        Inferred repository root.

    Raises
    ------
    FileNotFoundError
        If no suitable repository root can be identified.
    """
    current = Path(start or Path.cwd()).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "data").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate repository root. Pass repo_root explicitly."
    )


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Return SHA-256 of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _canonical_session_id(source_fish_index: int) -> str:
    return f"{SESSION_ID_PREFIX}{source_fish_index:03d}"


def _normalize_partition(value: str) -> Partition:
    value = value.strip().lower()
    if value not in {"train", "validation", "test"}:
        raise ValueError(f"Invalid partition: {value!r}")
    return value  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class DS005:
    """Lazy, leakage-safe loader for frozen DS-005.

    Parameters
    ----------
    repo_root:
        Repository root. If omitted, inferred from current working directory.
    h5_path:
        Optional override for the primary JM HDF5 path.
    fish_map_path:
        Optional override for the canonical fish-map CSV.
    split_path:
        Optional override for the frozen split CSV.
    validate:
        Run structural invariants during initialization.
    verify_split_hash:
        Verify split CSV against the frozen SHA-256. Recommended for
        confirmatory runs.

    Notes
    -----
    This object opens the HDF5 file lazily on first data access. Call
    ``close()`` when done, or use as a context manager.
    """

    def __init__(
        self,
        repo_root: Optional[Union[str, Path]] = None,
        *,
        h5_path: Optional[Union[str, Path]] = None,
        fish_map_path: Optional[Union[str, Path]] = None,
        split_path: Optional[Union[str, Path]] = None,
        validate: bool = True,
        verify_split_hash: bool = True,
    ) -> None:
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else find_repo_root()
        )

        self.h5_path = (
            Path(h5_path).resolve()
            if h5_path is not None
            else self.repo_root / DEFAULT_H5_RELATIVE_PATH
        )
        self.fish_map_path = (
            Path(fish_map_path).resolve()
            if fish_map_path is not None
            else self.repo_root / DEFAULT_FISH_MAP_RELATIVE_PATH
        )
        self.split_path = (
            Path(split_path).resolve()
            if split_path is not None
            else self.repo_root / DEFAULT_SPLIT_RELATIVE_PATH
        )

        self._h5: Optional[h5py.File] = None

        self._assert_paths_exist()

        if verify_split_hash:
            actual = sha256_file(self.split_path)
            if actual != FROZEN_SPLIT_SHA256:
                raise RuntimeError(
                    "Frozen split hash mismatch.\n"
                    f"Expected: {FROZEN_SPLIT_SHA256}\n"
                    f"Observed: {actual}\n"
                    f"File: {self.split_path}"
                )

        self._fish_records = self._load_fish_records()
        self._fish_by_id = {
            rec.canonical_fish_id: rec for rec in self._fish_records
        }
        self._fish_by_index = {
            rec.source_fish_index: rec for rec in self._fish_records
        }

        if validate:
            self.validate()

    # ------------------------------------------------------------------
    # Lifetime / file handling
    # ------------------------------------------------------------------

    def _assert_paths_exist(self) -> None:
        missing = [
            p for p in (self.h5_path, self.fish_map_path, self.split_path)
            if not p.exists()
        ]
        if missing:
            pretty = "\n".join(f"  - {p}" for p in missing)
            raise FileNotFoundError(
                "Required DS-005 files are missing:\n" + pretty
            )

    @property
    def h5(self) -> h5py.File:
        """Open and return HDF5 handle in read-only mode."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def close(self) -> None:
        """Close open HDF5 handle."""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self) -> "DS005":
        _ = self.h5
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    def _load_fish_records(self) -> List[FishRecord]:
        fish_rows = _read_csv(self.fish_map_path)
        split_rows = _read_csv(self.split_path)

        if not fish_rows:
            raise RuntimeError("Fish-map CSV is empty.")
        if not split_rows:
            raise RuntimeError("Split CSV is empty.")

        fish_cols = set(fish_rows[0].keys())
        split_cols = set(split_rows[0].keys())

        missing_fish_cols = REQUIRED_FISH_MAP_COLUMNS - fish_cols
        missing_split_cols = REQUIRED_SPLIT_COLUMNS - split_cols

        if missing_fish_cols:
            raise RuntimeError(
                f"Fish-map missing columns: {sorted(missing_fish_cols)}"
            )
        if missing_split_cols:
            raise RuntimeError(
                f"Split CSV missing columns: {sorted(missing_split_cols)}"
            )

        split_by_id: Dict[str, Dict[str, str]] = {}
        for row in split_rows:
            fish_id = row["canonical_fish_id"]
            if fish_id in split_by_id:
                raise RuntimeError(
                    f"Duplicate fish in frozen split: {fish_id}"
                )
            split_by_id[fish_id] = row

        records: List[FishRecord] = []

        for row in fish_rows:
            fish_id = row["canonical_fish_id"]

            if fish_id not in split_by_id:
                raise RuntimeError(
                    f"Fish {fish_id} missing from frozen split."
                )

            split_row = split_by_id[fish_id]

            fish_idx = int(row["source_fish_index"])
            split_idx = int(split_row["source_fish_index"])

            if fish_idx != split_idx:
                raise RuntimeError(
                    f"Index mismatch for {fish_id}: "
                    f"fish-map={fish_idx}, split={split_idx}"
                )

            if row["context_id"] != split_row["context_id"]:
                raise RuntimeError(
                    f"Context ID mismatch for {fish_id}."
                )

            if row["context_name"] != split_row["context_name"]:
                raise RuntimeError(
                    f"Context name mismatch for {fish_id}."
                )

            records.append(
                FishRecord(
                    dataset_id=row["dataset_id"],
                    cohort=row["cohort"],
                    canonical_fish_id=fish_id,
                    canonical_session_id=_canonical_session_id(fish_idx),
                    source_fish_index=fish_idx,
                    valid_bout_count=int(float(row["valid_bout_count"])),
                    context_id=row["context_id"],
                    context_name=row["context_name"],
                    partition=_normalize_partition(split_row["partition"]),
                )
            )

        records.sort(key=lambda x: x.source_fish_index)
        return records

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate DS-005 against frozen project invariants."""
        self._validate_metadata_records()
        self._validate_h5_structure()
        self._validate_lengths()
        self._validate_split()

    def _validate_metadata_records(self) -> None:
        if len(self._fish_records) != EXPECTED_FISH:
            raise RuntimeError(
                f"Expected {EXPECTED_FISH} fish, "
                f"found {len(self._fish_records)}."
            )

        ids = [r.canonical_fish_id for r in self._fish_records]
        indices = [r.source_fish_index for r in self._fish_records]

        if len(ids) != len(set(ids)):
            raise RuntimeError("Duplicate canonical fish IDs detected.")

        if sorted(indices) != list(range(EXPECTED_FISH)):
            raise RuntimeError(
                "Source fish indices are not exactly 0..462."
            )

        expected_ids = [
            f"{FISH_ID_PREFIX}{i:03d}" for i in range(EXPECTED_FISH)
        ]
        if ids != expected_ids:
            raise RuntimeError(
                "Canonical fish IDs do not match the frozen ID rule."
            )

        contexts = {r.context_name for r in self._fish_records}
        if len(contexts) != EXPECTED_CONTEXTS:
            raise RuntimeError(
                f"Expected {EXPECTED_CONTEXTS} contexts, "
                f"found {len(contexts)}."
            )

    def _validate_h5_structure(self) -> None:
        top_level = set(self.h5.keys())

        missing = REQUIRED_H5_DATASETS - top_level
        if missing:
            raise RuntimeError(
                f"HDF5 missing datasets: {sorted(missing)}"
            )

        if "MetaData" not in self.h5:
            raise RuntimeError("HDF5 missing MetaData group.")

        meta = set(self.h5["MetaData"].keys())
        missing_meta = REQUIRED_METADATA_DATASETS - meta
        if missing_meta:
            raise RuntimeError(
                f"HDF5 MetaData missing: {sorted(missing_meta)}"
            )

        frame_rate = float(self.h5["MetaData/frameRate"][0])
        if not np.isclose(frame_rate, EXPECTED_FRAME_RATE_HZ):
            raise RuntimeError(
                f"Expected frame rate {EXPECTED_FRAME_RATE_HZ}, "
                f"found {frame_rate}."
            )

        head_shape = self.h5["head_pos"].shape
        speed_shape = self.h5["speed_head"].shape
        orient_shape = self.h5["orientation_smooth"].shape
        times_shape = self.h5["times_bouts"].shape

        expected_head = (
            EXPECTED_FISH,
            EXPECTED_PADDED_BOUT_AXIS,
            EXPECTED_TEMPORAL_SAMPLES,
            2,
        )
        expected_temporal = (
            EXPECTED_FISH,
            EXPECTED_PADDED_BOUT_AXIS,
            EXPECTED_TEMPORAL_SAMPLES,
        )
        expected_times = (
            EXPECTED_FISH,
            EXPECTED_PADDED_BOUT_AXIS,
            2,
        )

        if head_shape != expected_head:
            raise RuntimeError(
                f"Unexpected head_pos shape: {head_shape}"
            )
        if speed_shape != expected_temporal:
            raise RuntimeError(
                f"Unexpected speed_head shape: {speed_shape}"
            )
        if orient_shape != expected_temporal:
            raise RuntimeError(
                f"Unexpected orientation_smooth shape: {orient_shape}"
            )
        if times_shape != expected_times:
            raise RuntimeError(
                f"Unexpected times_bouts shape: {times_shape}"
            )

    def _validate_lengths(self) -> None:
        lengths = self.lengths

        if lengths.shape != (EXPECTED_FISH,):
            raise RuntimeError(
                f"Unexpected lengths_data shape: {lengths.shape}"
            )

        if np.any(lengths < 0):
            raise RuntimeError("Negative valid-bout count detected.")

        if np.any(lengths > EXPECTED_PADDED_BOUT_AXIS):
            raise RuntimeError(
                "lengths_data exceeds padded bout-axis length."
            )

        observed_total = int(lengths.sum())
        if observed_total != EXPECTED_VALID_BOUTS:
            raise RuntimeError(
                f"Expected {EXPECTED_VALID_BOUTS} valid bouts, "
                f"found {observed_total}."
            )

        mapped_counts = np.array(
            [r.valid_bout_count for r in self._fish_records],
            dtype=np.int64,
        )
        if not np.array_equal(lengths, mapped_counts):
            bad = np.where(lengths != mapped_counts)[0][:10]
            raise RuntimeError(
                "Fish-map valid_bout_count disagrees with "
                f"MetaData/lengths_data for fish indices {bad.tolist()}."
            )

    def _validate_split(self) -> None:
        counts = self.partition_counts()

        if sum(counts.values()) != EXPECTED_FISH:
            raise RuntimeError("Partition counts do not sum to 463.")

        expected_counts = {
            "train": 323,
            "validation": 70,
            "test": 70,
        }
        if counts != expected_counts:
            raise RuntimeError(
                f"Frozen split count mismatch: {counts}"
            )

        context_parts: Dict[str, set] = {}
        for rec in self._fish_records:
            context_parts.setdefault(rec.context_name, set()).add(
                rec.partition
            )

        required_parts = {"train", "validation", "test"}
        failures = {
            c: parts
            for c, parts in context_parts.items()
            if parts != required_parts
        }
        if failures:
            raise RuntimeError(
                "Not every context appears in every partition: "
                f"{failures}"
            )

    # ------------------------------------------------------------------
    # Global dataset properties
    # ------------------------------------------------------------------

    @property
    def lengths(self) -> np.ndarray:
        """Return valid bout count for each fish as int64."""
        return np.asarray(
            self.h5["MetaData/lengths_data"][:],
            dtype=np.int64,
        )

    @property
    def frame_rate_hz(self) -> float:
        return float(self.h5["MetaData/frameRate"][0])

    @property
    def n_fish(self) -> int:
        return len(self._fish_records)

    @property
    def n_valid_bouts(self) -> int:
        return int(self.lengths.sum())

    @property
    def contexts(self) -> Tuple[str, ...]:
        return tuple(
            sorted({r.context_name for r in self._fish_records})
        )

    @property
    def fish_records(self) -> Tuple[FishRecord, ...]:
        return tuple(self._fish_records)

    def partition_counts(self) -> Dict[str, int]:
        counts = {"train": 0, "validation": 0, "test": 0}
        for rec in self._fish_records:
            counts[rec.partition] += 1
        return counts

    def summary(self) -> Dict[str, object]:
        """Return lightweight frozen-dataset summary."""
        return {
            "dataset_id": DATASET_ID,
            "n_fish": self.n_fish,
            "n_valid_bouts": self.n_valid_bouts,
            "n_contexts": len(self.contexts),
            "frame_rate_hz": self.frame_rate_hz,
            "partition_counts": self.partition_counts(),
            "split_sha256": sha256_file(self.split_path),
            "h5_path": str(self.h5_path),
            "fish_map_path": str(self.fish_map_path),
            "split_path": str(self.split_path),
        }

    # ------------------------------------------------------------------
    # Fish selection / metadata
    # ------------------------------------------------------------------

    def get_fish(
        self,
        fish: Union[str, int],
    ) -> FishRecord:
        """Return metadata for one fish by canonical ID or source index."""
        if isinstance(fish, str):
            try:
                return self._fish_by_id[fish]
            except KeyError as exc:
                raise KeyError(f"Unknown fish ID: {fish}") from exc

        try:
            return self._fish_by_index[int(fish)]
        except KeyError as exc:
            raise KeyError(f"Unknown fish index: {fish}") from exc

    def fish_in_partition(
        self,
        partition: Partition,
    ) -> Tuple[FishRecord, ...]:
        partition = _normalize_partition(partition)
        return tuple(
            r for r in self._fish_records
            if r.partition == partition
        )

    def fish_in_context(
        self,
        context_name: str,
        partition: Optional[Partition] = None,
    ) -> Tuple[FishRecord, ...]:
        rows = [
            r for r in self._fish_records
            if r.context_name == context_name
        ]

        if partition is not None:
            part = _normalize_partition(partition)
            rows = [r for r in rows if r.partition == part]

        return tuple(rows)

    # ------------------------------------------------------------------
    # Bout addressing
    # ------------------------------------------------------------------

    def _validate_bout_index(
        self,
        fish_index: int,
        bout_index: int,
    ) -> None:
        n = int(self.lengths[fish_index])

        if bout_index < 0 or bout_index >= n:
            raise IndexError(
                f"Bout {bout_index} invalid for fish {fish_index}; "
                f"valid range is 0..{n - 1}."
            )

    def bout_key(
        self,
        fish: Union[str, int],
        bout_index: int,
    ) -> BoutKey:
        rec = self.get_fish(fish)
        bout_index = int(bout_index)

        self._validate_bout_index(
            rec.source_fish_index,
            bout_index,
        )

        return BoutKey(
            fish_id=rec.canonical_fish_id,
            fish_index=rec.source_fish_index,
            bout_index=bout_index,
            partition=rec.partition,
            context_id=rec.context_id,
            context_name=rec.context_name,
        )

    def qc_for_bout(
        self,
        fish: Union[str, int],
        bout_index: int,
    ) -> BoutQC:
        """Compute frozen QC indicators for one valid bout."""
        key = self.bout_key(fish, bout_index)
        i = key.fish_index
        j = key.bout_index

        speed = np.asarray(self.h5["speed_head"][i, j])
        head = np.asarray(self.h5["head_pos"][i, j])
        orient = np.asarray(self.h5["orientation_smooth"][i, j])
        times = np.asarray(self.h5["times_bouts"][i, j])

        contains_nonfinite = not (
            np.all(np.isfinite(speed))
            and np.all(np.isfinite(head))
            and np.all(np.isfinite(orient))
            and np.all(np.isfinite(times))
        )

        max_abs_speed = float(np.max(np.abs(speed)))

        return BoutQC(
            contains_nonfinite=contains_nonfinite,
            all_zero_speed=bool(np.all(speed == 0)),
            extreme_speed_gt_100=max_abs_speed > 100.0,
            max_abs_speed=max_abs_speed,
        )

    # ------------------------------------------------------------------
    # Bout loading
    # ------------------------------------------------------------------

    def load_bout(
        self,
        fish: Union[str, int],
        bout_index: int,
        *,
        include_optional: bool = False,
        copy: bool = True,
    ) -> BoutData:
        """Load one valid bout.

        Parameters
        ----------
        fish:
            Canonical fish ID or source fish index.
        bout_index:
            Zero-based valid bout index within fish.
        include_optional:
            Include converge/eye-convergence fields.
        copy:
            Copy returned NumPy arrays. Recommended for safety.

        Returns
        -------
        BoutData
        """
        key = self.bout_key(fish, bout_index)
        i = key.fish_index
        j = key.bout_index

        def arr(name: str) -> np.ndarray:
            x = np.asarray(self.h5[name][i, j])
            return x.copy() if copy else x

        qc = self.qc_for_bout(i, j)

        converge = None
        eye = None
        eye_state = None

        if include_optional:
            converge = arr("converge_bouts")
            eye = float(self.h5["eye_convergence"][i, j])
            eye_state = float(
                self.h5["eye_convergence_state"][i, j]
            )

        return BoutData(
            key=key,
            qc=qc,
            head_pos=arr("head_pos"),
            orientation_smooth=arr("orientation_smooth"),
            speed_head=arr("speed_head"),
            times_bouts=arr("times_bouts"),
            bout_type=float(self.h5["bout_types"][i, j]),
            stimulus_code=float(self.h5["stims"][i, j]),
            converge_bout=converge,
            eye_convergence=eye,
            eye_convergence_state=eye_state,
        )

    def load_fish_field(
        self,
        fish: Union[str, int],
        field: str,
        *,
        valid_only: bool = True,
    ) -> np.ndarray:
        """Load one field for one fish.

        By default only valid bouts ``[:lengths_data[i]]`` are returned.
        """
        if field not in self.h5:
            raise KeyError(f"Unknown HDF5 field: {field}")

        rec = self.get_fish(fish)
        i = rec.source_fish_index

        data = self.h5[field]

        if data.shape[0] != EXPECTED_FISH:
            raise ValueError(
                f"Field {field!r} is not fish-indexed on axis 0."
            )

        if valid_only and data.ndim >= 2:
            n = rec.valid_bout_count
            return np.asarray(data[i, :n])

        return np.asarray(data[i])

    # ------------------------------------------------------------------
    # Iterators
    # ------------------------------------------------------------------

    def iter_bout_keys(
        self,
        *,
        partition: Optional[Partition] = None,
        context_name: Optional[str] = None,
        fish_ids: Optional[Sequence[str]] = None,
    ) -> Iterator[BoutKey]:
        """Iterate stable keys for valid bouts without loading bout arrays."""
        allowed_ids = set(fish_ids) if fish_ids is not None else None

        if partition is not None:
            partition = _normalize_partition(partition)

        for rec in self._fish_records:
            if partition is not None and rec.partition != partition:
                continue

            if context_name is not None and rec.context_name != context_name:
                continue

            if allowed_ids is not None and rec.canonical_fish_id not in allowed_ids:
                continue

            for bout_idx in range(rec.valid_bout_count):
                yield BoutKey(
                    fish_id=rec.canonical_fish_id,
                    fish_index=rec.source_fish_index,
                    bout_index=bout_idx,
                    partition=rec.partition,
                    context_id=rec.context_id,
                    context_name=rec.context_name,
                )

    def iter_bouts(
        self,
        *,
        partition: Optional[Partition] = None,
        context_name: Optional[str] = None,
        fish_ids: Optional[Sequence[str]] = None,
        primary_qc_only: bool = True,
        include_optional: bool = False,
    ) -> Iterator[BoutData]:
        """Iterate valid bouts lazily.

        Parameters
        ----------
        primary_qc_only:
            If True, skip bouts that fail frozen primary structural QC.
            Currently this means non-finite valid bouts only.
        """
        for key in self.iter_bout_keys(
            partition=partition,
            context_name=context_name,
            fish_ids=fish_ids,
        ):
            bout = self.load_bout(
                key.fish_index,
                key.bout_index,
                include_optional=include_optional,
            )

            if primary_qc_only and bout.qc.primary_exclude:
                continue

            yield bout

    # ------------------------------------------------------------------
    # Split-safety helpers
    # ------------------------------------------------------------------

    def assert_fish_partition(
        self,
        fish: Union[str, int],
        expected_partition: Partition,
    ) -> None:
        """Raise if a fish is not in the expected frozen partition."""
        rec = self.get_fish(fish)
        expected_partition = _normalize_partition(expected_partition)

        if rec.partition != expected_partition:
            raise RuntimeError(
                f"{rec.canonical_fish_id} belongs to "
                f"{rec.partition!r}, not {expected_partition!r}."
            )

    def assert_no_fish_overlap(self) -> None:
        """Verify frozen fish IDs occur in exactly one partition."""
        by_partition = {
            part: {
                r.canonical_fish_id
                for r in self.fish_in_partition(part)  # type: ignore[arg-type]
            }
            for part in ("train", "validation", "test")
        }

        if by_partition["train"] & by_partition["validation"]:
            raise RuntimeError("Train/validation fish overlap detected.")
        if by_partition["train"] & by_partition["test"]:
            raise RuntimeError("Train/test fish overlap detected.")
        if by_partition["validation"] & by_partition["test"]:
            raise RuntimeError("Validation/test fish overlap detected.")


# ---------------------------------------------------------------------------
# Small command-line smoke test
# ---------------------------------------------------------------------------

def main() -> None:
    """Run a lightweight validation/smoke test."""
    with DS005() as ds:
        ds.assert_no_fish_overlap()

        print("DS-005 VALIDATION PASSED")
        print("========================")
        print(f"Fish: {ds.n_fish}")
        print(f"Valid bouts: {ds.n_valid_bouts}")
        print(f"Contexts: {len(ds.contexts)}")
        print(f"Frame rate: {ds.frame_rate_hz:g} Hz")
        print(f"Partitions: {ds.partition_counts()}")

        example = ds.get_fish(0)
        print()
        print("Example fish:")
        print(example)

        first_bout = ds.load_bout(0, 0)
        print()
        print("First bout:")
        print(f"  fish_id: {first_bout.key.fish_id}")
        print(f"  partition: {first_bout.key.partition}")
        print(f"  context: {first_bout.key.context_name}")
        print(f"  head_pos shape: {first_bout.head_pos.shape}")
        print(
            "  orientation_smooth shape: "
            f"{first_bout.orientation_smooth.shape}"
        )
        print(f"  speed_head shape: {first_bout.speed_head.shape}")
        print(f"  times_bouts: {first_bout.times_bouts.tolist()}")
        print(f"  QC: {first_bout.qc}")


if __name__ == "__main__":
    main()
