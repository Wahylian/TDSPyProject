"""
Tests for the dataset splitter in ``create_split.py``.

``create_train_val_test_split`` loads ``photos_dataset.csv`` (the local-path
mapping produced by ``download_photos.py``) and writes a deterministic 70/15/15
train/val/test split to ``split_dataset.csv``. These tests exercise its real
splitting logic while keeping the filesystem isolated:

* ``os.path.isfile`` and ``pandas.read_csv`` are monkeypatched so the loader
  reads a synthetic in-memory frame (or reports the mapping as missing, for the
  error case);
* ``DataFrame.to_csv`` is monkeypatched to capture the result in memory instead
  of writing ``split_dataset.csv`` into the project tree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import create_split as cs


def _make_dataset_df(n_valid: int = 20) -> pd.DataFrame:
    """Build a photos_dataset-style frame with ``n_valid`` fully-valid rows.

    Adds an extra ``label_text`` column (to prove it is dropped), one row missing
    its ``label_numeric`` and one row missing its ``image_path`` (a failed
    download) — both of which must be dropped before splitting.
    """
    rows = [
        {"image_path": f"photos/{i}.jpg", "label_numeric": i % 2, "label_text": f"cat{i % 2}"}
        for i in range(n_valid)
    ]
    # One row with no label and one row with no path: both dropped.
    rows.append({"image_path": "photos/nolabel.jpg", "label_numeric": np.nan, "label_text": "cat"})
    rows.append({"image_path": np.nan, "label_numeric": 1, "label_text": "dog"})
    return pd.DataFrame(rows)


@pytest.fixture
def captured_split(monkeypatch):
    """Run the splitter against a synthetic mapping and capture the written DataFrame.

    Reports the mapping file as present, feeds the splitter a fabricated frame in
    place of ``photos_dataset.csv``, and replaces ``DataFrame.to_csv`` with a
    capturing stub, so calling the splitter never touches the repository. Returns
    a callable that runs the split for a given seed and hands back the captured
    output DataFrame.
    """
    n_valid = 20
    source_df = _make_dataset_df(n_valid)

    monkeypatch.setattr(cs.os.path, "isfile", lambda *a, **k: True)
    monkeypatch.setattr(cs.pd, "read_csv", lambda *a, **k: source_df.copy())

    captured = {}

    def fake_to_csv(self, path, *args, **kwargs):
        # Capture a copy of the frame the splitter intended to persist.
        captured["df"] = self.copy()
        captured["path"] = path

    monkeypatch.setattr(cs.pd.DataFrame, "to_csv", fake_to_csv, raising=True)

    def run(seed: int = 42):
        cs.create_train_val_test_split(seed=seed)
        return captured["df"]

    # Both the label-less row and the path-less row are dropped.
    run.n_surviving = n_valid
    return run


class TestCreateSplitErrors:
    """Failure modes of dataset resolution."""

    def test_missing_mapping_raises_filenotfound(self, monkeypatch):
        """A missing ``photos_dataset.csv`` raises ``FileNotFoundError``.

        Edge case: with no mapping on disk the splitter must fail loudly
        (pointing the user at ``download_photos.py``) rather than proceeding on
        empty data.
        """
        # Arrange: the mapping file is reported as absent.
        monkeypatch.setattr(cs.os.path, "isfile", lambda *a, **k: False)
        # Act + Assert
        with pytest.raises(FileNotFoundError):
            cs.create_train_val_test_split(seed=42)


class TestCreateSplitLogic:
    """The split proportions, column selection, row cleaning and determinism."""

    def test_output_has_only_expected_columns(self, captured_split):
        """The written frame keeps exactly image_path, label_numeric, data_split.

        The extra ``label_text`` column in the source must be dropped — only the
        two needed columns plus the assigned split are persisted.
        """
        # Act
        df = captured_split()
        # Assert
        assert list(df.columns) == ["image_path", "label_numeric", "data_split"]

    def test_missing_label_or_path_rows_are_dropped(self, captured_split):
        """Rows missing a label or a path are removed before splitting.

        Both the label-less row and the failed-download row (empty
        ``image_path``) must be dropped, so only the fully-valid rows survive and
        no surviving row has an empty path.
        """
        # Act
        df = captured_split()
        # Assert: only the valid rows survive, none with a missing path.
        assert len(df) == captured_split.n_surviving
        assert df["image_path"].isna().sum() == 0

    def test_split_proportions_are_70_15_15(self, captured_split):
        """The split sizes follow the documented 70/15/15 partition.

        Every row is assigned exactly one of train/val/test, with counts
        matching ``int(0.70*n)`` / ``int(0.15*n)`` / remainder.
        """
        # Act
        df = captured_split()
        n = captured_split.n_surviving
        counts = df["data_split"].value_counts().to_dict()
        # Assert: each partition has its expected size and they sum to n.
        assert counts.get("train") == int(0.70 * n)
        assert counts.get("val") == int(0.15 * n)
        assert counts.get("test") == n - int(0.70 * n) - int(0.15 * n)
        assert sum(counts.values()) == n

    def test_every_row_is_assigned_a_known_split(self, captured_split):
        """No row is left with an empty/unknown split label.

        Guards against an off-by-one in the index partitioning that would leave
        a row with the initial empty-string placeholder.
        """
        # Act
        df = captured_split()
        # Assert: the only labels present are the three valid splits.
        assert set(df["data_split"]) == {"train", "val", "test"}

    def test_split_is_deterministic_for_fixed_seed(self, captured_split):
        """The same seed produces an identical split assignment across runs.

        Reproducibility is the whole point of the seeded shuffle: two runs with
        seed 42 must assign every row to the same split.
        """
        # Act: run the split twice with the same seed.
        first = captured_split(seed=42)["data_split"].tolist()
        second = captured_split(seed=42)["data_split"].tolist()
        # Assert: identical assignments.
        assert first == second

    def test_different_seeds_can_change_assignment(self, captured_split):
        """Different seeds generally yield a different shuffle.

        Confirms the seed actually drives the shuffle (not ignored): two
        distinct seeds must not produce the identical assignment for a
        reasonably sized dataset.
        """
        # Act
        first = captured_split(seed=1)["data_split"].tolist()
        second = captured_split(seed=2)["data_split"].tolist()
        # Assert: the seed influences the partitioning.
        assert first != second
