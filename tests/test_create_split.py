"""
Tests for the dataset splitter in ``create_split.py``.

``create_train_val_test_split`` loads ``FINAL_DATASET.csv`` and writes a
deterministic 70/15/15 train/val/test split to ``split_dataset.csv``. These
tests exercise its real splitting logic while keeping the filesystem isolated:

* ``glob.glob`` is monkeypatched so the loader reads a synthetic CSV from
  ``tmp_path`` (or finds nothing, for the error case);
* ``DataFrame.to_csv`` is monkeypatched to capture the result in memory instead
  of writing ``split_dataset.csv`` into the project tree.
"""

from __future__ import annotations

import pytest

import create_split as cs


def _write_dataset_csv(path, n_valid: int = 20):
    """Write a FINAL_DATASET-style CSV with ``n_valid`` rows plus 2 invalid ones.

    Includes an extra ``label_text`` column (to prove it is dropped) and two
    rows with a missing ``label_numeric`` / ``image_url`` (to exercise dropna).

    Returns:
        int: the number of rows that survive ``dropna`` (i.e. ``n_valid``).
    """
    lines = ["image_url,label_numeric,label_text"]
    for i in range(n_valid):
        lines.append(f"http://example.com/{i}.jpg,{i % 2},cat{i % 2}")
    # Two invalid rows: one missing the label, one missing the URL.
    lines.append("http://example.com/missing_label.jpg,,cat")
    lines.append(",1,dog")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_valid


@pytest.fixture
def captured_split(monkeypatch, tmp_path):
    """Run the splitter against a synthetic CSV and capture the written DataFrame.

    Points ``glob.glob`` at a fabricated ``FINAL_DATASET.csv`` and replaces
    ``DataFrame.to_csv`` with a capturing stub, so calling the splitter never
    writes into the repository. Returns a callable that runs the split for a
    given seed and hands back the captured output DataFrame.
    """
    csv_path = tmp_path / "FINAL_DATASET.csv"
    n_valid = _write_dataset_csv(csv_path)

    # The loader resolves the dataset via glob; point it at our temp file.
    monkeypatch.setattr(cs.glob, "glob", lambda *a, **k: [str(csv_path)])

    captured = {}

    def fake_to_csv(self, path, *args, **kwargs):
        # Capture a copy of the frame the splitter intended to persist.
        captured["df"] = self.copy()
        captured["path"] = path

    monkeypatch.setattr(cs.pd.DataFrame, "to_csv", fake_to_csv, raising=True)

    def run(seed: int = 42):
        cs.create_train_val_test_split(seed=seed)
        return captured["df"]

    run.n_valid = n_valid
    return run


class TestCreateSplitErrors:
    """Failure modes of dataset resolution."""

    def test_missing_dataset_raises_filenotfound(self, monkeypatch):
        """A missing ``FINAL_DATASET.csv`` raises ``FileNotFoundError``.

        Edge case: with no dataset on disk the glob returns nothing, so the
        splitter must fail loudly (pointing the user at ``downloaddataset.py``)
        rather than proceeding on empty data.
        """
        # Arrange: glob finds no dataset file.
        monkeypatch.setattr(cs.glob, "glob", lambda *a, **k: [])
        # Act + Assert
        with pytest.raises(FileNotFoundError):
            cs.create_train_val_test_split(seed=42)


class TestCreateSplitLogic:
    """The split proportions, column selection, row cleaning and determinism."""

    def test_output_has_only_expected_columns(self, captured_split):
        """The written frame keeps exactly image_url, label_numeric, data_split.

        The extra ``label_text`` column in the source must be dropped — only the
        two needed columns plus the assigned split are persisted.
        """
        # Act
        df = captured_split()
        # Assert
        assert list(df.columns) == ["image_url", "label_numeric", "data_split"]

    def test_invalid_rows_are_dropped(self, captured_split):
        """Rows missing a URL or label are removed before splitting.

        Edge case: the two malformed source rows must not reach the output, so
        the row count equals the number of valid rows only.
        """
        # Act
        df = captured_split()
        # Assert: only the valid rows survive dropna.
        assert len(df) == captured_split.n_valid

    def test_split_proportions_are_70_15_15(self, captured_split):
        """The split sizes follow the documented 70/15/15 partition.

        Every row is assigned exactly one of train/val/test, with counts
        matching ``int(0.70*n)`` / ``int(0.15*n)`` / remainder.
        """
        # Act
        df = captured_split()
        n = captured_split.n_valid
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
