"""Tests for the dataset splitter in create_split.py.

create_split scans real/ and fake/ image folders and writes a manifest CSV
(photo_name, photo_path, label, split) shuffled and partitioned 70/15/15. These
build a small real dataset tree under tmp_path (empty files suffice, the splitter
only scans filenames) so nothing touches the repository's real dataset.
"""

from __future__ import annotations

import pandas as pd
import pytest

import create_split as cs


def _make_dataset_tree(root, n_real: int, n_fake: int):
    """Create real/ and fake/ trees of empty .jpg files under root; return root."""
    real_dir = root / "real"
    fake_dir = root / "fake"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    for i in range(n_real):
        (real_dir / f"real_{i}.jpg").write_bytes(b"")
    for i in range(n_fake):
        (fake_dir / f"fake_{i}.jpg").write_bytes(b"")
    return root


@pytest.fixture
def run_split(tmp_path):
    """Build a 20-image dataset and return a callable that runs the splitter.

    The callable runs create_split for a given seed and returns the DataFrame.
    Exposes n, out_csv, and data_dir for assertions.
    """
    n_real, n_fake = 10, 10
    data_dir = _make_dataset_tree(tmp_path / "data", n_real, n_fake)
    out_csv = tmp_path / "out" / "dataset_split.csv"

    def run(seed: int = 42):
        return cs.create_split(data_dir=data_dir, output_csv=out_csv, seed=seed)

    run.n = n_real + n_fake
    run.out_csv = out_csv
    run.data_dir = data_dir
    return run


class TestCreateSplitErrors:
    """Failure modes when the dataset layout is missing."""

    def test_missing_data_dir_raises_filenotfound(self, tmp_path):
        """A non-existent dataset directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            cs.create_split(
                data_dir=tmp_path / "does_not_exist",
                output_csv=tmp_path / "out.csv",
            )

    def test_missing_class_subfolder_raises_filenotfound(self, tmp_path):
        """A dataset dir missing the fake/ subfolder raises FileNotFoundError."""
        (tmp_path / "data" / "real").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            cs.create_split(
                data_dir=tmp_path / "data",
                output_csv=tmp_path / "out.csv",
            )


class TestManifestContent:
    """Columns, labelling, path format, and image coverage of the manifest."""

    def test_output_has_exactly_the_expected_columns(self, run_split):
        """The manifest carries exactly photo_name, photo_path, label, split."""
        df = run_split()
        assert list(df.columns) == ["photo_name", "photo_path", "label", "split"]

    def test_all_images_are_included(self, run_split):
        """Every .jpg under both class folders appears exactly once."""
        df = run_split()
        assert len(df) == run_split.n

    def test_png_and_jpg_images_are_both_included(self, tmp_path):
        """The scan picks up .png images as well as .jpg ones."""
        real_dir = tmp_path / "data" / "real"
        fake_dir = tmp_path / "data" / "fake"
        real_dir.mkdir(parents=True)
        fake_dir.mkdir(parents=True)
        (real_dir / "real_0.jpg").write_bytes(b"")
        (real_dir / "real_1.png").write_bytes(b"")
        (fake_dir / "fake_0.png").write_bytes(b"")

        df = cs.create_split(
            data_dir=tmp_path / "data", output_csv=tmp_path / "out.csv"
        )
        assert set(df["photo_name"]) == {"real_0.jpg", "real_1.png", "fake_0.png"}

    def test_labels_match_class_folders(self, run_split):
        """Real images get label 0 and fake images get label 1, from the folder."""
        df = run_split().set_index("photo_name")
        assert (df.loc[[f"real_{i}.jpg" for i in range(10)], "label"] == 0).all()
        assert (df.loc[[f"fake_{i}.jpg" for i in range(10)], "label"] == 1).all()

    def test_photo_path_uses_forward_slashes_and_ends_with_name(self, run_split):
        """photo_path is forward-slashed and ends with its photo_name."""
        df = run_split()
        for _, row in df.iterrows():
            assert "\\" not in row["photo_path"]
            assert row["photo_path"].endswith(row["photo_name"])

    def test_manifest_is_written_to_disk(self, run_split):
        """The manifest is persisted to output_csv and reads back identically."""
        df = run_split()
        assert run_split.out_csv.is_file()
        from_disk = pd.read_csv(run_split.out_csv)
        assert len(from_disk) == len(df)
        assert list(from_disk.columns) == list(df.columns)


class TestSplitLogic:
    """The 70/15/15 partition sizes, coverage, and reproducibility."""

    def test_split_proportions_are_70_15_15(self, run_split):
        """Split sizes follow int(0.70*n) / int(0.15*n) / remainder."""
        df = run_split()
        n = run_split.n
        counts = df["split"].value_counts().to_dict()
        assert counts.get("train") == int(0.70 * n)
        assert counts.get("val") == int(0.15 * n)
        assert counts.get("test") == n - int(0.70 * n) - int(0.15 * n)
        assert sum(counts.values()) == n

    def test_every_row_is_assigned_a_known_split(self, run_split):
        """Only the three valid split labels appear; no row is left unassigned."""
        df = run_split()
        assert set(df["split"]) == {"train", "val", "test"}

    def test_split_is_deterministic_for_fixed_seed(self, run_split):
        """The same seed produces an identical split assignment across runs."""
        first = run_split(seed=42).sort_values("photo_name")["split"].tolist()
        second = run_split(seed=42).sort_values("photo_name")["split"].tolist()
        assert first == second

    def test_different_seeds_can_change_assignment(self, run_split):
        """Two distinct seeds yield a different per-photo split assignment."""
        first = run_split(seed=1).sort_values("photo_name")["split"].tolist()
        second = run_split(seed=2).sort_values("photo_name")["split"].tolist()
        assert first != second
