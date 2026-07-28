"""Tests for download_dataset.restructure_to_real_fake: the post-download merge.

main() itself does a real kagglehub network download and isn't tested; this
covers the pure filesystem restructuring it delegates to: merging ddata's
train/test splits into <root>/real and <root>/fake, its idempotent skip when
already restructured, and its name-collision guard.
"""

from __future__ import annotations

import pytest

from download_dataset import restructure_to_real_fake


def _make_ddata(root, real_train, real_test, fake_train, fake_test):
    """Build a ddata/{train,test}/{real,fake} tree with the given filenames."""
    for split, names_by_label in (
        ("train", {"real": real_train, "fake": fake_train}),
        ("test", {"real": real_test, "fake": fake_test}),
    ):
        for label, names in names_by_label.items():
            folder = root / "ddata" / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for name in names:
                (folder / name).write_bytes(b"x")


class TestRestructureToRealFake:
    """Merging ddata/{train,test}/{real,fake} into <root>/{real,fake}."""

    def test_merges_train_and_test_into_label_folders(self, tmp_path):
        _make_ddata(
            tmp_path,
            real_train=["r1.jpg", "r2.jpg"], real_test=["r3.jpg"],
            fake_train=["f1.jpg"], fake_test=["f2.jpg", "f3.jpg"],
        )

        restructure_to_real_fake(tmp_path)

        assert {p.name for p in (tmp_path / "real").iterdir()} == {"r1.jpg", "r2.jpg", "r3.jpg"}
        assert {p.name for p in (tmp_path / "fake").iterdir()} == {"f1.jpg", "f2.jpg", "f3.jpg"}
        assert not (tmp_path / "ddata").exists()

    def test_is_idempotent_once_ddata_is_gone(self, tmp_path, capsys):
        _make_ddata(tmp_path, real_train=["r1.jpg"], real_test=[], fake_train=["f1.jpg"], fake_test=[])
        restructure_to_real_fake(tmp_path)

        # Second call: ddata is already gone, so this must be a clean no-op.
        restructure_to_real_fake(tmp_path)

        assert {p.name for p in (tmp_path / "real").iterdir()} == {"r1.jpg"}
        assert "already restructured" in capsys.readouterr().out

    def test_name_collision_between_train_and_test_raises(self, tmp_path):
        _make_ddata(
            tmp_path,
            real_train=["dup.jpg"], real_test=["dup.jpg"],
            fake_train=[], fake_test=[],
        )

        with pytest.raises(FileExistsError):
            restructure_to_real_fake(tmp_path)
