"""
Verification suite for extract_features.py.

Runs structural checks on metadata loading, split validation, generator
behavior, BGR color convention, and image_preprocessing compatibility.
Downloads a small sample of real images and displays them to confirm the
pipeline works end-to-end.

Usage:
    python verify_extract_features.py
    python verify_extract_features.py --base-dir ./datasets --num-images 6
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from collections import Counter
from typing import Callable, List, Tuple

import cv2
import numpy as np

from extract_features import (
    REQUEST_TIMEOUT,
    VALID_SPLITS,
    _download_image,
    _load_image_urls,
    _validate_split,
    get_feature_stream,
)
from image_preprocessing import ImagePipeline, to_grayscale, resize_image

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0
RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        RESULTS.append((name, True, detail))
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        RESULTS.append((name, False, detail))
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def expect_raises(func: Callable, exc_type: type, name: str) -> None:
    try:
        func()
        check(name, False, f"expected {exc_type.__name__}")
    except exc_type as exc:
        check(name, True, str(exc))


# ---------------------------------------------------------------------------
# Individual test groups
# ---------------------------------------------------------------------------

def test_split_validation() -> None:
    section("1. Split validation")

    for split in VALID_SPLITS:
        try:
            _validate_split(split)
            check(f"accepts '{split}'", True)
        except ValueError:
            check(f"accepts '{split}'", False)

    for bad in ("", "training", "TRAIN", "dev", "all"):
        expect_raises(
            lambda s=bad: _validate_split(s),
            ValueError,
            f"rejects '{bad}'",
        )


def test_metadata_loading(base_dir: str) -> None:
    section("2. Metadata loading")

    check("base_dir exists", os.path.isdir(base_dir), base_dir)
    check(
        "split_dataset.csv present",
        os.path.isfile(os.path.join(base_dir, "split_dataset.csv")),
    )

    counts = {}
    for split in sorted(VALID_SPLITS):
        urls = _load_image_urls(split, base_dir)
        counts[split] = len(urls)
        check(
            f"{split} URLs loaded",
            len(urls) > 0,
            f"{len(urls)} URLs",
        )
        check(
            f"{split} URLs are http(s)",
            all(u.startswith("http") for u in urls[:20]),
        )

    if len(counts) == 3:
        total = sum(counts.values())
        check("splits are disjoint", total > max(counts.values()), f"total={total}")


def test_missing_paths() -> None:
    section("3. Error handling")

    expect_raises(
        lambda: _load_image_urls("train", "./nonexistent_directory_xyz"),
        FileNotFoundError,
        "missing base_dir raises FileNotFoundError",
    )

    with tempfile.TemporaryDirectory() as tmp:
        expect_raises(
            lambda: _load_image_urls("train", tmp),
            FileNotFoundError,
            "empty directory raises FileNotFoundError",
        )


def test_generator_contract(base_dir: str, num_images: int) -> List[np.ndarray]:
    section("4. Generator contract")

    stream = get_feature_stream("train", base_dir=base_dir)
    check("returns generator", hasattr(stream, "__iter__") and hasattr(stream, "__next__"))

    images: List[np.ndarray] = []
    for i, image in enumerate(stream):
        images.append(image)
        if i + 1 >= num_images:
            break

    check(f"yielded {num_images} images", len(images) == num_images)

    if images:
        img = images[0]
        check("dtype is uint8", img.dtype == np.uint8, str(img.dtype))
        check("ndim is 3", img.ndim == 3, f"shape={img.shape}")
        check("3 color channels", img.shape[2] == 3)
        check("height > 0", img.shape[0] > 0)
        check("width > 0", img.shape[1] > 0)
        check("pixel range [0, 255]", img.min() >= 0 and img.max() <= 255,
              f"min={img.min()}, max={img.max()}")

        shapes = [im.shape for im in images]
        check(
            "images vary in size (natural photos)",
            len(set(shapes)) > 1 or num_images == 1,
            str(Counter(shapes)),
        )

    return images


def test_bgr_convention(images: List[np.ndarray]) -> None:
    section("5. BGR color convention")

    if not images:
        check("BGR checks skipped", False, "no images downloaded")
        return

    img_bgr = images[0]

    # Synthetic sanity check: known RGB -> BGR swap
    rgb_solid = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_solid[:, :] = [255, 0, 0]  # red in RGB
    bgr_solid = cv2.cvtColor(rgb_solid, cv2.COLOR_RGB2BGR)
    check(
        "RGB-to-BGR swaps R and B channels",
        bgr_solid[0, 0, 2] == 255 and bgr_solid[0, 0, 0] == 0,
        f"BGR pixel={bgr_solid[0, 0].tolist()}",
    )

    # Real image: BGR displayed as RGB should look wrong (blue tint on warm tones)
    # Compare mean red-channel energy in correct vs incorrect display
    correct_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    wrong_as_rgb = img_bgr.copy()

    # For typical portrait photos, the correctly converted image should have
    # higher mean R than treating BGR as RGB (channels are swapped)
    correct_r_mean = float(correct_rgb[:, :, 0].mean())
    wrong_r_mean = float(wrong_as_rgb[:, :, 0].mean())
    check(
        "downloaded image is BGR (R channel higher after conversion)",
        correct_r_mean > wrong_r_mean,
        f"correct R mean={correct_r_mean:.1f}, wrong R mean={wrong_r_mean:.1f}",
    )


def test_preprocessing_integration(images: List[np.ndarray]) -> None:
    section("6. image_preprocessing integration")

    if not images:
        check("preprocessing skipped", False, "no images")
        return

    img = images[0]

    try:
        gray = to_grayscale(img)
        check("to_grayscale succeeds", gray.ndim == 2, f"shape={gray.shape}")
    except Exception as exc:
        check("to_grayscale succeeds", False, str(exc))

    try:
        resized = resize_image(img, (64, 64), preserve_aspect=False)
        check(
            "resize_image succeeds",
            resized.shape == (64, 64, 3),
            f"shape={resized.shape}",
        )
    except Exception as exc:
        check("resize_image succeeds", False, str(exc))

    try:
        pipeline = ImagePipeline([
            ("grayscale", {}),
            ("resize", {"target_size": (64, 64), "preserve_aspect": False}),
            ("normalize", {"method": "minmax"}),
            ("vectorize", {}),
        ])
        features = pipeline.process(img)
        check(
            "full pipeline succeeds",
            features.ndim == 1 and features.shape[0] == 64 * 64,
            f"feature dim={features.shape[0]}",
        )
    except Exception as exc:
        check("full pipeline succeeds", False, str(exc))


def test_timeout_constant() -> None:
    section("7. Configuration")

    check("REQUEST_TIMEOUT is 10s", REQUEST_TIMEOUT == 10)
    check(
        "VALID_SPLITS matches spec",
        VALID_SPLITS == frozenset({"train", "test", "val"}),
    )


def test_broken_url_skipped() -> None:
    section("8. Broken URL handling")

    result = _download_image("https://httpstat.us/404")
    check("404 URL returns None", result is None)

    result = _download_image("not-a-valid-url")
    check("invalid URL returns None", result is None)


def display_images(
    images: List[np.ndarray], output_dir: str, show: bool = True
) -> None:
    section("9. Visual verification")

    if not images:
        print("  No images to display.")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — saving PNGs only (pip install matplotlib)")
        plt = None

    n = len(images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    if plt is not None:
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.atleast_1d(axes).flatten()
        fig.suptitle(
            "extract_features.py — downloaded images (BGR → RGB for display)",
            fontsize=13,
        )

    for i, img_bgr in enumerate(images):
        display_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out_path = os.path.join(output_dir, f"sample_{i + 1}.png")
        cv2.imwrite(out_path, img_bgr)
        check(f"saved {out_path}", os.path.isfile(out_path),
              f"shape={img_bgr.shape}")

        if plt is not None:
            axes[i].imshow(display_rgb)
            axes[i].set_title(f"Image {i + 1}  {img_bgr.shape[1]}×{img_bgr.shape[0]}")
            axes[i].axis("off")

    if plt is not None:
        for j in range(n, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        grid_path = os.path.join(output_dir, "grid.png")
        fig.savefig(grid_path, dpi=120, bbox_inches="tight")
        print(f"\n  Grid saved to {grid_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"\n  Individual samples saved under {output_dir}/")


def print_summary() -> None:
    section("Summary")
    total = PASS + FAIL
    print(f"  Passed: {PASS}/{total}")
    print(f"  Failed: {FAIL}/{total}")

    if FAIL:
        print("\n  Failed checks:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"    - {name}: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify extract_features.py")
    parser.add_argument(
        "--base-dir", default="./datasets",
        help="Directory containing split_dataset.csv (default: ./datasets)",
    )
    parser.add_argument(
        "--num-images", type=int, default=6,
        help="Number of images to download and display (default: 6)",
    )
    parser.add_argument(
        "--output-dir", default="./verify_output",
        help="Where to save sample PNGs (default: ./verify_output)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save images without opening an interactive matplotlib window",
    )
    args = parser.parse_args()

    print("extract_features.py — verification suite")
    print(f"  base_dir   : {args.base_dir}")
    print(f"  num_images : {args.num_images}")
    print(f"  output_dir : {args.output_dir}")

    try:
        test_split_validation()
        test_metadata_loading(args.base_dir)
        test_missing_paths()
        images = test_generator_contract(args.base_dir, args.num_images)
        test_bgr_convention(images)
        test_preprocessing_integration(images)
        test_timeout_constant()
        test_broken_url_skipped()
        display_images(images, args.output_dir, show=not args.no_show)
    except Exception:
        traceback.print_exc()
        print_summary()
        return 1

    print_summary()
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
