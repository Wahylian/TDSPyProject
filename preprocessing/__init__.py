"""
Modular image preprocessing package — internal implementation.

This package holds the focused submodules that do the actual work:

    transforms.py  - per-image transforms (to_grayscale, resize_image,
                     normalize_image, reduce_noise)
    vectorize.py   - per-image vectorization (vectorize_image)
    reduce.py      - batch-level dimensionality reduction (reduce_dimensions)
    pipeline.py    - composition / batching (ImagePipeline, batch_process,
                     compose, pipeline_decorator, BATCH_LEVEL_OPS)
    io.py          - image I/O helpers (load_image_from_bytes / _file / _pil)

Do not import the public API from this package directly. The single, stable
front door is the ``image_preprocessing`` facade at the project root::

    from image_preprocessing import ImagePipeline, batch_process

Keeping the public surface in exactly one place (the facade's ``__all__``)
means there is only one list to maintain when the API changes, and it lets the
internal layout here be reorganized freely without touching callers. This
``__init__`` is intentionally left without re-exports so the two surfaces can
never drift; submodules are still importable as ``preprocessing.<module>`` for
the facade and for tests that need a specific module.
"""
