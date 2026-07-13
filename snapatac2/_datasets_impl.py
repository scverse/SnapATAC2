"""Internal plumbing for the SnapATAC2 dataset registry.

This module wraps ``scverse_misc.datasets`` and registers the loaders SnapATAC2
needs. The public accessors in :mod:`snapatac2.datasets` call :func:`load`;
end users should not import from this module directly.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import replace
from importlib.resources import files as _resource_files
from pathlib import Path
from typing import Any

import pooch
from scverse_misc.datasets import (
    DatasetEntry,
    FileEntry,
    fetch as _fetch,
    parse_registry,
    register_loader,
)

from snapatac2._snapatac2 import read_motifs

_YAML_RESOURCE = _resource_files("snapatac2").joinpath("datasets.yaml")

_base_url: str | None = None
_registry: dict[str, DatasetEntry] | None = None


def _registry_and_base_url() -> tuple[str | None, dict[str, DatasetEntry]]:
    global _base_url, _registry
    if _registry is None:
        _base_url, _registry = parse_registry(str(_YAML_RESOURCE))
    return _base_url, _registry


def _cache_dir() -> Path:
    """Location where downloaded files are cached.

    Honours the ``SNAP_DATA_DIR`` environment variable for backward
    compatibility with the previous pooch-based implementation.
    """
    override = os.environ.get("SNAP_DATA_DIR")
    if override:
        return Path(override)
    return Path(pooch.os_cache("snapatac2"))


def load(dataset: str, **kwargs: Any):
    """Fetch ``dataset`` from the registry and return whatever its loader produces.

    ``kwargs`` are forwarded to the registered loader for ``entry.type``.
    """
    base_url, registry = _registry_and_base_url()
    if dataset not in registry:
        raise KeyError(
            f"unknown dataset {dataset!r}; known: {sorted(registry)}"
        )
    return fetch_with_fallback(registry[dataset], _cache_dir(), base_url=base_url, **kwargs)


def fetch_with_fallback(
    entry: DatasetEntry,
    cache_dir: str | Path,
    *,
    base_url: str | None = None,
    retries: int = 3,
    **kwargs: Any,
):
    """Call :func:`scverse_misc.datasets.fetch`; on failure, retry via fallback URLs.

    Fallback URLs are read from ``entry.metadata['fallback_urls']`` as a
    ``{file_name: url}`` mapping populated in ``datasets.yaml``. Only files
    with an entry in that mapping are switched to their fallback source; the
    others keep their primary URL (rebuilt from ``base_url + s3_key``).

    Not exposed publicly — call via :func:`load`, which routes through this.
    """
    try:
        return _fetch(entry, cache_dir, base_url=base_url, retries=retries, **kwargs)
    except (OSError, ValueError) as primary_err:
        fallbacks: dict[str, str] = entry.metadata.get("fallback_urls", {}) or {}
        if not fallbacks:
            raise
        alt_files = tuple(
            replace(f, url=fallbacks[f.name], s3_key=None)
            if f.name in fallbacks
            else f
            for f in entry.files
        )
        if alt_files == entry.files:
            raise
        warnings.warn(
            f"primary download for {entry.name!r} failed ({primary_err!r}); "
            f"retrying via fallback URLs",
            RuntimeWarning,
            stacklevel=2,
        )
        alt_entry = replace(entry, files=alt_files)
        return _fetch(alt_entry, cache_dir, base_url=None, retries=retries, **kwargs)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@register_loader("snapatac2")
def _load_snapatac2_file(
    entry: DatasetEntry,
    target: Path,
    download,
    *,
    file: str | None = None,
    processor=None,
    **_: Any,
):
    """Return one file's local path (or a list of paths if a processor extracts).

    Callers pass ``file=<name>`` to select which file in the dataset to fetch.
    ``processor`` is passed through to pooch (e.g. :class:`pooch.Untar`,
    :class:`pooch.Decompress`).
    """
    if file is None:
        raise TypeError(
            f"dataset {entry.name!r}: a `file=` argument is required "
            f"(available: {[f.name for f in entry.files]})"
        )
    fe: FileEntry = entry.file(name=file)
    result = download(fe, processor=processor)
    if isinstance(result, list):
        return [Path(p) for p in result]
    return Path(result)


@register_loader("motif")
def _load_motif(
    entry: DatasetEntry,
    target: Path,
    download,
    **_: Any,
):
    """Parse the entry's single ``.meme`` file into a list of :class:`PyDNAMotif`."""
    fe = entry.files[0]
    return read_motifs(str(download(fe)))
