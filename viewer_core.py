"""Pure file discovery / pixel-exact duplicate detection (no OCR or network)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
from typing import Callable, Iterable

from PIL import Image, ImageOps

EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.gif'})


def natural_key(value: str) -> list[tuple[int, object]]:
    return [(1, int(p)) if p.isdecimal() else (0, p.casefold())
            for p in re.split(r'(\d+)', value)]


def discover_images(directory: Path, recursive: bool = False) -> list[Path]:
    """Never follow folder symlinks; skip inaccessible folders without crashing."""
    result: list[Path] = []
    if recursive:
        for base, dirs, files in os.walk(directory, followlinks=False):
            dirs[:] = sorted((d for d in dirs if not (Path(base) / d).is_symlink()), key=natural_key)
            for name in files:
                candidate = Path(base) / name
                if candidate.suffix.lower() in EXTENSIONS and candidate.is_file():
                    result.append(candidate)
    else:
        result = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
    result.sort(key=lambda p: natural_key(str(p.relative_to(directory))))
    return result


def sort_images(paths: Iterable[Path], root: Path | None = None,
                sort_by: str = 'name') -> list[Path]:
    paths = list(paths)
    if sort_by == 'modified':
        return sorted(paths, key=lambda p: (p.stat().st_mtime_ns, natural_key(str(p))))
    if sort_by == 'size':
        return sorted(paths, key=lambda p: (p.stat().st_size, natural_key(str(p))))
    return sorted(paths, key=lambda p: natural_key(str(p.relative_to(root) if root else p.name)))


@dataclass(frozen=True)
class DuplicateGroup:
    paths: tuple[Path, ...]
    kind: str  # 'file' = identical bytes, 'pixels' = identical decoded RGBA pixels
    width: int
    height: int


@dataclass
class DuplicateScan:
    groups: list[DuplicateGroup]
    errors: list[str]
    checked: int
    cancelled: bool = False

    def path_to_group(self) -> dict[Path, int]:
        return {p: index + 1 for index, group in enumerate(self.groups)
                for p in group.paths}


def _file_digest(path: Path, cancelled: Callable[[], bool]) -> bytes:
    digest = sha256()
    with path.open('rb') as f:
        while block := f.read(1024 * 1024):
            if cancelled():
                raise InterruptedError
            digest.update(block)
    return digest.digest()


def find_duplicates(paths: Iterable[Path], *,
                    progress: Callable[[int, int, Path], None] | None = None,
                    cancelled: Callable[[], bool] | None = None) -> DuplicateScan:
    """Strict matches: same display-oriented RGBA pixels and dimensions.

    No perceptual similarity matching: resized, lossy-recompressed and edited
    copies are deliberately *not* duplicates. EXIF orientation is respected.
    SHA-256 collisions are negligible for this utility, but hashes alone are
    not a proof of pixel identity against an adversarially crafted input.
    """
    files = list(dict.fromkeys(Path(p) for p in paths))
    is_cancelled = cancelled or (lambda: False)
    pixel_groups: dict[tuple[int, int, bytes], list[Path]] = defaultdict(list)
    errors: list[str] = []
    checked = 0
    for idx, path in enumerate(files):
        if is_cancelled():
            return DuplicateScan([], errors, checked, cancelled=True)
        try:
            with Image.open(path) as original:
                frame = ImageOps.exif_transpose(original)
                rgba = frame.convert('RGBA')
                w, h = rgba.size
                digest = sha256(rgba.tobytes()).digest()
                pixel_groups[(w, h, digest)].append(path)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            errors.append(f'{path}: {exc}')
        checked = idx + 1
        if progress is not None:
            progress(checked, len(files), path)
    groups = []
    for (w, h, _), group_paths in pixel_groups.items():
        if len(group_paths) < 2:
            continue
        try:
            file_hashes = [_file_digest(path, is_cancelled) for path in group_paths]
        except InterruptedError:
            return DuplicateScan([], errors, checked, cancelled=True)
        except OSError as exc:
            errors.append(f'Ошибка чтения файлов при сравнении: {exc}')
            file_hashes = []
        kind = 'file' if len(file_hashes) == len(group_paths) and len(set(file_hashes)) == 1 else 'pixels'
        groups.append(DuplicateGroup(tuple(sorted(group_paths, key=lambda p: natural_key(str(p)))), kind, w, h))
    groups.sort(key=lambda group: natural_key(str(group.paths[0])))
    return DuplicateScan(groups, errors, checked)