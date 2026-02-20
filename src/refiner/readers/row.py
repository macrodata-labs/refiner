from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

_MISSING = object()


class Row(Mapping[str, Any]):
    """A row-like object exposed to user code.

    - Behaves like an immutable Mapping[str, Any].
    - Backed either by a dict (CSV) or by a lightweight view (Parquet/Arrow).
    - Use `update(...)`/`drop(...)`/`pop(...)` to produce modified rows (no in-place mutation).
    """

    def to_dict(self) -> dict[str, Any]:
        return dict(self.items())

    def update(self, patch: Mapping[str, Any] | None = None, /, **kwargs: Any) -> "Row":
        """Return a new Row with the given updates applied (immutable).

        Notes:
            - Overwrites existing keys by default.
            - Accepts mapping + kwargs; kwargs win on conflicts.
        """

        merged: dict[str, Any] = {}
        if patch:
            merged.update(patch)
        if kwargs:
            merged.update(kwargs)

        if not merged:
            return self

        if isinstance(self, _OverlayRow):
            # Updating an overlay should "undelete" any keys being set.
            deleted = self.deleted.difference(merged.keys())
            combined_patch = dict(self.patch)
            combined_patch.update(merged)
            return _OverlayRow(base=self.base, patch=combined_patch, deleted=deleted)

        return _OverlayRow(base=self, patch=merged, deleted=frozenset())

    def drop(self, *keys: str) -> "Row":
        """Return a new Row with the given keys hidden (immutable).

        Notes:
            - Dropping a missing key is a no-op.
        """
        if not keys:
            return self

        if isinstance(self, _OverlayRow):
            deleted = self.deleted.union(keys)
            # If a key is dropped, it should not be present in the patch either.
            patch = dict(self.patch)
            for k in keys:
                patch.pop(k, None)
            return _OverlayRow(base=self.base, patch=patch, deleted=deleted)

        return _OverlayRow(base=self, patch={}, deleted=frozenset(keys))

    def pop(self, key: str, default: Any = _MISSING) -> tuple["Row", Any]:
        """Persistent pop: returns (new_row, value) without mutating the base row."""
        try:
            value = self[key]
        except KeyError:
            if default is _MISSING:
                raise
            return (self, default)
        return (self.drop(key), value)


@dataclass(frozen=True, slots=True)
class _OverlayRow(Row):
    """A `Row` overlay that applies a patch and/or deletes keys over a base row."""

    base: Row
    patch: Mapping[str, Any]
    deleted: frozenset[str]

    def __getitem__(self, key: str) -> Any:
        if key in self.deleted:
            raise KeyError(key)
        if key in self.patch:
            return self.patch[key]
        return self.base[key]

    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        for k in self.base:
            if k in self.deleted:
                continue
            seen.add(k)
            yield k
        for k in self.patch:
            if k in self.deleted or k in seen:
                continue
            yield k

    def __len__(self) -> int:
        keys: set[str] = set(self.base)
        keys.update(self.patch.keys())
        keys.difference_update(self.deleted)
        return len(keys)


@dataclass(frozen=True, slots=True)
class DictRow(Row):
    """A `Row` backed by a plain mapping (e.g. from CSV parsing)."""

    data: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


@dataclass(frozen=True, slots=True)
class ArrowRowView(Row):
    """A lightweight view over an Arrow RecordBatch/Table row.

    Notes:
        - `columns` should yield Arrow arrays with __getitem__ returning Arrow Scalars.
        - Values are converted to Python via `.as_py()` on demand.
    """

    names: tuple[str, ...]
    columns: tuple[Any, ...]
    index_by_name: Mapping[str, int]
    row_idx: int

    def __getitem__(self, key: str) -> Any:
        try:
            j = self.index_by_name[key]
        except KeyError as e:
            raise KeyError(key) from e
        return self.columns[j][self.row_idx].as_py()

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)


__all__ = ["Row", "DictRow", "ArrowRowView"]
