from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from refiner.pipeline.data.tabular import Tabular

_MISSING = object()
_SHARD_ID_KEY = "__shard_id"


def _next_shard_id(current: str | None, patch: Mapping[str, Any]) -> str | None:
    if _SHARD_ID_KEY not in patch:
        return current
    value = patch[_SHARD_ID_KEY]
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


class Row(Mapping[str, Any]):
    """A row-like object exposed to user code.

    - Behaves like an immutable Mapping[str, Any].
    - Backed either by a dict (CSV) or by a lightweight view (Parquet/Arrow).
    - Use `update(...)`/`drop(...)`/`pop(...)` to produce modified rows (no in-place mutation).
    """

    def to_dict(self) -> dict[str, Any]:
        return dict(self.items())

    @property
    def shard_id(self) -> str | None:
        return None

    def require_shard_id(self) -> str:
        if self.shard_id is None:
            raise ValueError("row is missing shard_id")
        return self.shard_id

    @property
    def tabular_type(self) -> type["Tabular"]:
        from refiner.pipeline.data.tabular import Tabular

        return Tabular

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

        shard_id = _next_shard_id(self.shard_id, merged)
        merged.pop(_SHARD_ID_KEY, None)
        if not merged and shard_id == self.shard_id:
            return self

        if isinstance(self, _OverlayRow):
            # Updating an overlay should "undelete" any keys being set.
            deleted = self.deleted.difference(merged.keys())
            combined_patch = dict(self.patch)
            combined_patch.update(merged)
            return _OverlayRow(
                base=self.base,
                patch=combined_patch,
                deleted=deleted,
                shard_id=shard_id,
            )

        return _OverlayRow(
            base=self,
            patch=merged,
            deleted=frozenset(),
            shard_id=shard_id,
        )

    def with_shard_id(self, shard_id: str) -> "Row":
        return self.update({_SHARD_ID_KEY: shard_id})

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
            return _OverlayRow(
                base=self.base,
                patch=patch,
                deleted=deleted,
                shard_id=self.shard_id,
            )

        return _OverlayRow(
            base=self,
            patch={},
            deleted=frozenset(keys),
            shard_id=self.shard_id,
        )

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
    shard_id: str | None = None

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
    shard_id: str | None = None

    def __post_init__(self) -> None:
        if self.shard_id is not None:
            return
        value = self.data.get(_SHARD_ID_KEY)
        if value is None:
            return
        object.__setattr__(
            self, "shard_id", value if isinstance(value, str) else str(value)
        )

    def __getitem__(self, key: str) -> Any:
        if key == _SHARD_ID_KEY:
            raise KeyError(key)
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        for key in self.data:
            if key == _SHARD_ID_KEY:
                continue
            yield key

    def __len__(self) -> int:
        return len(self.data) - (1 if _SHARD_ID_KEY in self.data else 0)


@dataclass(frozen=True, slots=True)
class ArrowRowView(Row):
    """A lightweight view over an Arrow table row.

    Notes:
        - `columns` should yield Arrow arrays with __getitem__ returning Arrow Scalars.
        - Values are converted to Python via `.as_py()` on demand.
    """

    tabular: "Tabular"
    row_idx: int
    shard_id: str | None = None

    def __getitem__(self, key: str) -> Any:
        if key == _SHARD_ID_KEY:
            raise KeyError(key)
        try:
            j = self.tabular.index_by_name[key]
        except KeyError as e:
            raise KeyError(key) from e
        return self.tabular.columns[j][self.row_idx].as_py()

    def __iter__(self) -> Iterator[str]:
        for key in self.tabular.names:
            if key == _SHARD_ID_KEY:
                continue
            yield key

    def __len__(self) -> int:
        return len(self.tabular.names) - (
            1 if _SHARD_ID_KEY in self.tabular.index_by_name else 0
        )


__all__ = ["Row", "DictRow", "ArrowRowView"]
