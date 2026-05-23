from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, cast


MissingPolicy = Literal["error", "drop_row", "set_null"]
PathSelection = Mapping[str, str] | Sequence[str] | str


def path_selection_map(
    value: PathSelection | None,
    *,
    format_name: str,
) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, str):
        return {value.rsplit("/", 1)[-1]: value}
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, str], value))
    out: dict[str, str] = {}
    for path in value:
        name = path.rsplit("/", 1)[-1]
        if name in out:
            raise ValueError(
                f"{format_name} path selections must have unique derived column names; "
                f"use an explicit mapping for duplicate name {name!r}"
            )
        out[name] = path
    return out


__all__ = ["MissingPolicy", "PathSelection", "path_selection_map"]
