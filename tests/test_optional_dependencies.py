from __future__ import annotations

import pytest

from refiner.utils.imports import check_required_dependencies


def test_check_required_dependencies_has_install_hint() -> None:
    with pytest.raises(
        ImportError,
        match=r"Please install `missing_package` to use refiner.video "
        r"\(`pip install missing_package`\), or simply "
        r"`pip install macrodata-refiner\[video\]`\.",
    ):
        check_required_dependencies(
            "refiner.video",
            ["missing_package"],
            dist="video",
        )
