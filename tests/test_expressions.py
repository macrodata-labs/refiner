import pytest

import refiner as mdr


def test_expr_rejects_boolean_coercion() -> None:
    with pytest.raises(TypeError, match="cannot be coerced to bool"):
        bool(mdr.col("x") > 1)


def test_expr_rejects_python_and_or_short_circuit() -> None:
    with pytest.raises(TypeError, match="cannot be coerced to bool"):
        _ = (mdr.col("a") > 1) and (mdr.col("b") < 2)

    with pytest.raises(TypeError, match="cannot be coerced to bool"):
        _ = (mdr.col("a") > 1) or (mdr.col("b") < 2)


def test_expr_uses_bitwise_ops_for_logical_composition() -> None:
    expr = (mdr.col("a") > 1) & (mdr.col("b") < 2)
    assert expr.to_plan()["op"] == "and"

