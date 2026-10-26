
from pathlib import Path

import pytest

from compile import EmmyIRParser, EmmyIRLowering

from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]

IR_PATH = (
    ROOT
    / "problems"
    / "p00_scalar_add"
    / "k_add_pointwise.ir.json"
)


@pytest.fixture
def kernel():
    assert IR_PATH.is_file(), f"Missing IR: {IR_PATH}"

    return EmmyIRParser(
        IR_PATH,
        require_meta=True,
    ).parse()


def test_pointwise_kernel_name(kernel):
    assert kernel.name == "k_add_pointwise"


def test_pointwise_arguments(kernel):
    assert kernel.arg_order == (
        "x0", "x1", "add"
    )


def test_pointwise_lowering_generates_entry(kernel):
    ptx = EmmyIRLowering(kernel).lower()

    assert ".visible .entry k_add_pointwise(" in ptx
    assert ".param .u64 param_x0" in ptx
    assert ".param .u64 param_x1" in ptx
    assert ".param .u64 param_add" in ptx


def test_pointwise_loads_parameters(kernel):
    ptx = EmmyIRLowering(kernel).lower()

    assert "ld.param.u64" in ptx
    assert "[param_x0]" in ptx
    assert "[param_x1]" in ptx
    assert "[param_add]" in ptx


def test_pointwise_global_memory(kernel):
    ptx = EmmyIRLowering(kernel).lower()

    assert ptx.count("ld.global.f32") >= 4
    assert ptx.count("st.global.f32") >= 2


def test_pointwise_arithmetic(kernel):
    ptx = EmmyIRLowering(kernel).lower()

    assert ptx.count("add.f32") >= 2


def test_pointwise_bounds_guard(kernel):
    ptx = EmmyIRLowering(kernel).lower()

    assert "setp.ge.u32" in ptx
    assert "bra L_pointwise_end_1;" in ptx
    assert "L_pointwise_end_1:" in ptx


def test_pointwise_terminates(kernel):
    ptx = EmmyIRLowering(kernel).lower()

    assert ptx.count("ret;") == 1
    assert ptx.rstrip().endswith("}")


def test_unsupported_reduction_fails():
    reduction_path = (
        ROOT
        / "problems"
        / "r01_array_sum"
        / "k_sum_1_reduce.ir.json"
    )

    assert reduction_path.is_file()

    kernel = EmmyIRParser(
        reduction_path,
        require_meta=True,
    ).parse()

    with pytest.raises(NotImplementedError):
        EmmyIRLowering(kernel).lower()





@pytest.mark.parametrize(
    "operation,expected_instruction",
    [
        ("relu", "max.f32"),
        ("negative", "neg.f32"),
        ("absolute", "abs.f32"),
        ("identity", "mov.f32"),
    ],
)
def test_unary_f32_operations(
    kernel, operation, expected_instruction
):
    lowering = EmmyIRLowering(kernel)

    src = lowering.gen.register(".f32")
    lowering.values["input_value"] = src

    stmt = SimpleNamespace(
        name="result",
        op=SimpleNamespace(name=operation),
        args=("input_value",),
        dtype=SimpleNamespace(name="f32"),
    )

    lowering._assign(stmt)

    assert "result" in lowering.values

    ptx = lowering.gen.render_entry()

    assert expected_instruction in ptx


@pytest.mark.parametrize("operation", ["maximum", "minimum"])
def test_binary_minmax(kernel, operation):
    lowering = EmmyIRLowering(kernel)

    lowering.values["a"] = lowering.gen.register(".f32")
    lowering.values["b"] = lowering.gen.register(".f32")

    stmt = SimpleNamespace(
        name="result",
        op=SimpleNamespace(name=operation),
        args=("a", "b"),
        dtype=SimpleNamespace(name="f32"),
    )

    lowering._assign(stmt)

    ptx = lowering.gen.render_entry()

    assert ("max.f32" if operation == "maximum" else "min.f32") in ptx


def test_unknown_unary_operation_fails(kernel):
    lowering = EmmyIRLowering(kernel)
    lowering.values["a"] = lowering.gen.register(".f32")

    stmt = SimpleNamespace(
        name="result",
        op=SimpleNamespace(name="unsupported_op"),
        args=("a",),
        dtype=SimpleNamespace(name="f32"),
    )

    with pytest.raises(
        NotImplementedError,
        match="Unsupported unary",
    ):
        lowering._assign(stmt)

@pytest.mark.parametrize(
    "operation,arity,expected",
    [
        ("sqrt", 1, "sqrt.rn.f32"),
        ("rsqrt", 1, "rsqrt.approx.f32"),
        ("divide", 2, "div.rn.f32"),
    ],
)
def test_normalization_arithmetic(
    kernel, operation, arity, expected
):
    from types import SimpleNamespace

    lowering = EmmyIRLowering(kernel)

    lowering.values["a"] = lowering.gen.register(".f32")
    lowering.values["b"] = lowering.gen.register(".f32")

    args = ("a",) if arity == 1 else ("a", "b")

    stmt = SimpleNamespace(
        name="normalized",
        op=SimpleNamespace(name=operation),
        args=args,
        dtype=SimpleNamespace(name="f32"),
    )

    lowering._assign(stmt)

    ptx = lowering.gen.render_entry()

    assert expected in ptx
    assert "normalized" in lowering.values
