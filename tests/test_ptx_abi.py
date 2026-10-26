
"""PTX ABI Section 7 regression tests.

Tests import PTXABI from the single-file compile.py.
"""

import pytest

from compile import PTXABI


P = PTXABI.Parameter


def test_simple_function_declaration():
    assert PTXABI.function("foo") == ".func foo;"


def test_function_definition():
    src = PTXABI.function("foo", body="ret;")
    assert src == ".func foo\n{\n    ret;\n}"


def test_typed_function_signature():
    src = PTXABI.function(
        "inc_ptr",
        inputs=(
            P("%ptr", ".u32"),
            P("%inc", ".u32"),
        ),
        returns=(P("%res", ".u32"),),
    )

    assert src == (
        ".func (.reg .u32 %res) inc_ptr "
        "(.reg .u32 %ptr, .reg .u32 %inc);"
    )


def test_structure_byte_array_parameter():
    p = P(
        "y",
        ".b8",
        space=".param",
        size=12,
        align=8,
    )

    assert p.declaration() == ".param .align 8 .b8 y[12]"


def test_no_subword_reg_formals():
    for dtype in (".u8", ".s16", ".pred", ".f16"):
        with pytest.raises(ValueError):
            P("v", dtype)


def test_reject_unaligned_param():
    with pytest.raises(ValueError):
        P(
            "buf",
            ".b8",
            space=".param",
            size=12,
            align=3,
        )


def test_max_one_return():
    with pytest.raises(ValueError):
        PTXABI.function(
            "foo",
            returns=(
                P("x", ".u32"),
                P("y", ".u32"),
            ),
        )


def test_duplicate_parameters():
    with pytest.raises(ValueError):
        PTXABI.function(
            "foo",
            inputs=(
                P("x", ".u32"),
                P("x", ".u64"),
            ),
        )


def test_call():
    assert PTXABI.call(
        "inc_ptr",
        arguments=("%r1", "4"),
        returns=("%r2",),
    ) == "call (%r2), inc_ptr, (%r1, 4);"


def test_call_rejects_injection():
    with pytest.raises(ValueError):
        PTXABI.call(
            "foo",
            arguments=("%r1; trap;",),
        )


def test_stack_instructions():
    assert PTXABI.stack_save("%rd1") == (
        "stacksave.u64 %rd1;"
    )

    assert PTXABI.stack_alloc("%rd2", 64) == (
        "alloca.u64 %rd2, 64, 8;"
    )

    assert PTXABI.stack_restore("%rd1") == (
        "stackrestore.u64 %rd1;"
    )


def test_stack_align_constraint():
    with pytest.raises(ValueError):
        PTXABI.stack_alloc(
            "%rd2",
            64,
            alignment=3,
        )

    with pytest.raises(ValueError):
        PTXABI.stack_alloc(
            "%rd2",
            64,
            alignment=1 << 24,
        )


def test_stack_size_constraint():
    with pytest.raises(ValueError):
        PTXABI.stack_alloc("%rd2", 0)
