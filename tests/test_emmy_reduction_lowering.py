
from pathlib import Path

import pytest

from compile import (
    EmmyIRParser,
    EmmyIRLowering,
    PTXEmitter,
)


ROOT = Path(__file__).resolve().parents[1]
IR_PATH = (
    ROOT
    / "problems"
    / "r01_array_sum"
    / "k_sum_1_reduce.ir.json"
)


@pytest.fixture
def reduction_kernel():
    assert IR_PATH.is_file()

    return EmmyIRParser(
        IR_PATH,
        require_meta=True,
    ).parse()


@pytest.fixture
def reduction_ptx(reduction_kernel):
    return EmmyIRLowering(reduction_kernel).lower()


def test_reduction_entry(reduction_ptx):
    assert ".visible .entry k_sum_1_reduce(" in reduction_ptx
    assert ".param .u64 param_x" in reduction_ptx
    assert ".param .u64 param_sum_1" in reduction_ptx


def test_reduction_loop(reduction_ptx):
    assert "setp.ge.u32" in reduction_ptx
    assert "1024;" in reduction_ptx
    assert "add.u32" in reduction_ptx
    assert "bra L_pointwise_end_" in reduction_ptx


def test_reduction_accumulator(reduction_ptx):
    assert "mov.f32" in reduction_ptx
    assert "ld.global.f32" in reduction_ptx
    assert "add.f32" in reduction_ptx


def test_warp_shuffle(reduction_ptx):
    assert reduction_ptx.count(
        "shfl.sync.bfly.b32"
    ) == 5

    for mask in (16, 8, 4, 2, 1):
        assert (
            f", {mask}, 0x1f, 0xffffffff;"
            in reduction_ptx
        )


def test_shared_memory(reduction_ptx):
    assert (
        ".shared .align 4 .f32 acc0_smem[4];"
        in reduction_ptx
    )
    assert "st.shared.f32" in reduction_ptx
    assert reduction_ptx.count("ld.shared.f32") == 4


def test_cta_barrier(reduction_ptx):
    assert reduction_ptx.count("bar.sync 0;") == 1

    store_pos = reduction_ptx.index("st.shared.f32")
    barrier_pos = reduction_ptx.index("bar.sync 0;")
    load_pos = reduction_ptx.index("ld.shared.f32")

    assert store_pos < barrier_pos < load_pos


def test_final_output(reduction_ptx):
    assert "st.global.f32" in reduction_ptx
    assert reduction_ptx.count("st.global.f32") == 1
    assert reduction_ptx.count("ret;") == 1


def test_complete_compiler_output():
    ptx = PTXEmitter().compile_ptx(IR_PATH)

    assert ptx.startswith(".version 8.0")
    assert ".target sm_89" in ptx
    assert ".visible .entry k_sum_1_reduce" in ptx
    assert "shfl.sync.bfly.b32" in ptx
    assert "st.global.f32" in ptx


MEAN_IR_PATH = (
    ROOT
    / "problems"
    / "r03_mean"
    / "k_mean_ffcca3.ir.json"
)


def test_mean_reduction_lowering():
    kernel = EmmyIRParser(
        MEAN_IR_PATH,
        require_meta=True,
    ).parse()

    ptx = EmmyIRLowering(kernel).lower()

    assert ".visible .entry k_mean_ffcca3(" in ptx
    assert ".param .u64 param_mean_keepdim_count" in ptx
    assert ".param .u64 param_mean" in ptx

    assert "shfl.sync.bfly.b32" in ptx
    assert "bar.sync 0;" in ptx
    assert "ld.global.f32" in ptx
    assert "div.rn.f32" in ptx
    assert ptx.count("st.global.f32") == 1


def test_sum_reduction_still_has_no_division():
    kernel = EmmyIRParser(
        IR_PATH,
        require_meta=True,
    ).parse()

    ptx = EmmyIRLowering(kernel).lower()

    assert "div.rn.f32" not in ptx
    assert "st.global.f32" in ptx


ROW_SUM_IR_PATH = (
    ROOT
    / "problems"
    / "r04_row_sum"
    / "k_sum_1_reduce.ir.json"
)


def test_row_sum_reduction_lowering():
    kernel = EmmyIRParser(
        ROW_SUM_IR_PATH,
        require_meta=True,
    ).parse()

    ptx = EmmyIRLowering(kernel).lower()

    assert ".visible .entry k_sum_1_reduce(" in ptx
    assert "mov.u32" in ptx
    assert "%ctaid.x" in ptx

    # Flatten matrix indexes into linear indexes.
    assert "mad.lo.u32" in ptx

    # Reuse the existing cooperative reduction.
    assert ptx.count("shfl.sync.bfly.b32") == 5
    assert "st.shared.f32" in ptx
    assert "bar.sync 0;" in ptx
    assert "ld.shared.f32" in ptx

    # Each block writes one result.
    assert ptx.count("st.global.f32") == 1


def test_row_sum_launch_geometry():
    kernel = EmmyIRParser(
        ROW_SUM_IR_PATH,
        require_meta=True,
    ).parse()

    assert kernel.launch["grid"] == [64, 1, 1]
    assert kernel.launch["block"] == [128, 1, 1]


def test_row_sum_uses_all_rows():
    kernel = EmmyIRParser(
        ROW_SUM_IR_PATH,
        require_meta=True,
    ).parse()

    ptx = EmmyIRLowering(kernel).lower()

    assert "%ctaid.x" in ptx
    assert "mad.lo.u32" in ptx
    assert "st.global.f32" in ptx
