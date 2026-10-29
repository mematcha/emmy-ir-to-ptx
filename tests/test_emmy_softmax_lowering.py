
from pathlib import Path

import pytest

from compile import EmmyIRParser, EmmyIRLowering, PTXEmitter


ROOT = Path(__file__).resolve().parents[1]

IR_PATH = (
    ROOT
    / "problems"
    / "f02_row_softmax"
    / "k_softmax_05263d.ir.json"
)


@pytest.fixture(scope="module")
def ptx():
    kernel = EmmyIRParser(
        IR_PATH, require_meta=True
    ).parse()
    return EmmyIRLowering(kernel).lower()


def test_softmax_entry(ptx):
    assert ".visible .entry k_softmax_05263d(" in ptx
    assert "ret;" in ptx


def test_softmax_uses_both_shared_buffers(ptx):
    assert "acc0_smem" in ptx
    assert "acc1_smem" in ptx


def test_softmax_has_max_and_sum_reductions(ptx):
    assert "max.f32" in ptx
    assert "add.f32" in ptx

    # Five butterfly stages for each reduction.
    assert ptx.count("shfl.sync.bfly.b32") == 10


def test_softmax_has_exponentials(ptx):
    # PTX uses base-two exponential.
    assert "ex2.approx.f32" in ptx
    assert "0f3fb8aa3b" in ptx


def test_softmax_has_reduction_barriers(ptx):
    # The max pass and the sum pass each store warp partials,
    # then broadcast the reduced value. Two barriers per pass.
    assert ptx.count("bar.sync 0;") == 4


def test_softmax_uses_block_local_indexing(ptx):
    assert "%ctaid.x" in ptx
    assert "%tid.x" in ptx
    assert "mad.lo.u32" in ptx


def test_softmax_writes_results(ptx):
    # 1 / row_sum, rounded to nearest.
    assert "div.rn.f32" in ptx
    assert "0f3f800000" in ptx
    assert "st.global.f32" in ptx


def test_full_compiler_supports_softmax():
    output = PTXEmitter().compile_ptx(IR_PATH)

    assert output.startswith(".version 8.0")
    assert ".visible .entry k_softmax_05263d(" in output
    assert "ex2.approx.f32" in output
    assert "st.global.f32" in output

