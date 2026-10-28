
from pathlib import Path

from compile import EmmyIRParser, EmmyIRLowering, PTXEmitter


ROOT = Path(__file__).resolve().parents[1]

IR_PATH = (
    ROOT
    / "problems"
    / "f03_l2norm"
    / "k_div_reduce.ir.json"
)


def get_ptx():
    kernel = EmmyIRParser(
        IR_PATH, require_meta=True
    ).parse()
    return EmmyIRLowering(kernel).lower()


def test_l2norm_compiles():
    ptx = get_ptx()

    assert ".visible .entry k_div_reduce(" in ptx
    assert "ret;" in ptx


def test_l2norm_coordinates():
    ptx = get_ptx()

    assert "%tid.x" in ptx
    assert "%ctaid.x" in ptx
    assert "mad.lo.u32" in ptx


def test_l2norm_sum_of_squares():
    ptx = get_ptx()

    assert "ld.global.f32" in ptx
    assert "mul.f32" in ptx
    assert "add.f32" in ptx


def test_l2norm_cooperative_reduction():
    ptx = get_ptx()

    assert ptx.count("shfl.sync.bfly.b32") == 5
    assert "st.shared.f32" in ptx
    assert "ld.shared.f32" in ptx


def test_l2norm_normalization():
    ptx = get_ptx()

    assert "sqrt.rn.f32" in ptx
    assert "rcp.rn.f32" in ptx
    assert "st.global.f32" in ptx


def test_l2norm_two_barriers():
    ptx = get_ptx()

    assert ptx.count("bar.sync 0;") == 2

    first = ptx.index("bar.sync 0;")
    second = ptx.index("bar.sync 0;", first + 1)
    final_store = ptx.index("st.global.f32")

    assert first < second < final_store


def test_full_emitter_supports_l2norm():
    ptx = PTXEmitter().compile_ptx(IR_PATH)

    assert ptx.startswith(".version 8.0")
    assert ".visible .entry k_div_reduce(" in ptx
    assert "sqrt.rn.f32" in ptx
    assert "st.global.f32" in ptx
