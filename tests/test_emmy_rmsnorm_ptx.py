
from pathlib import Path

from compile import EmmyIRParser, EmmyIRLowering, PTXEmitter


ROOT = Path(__file__).resolve().parents[1]
IR_PATH = (
    ROOT / "problems" / "f01_rmsnorm"
    / "k_mean_cbd1de.ir.json"
)


def generate_ptx():
    kernel = EmmyIRParser(
        IR_PATH, require_meta=True
    ).parse()
    return EmmyIRLowering(kernel).lower()


def test_rmsnorm_compiles():
    ptx = generate_ptx()
    assert ".visible .entry k_mean_cbd1de(" in ptx
    assert "ret;" in ptx


def test_rmsnorm_uses_row_indexing():
    ptx = generate_ptx()
    assert "%ctaid.x" in ptx
    assert "%tid.x" in ptx
    assert "mad.lo.u32" in ptx


def test_rmsnorm_reduction():
    ptx = generate_ptx()
    assert ptx.count("shfl.sync.bfly.b32") == 5
    assert "st.shared.f32" in ptx
    assert "ld.shared.f32" in ptx


def test_rmsnorm_scalar_parameters():
    ptx = generate_ptx()
    assert "param_mean_count" in ptx
    assert "param_add_c1" in ptx
    assert "param_w" in ptx
    assert "param_mul_1" in ptx


def test_rmsnorm_arithmetic():
    ptx = generate_ptx()
    assert "sqrt.rn.f32" in ptx
    assert ptx.count("div.rn.f32") == 2
    assert "mul.f32" in ptx
    assert "add.f32" in ptx


def test_rmsnorm_barriers():
    ptx = generate_ptx()
    assert ptx.count("bar.sync 0;") == 2

    first = ptx.index("bar.sync 0;")
    second = ptx.index("bar.sync 0;", first + 1)
    final_store = ptx.index("st.global.f32")

    assert first < second < final_store


def test_rmsnorm_weighted_output():
    ptx = generate_ptx()
    assert "st.global.f32" in ptx
    assert ptx.count("st.global.f32") == 1


def test_full_compiler_supports_rmsnorm():
    ptx = PTXEmitter().compile_ptx(IR_PATH)

    assert ptx.startswith(".version 8.0")
    assert ".visible .entry k_mean_cbd1de(" in ptx
    assert "sqrt.rn.f32" in ptx
    assert "st.global.f32" in ptx
