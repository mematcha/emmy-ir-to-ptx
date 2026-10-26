
from pathlib import Path

from compile import EmmyIRParser


ROOT = Path(__file__).resolve().parents[1]

IR_PATH = (
    ROOT
    / "problems"
    / "f01_rmsnorm"
    / "k_mean_cbd1de.ir.json"
)


def test_rmsnorm_ir_has_reduction_and_output_pass():
    kernel = EmmyIRParser(
        IR_PATH,
        require_meta=True,
    ).parse()

    assert len(kernel.body) == 1

    tile = kernel.body[0]
    assert type(tile).__name__ == "Tile"

    kinds = [
        type(stmt).__name__
        for stmt in tile.body
    ]

    assert kinds[0] == "StridedLoop"
    assert kinds[-1] == "StridedLoop"

    assert "WarpShuffle" in kinds
    assert "Smem" in kinds
    assert "TreeHalve" in kinds
    assert "Sync" in kinds


def test_rmsnorm_uses_explicit_launch_metadata():
    kernel = EmmyIRParser(
        IR_PATH,
        require_meta=True,
    ).parse()

    assert kernel.launch["grid"]
    assert kernel.launch["block"]

    arg_names = {
        arg["name"]
        for arg in kernel.metadata["selected_kernel"]["args"]
    }

    assert set(kernel.arg_order) == arg_names
