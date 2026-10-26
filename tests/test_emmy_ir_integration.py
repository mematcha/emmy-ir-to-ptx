
from pathlib import Path

import pytest

from compile import EmmyIRParser, EmmyKernelIR


# tests/ is one level below the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IR_PATH = (
    PROJECT_ROOT
    / "problems"
    / "r01_array_sum"
    / "k_sum_1_reduce.ir.json"
)


def test_real_emmy_ir_deserialization():
    """Deserialize an official Emmy Kernel IR file."""

    assert IR_PATH.is_file(), (
        f"Missing example IR: {IR_PATH}"
    )

    kernel = EmmyIRParser(
        IR_PATH,
        require_meta=True,
    ).parse()

    assert isinstance(kernel, EmmyKernelIR)
    assert kernel.name == "k_sum_1_reduce"
    assert kernel.body is not None
    assert kernel.graph is not None

    # Argument ordering must come from meta.json.
    assert kernel.arg_order is not None
    assert len(kernel.arg_order) > 0
    assert len(set(kernel.arg_order)) == len(
        kernel.arg_order
    )

    print("\nKernel:", kernel.name)
    print("Arguments:", kernel.arg_order)
    print("Body type:", type(kernel.body).__name__)
    print("Launch metadata:", kernel.launch)


def test_real_emmy_kernel_body():
    """Verify that we obtained a native Emmy statement tree."""

    kernel = EmmyIRParser(IR_PATH).parse()

    print("\nKernel body:", repr(kernel.body)[:1000])

    assert kernel.body is not None



def test_real_metadata_present():
    """Verify official per-kernel metadata is accessible."""

    kernel = EmmyIRParser(
        IR_PATH,
        require_meta=True,
    ).parse()

    assert "kernels" in kernel.metadata
    assert "selected_kernel" in kernel.metadata

    selected = kernel.metadata["selected_kernel"]

    assert "arg_order" in selected
    assert selected["entry"] == kernel.name
    assert selected["node"] == kernel.node_key

    assert tuple(selected["arg_order"]) == kernel.arg_order
    assert selected["block"] == [128, 1, 1]
    assert selected["smem_bytes"] == 0

