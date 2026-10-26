
import json
import sys
import types

import pytest

from compile import EmmyIRParser, EmmyKernelIR


def fake_emmy(monkeypatch, *, kernels=1, error=None):
    """Simulate Emmy Graph objects for isolated parser testing."""

    class KernelOp:
        def __init__(self, name="sample_kernel"):
            self.name = name
            self.body = ["opaque native Emmy statement"]

    class Node:
        def __init__(self, op):
            self.op = op
            self.inputs = ["a", "b"]
            self.outputs = [
                types.SimpleNamespace(name="out")
            ]

    class Graph:
        @classmethod
        def from_dict(cls, data):
            if error:
                raise ValueError(error)

            graph = cls()
            graph.nodes = {
                f"node{i}": Node(KernelOp())
                for i in range(kernels)
            }
            return graph

    emmy = types.ModuleType("emmy")
    compiler = types.ModuleType("emmy.compiler")
    graph_module = types.ModuleType("emmy.compiler.graph")

    graph_module.Graph = Graph

    monkeypatch.setitem(sys.modules, "emmy", emmy)
    monkeypatch.setitem(sys.modules, "emmy.compiler", compiler)
    monkeypatch.setitem(
        sys.modules, "emmy.compiler.graph", graph_module
    )


def write_ir(tmp_path, data=None):
    """Create a temporary Emmy IR JSON file."""

    path = tmp_path / "sample.ir.json"

    path.write_text(
        json.dumps({"nodes": []} if data is None else data),
        encoding="utf-8",
    )

    return path


def test_parse_native_graph(tmp_path, monkeypatch):
    fake_emmy(monkeypatch)

    parsed = EmmyIRParser(write_ir(tmp_path)).parse()

    assert isinstance(parsed, EmmyKernelIR)
    assert parsed.name == "sample_kernel"
    assert parsed.inputs == ("a", "b")
    assert parsed.outputs == ("out",)
    assert parsed.body == ["opaque native Emmy statement"]
    assert parsed.arg_order is None




def test_missing_required_metadata(tmp_path, monkeypatch):
    fake_emmy(monkeypatch)

    with pytest.raises(
        ValueError,
        match="Missing required metadata",
    ):
        EmmyIRParser(
            write_ir(tmp_path),
            require_meta=True,
        ).parse()



@pytest.mark.parametrize("count", [0, 2])
def test_kernel_count(tmp_path, monkeypatch, count):
    fake_emmy(monkeypatch, kernels=count)

    with pytest.raises(
        ValueError,
        match="exactly one KernelOp",
    ):
        EmmyIRParser(write_ir(tmp_path)).parse()


def test_deserialization_error(tmp_path, monkeypatch):
    fake_emmy(monkeypatch, error="bad Emmy graph")

    with pytest.raises(
        ValueError,
        match="could not deserialize",
    ):
        EmmyIRParser(write_ir(tmp_path)).parse()


def test_malformed_json(tmp_path):
    path = tmp_path / "broken.ir.json"

    path.write_text(
        "{bad json",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid JSON"):
        EmmyIRParser(path).parse()


def test_missing_ir_file(tmp_path):
    with pytest.raises(ValueError, match="Cannot read"):
        EmmyIRParser(
            tmp_path / "missing.ir.json"
        ).parse()


def test_metadata_order_is_authoritative(tmp_path, monkeypatch):
    fake_emmy(monkeypatch)

    path = write_ir(tmp_path)

    metadata = make_metadata(
        arg_order=("out", "b", "a"),
        rtol=1e-5,
    )

    (tmp_path / "meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    parsed = EmmyIRParser(
        path,
        require_meta=True,
    ).parse()

    assert parsed.arg_order == ("out", "b", "a")
    assert parsed.launch["block"] == [128, 1, 1]
    assert parsed.launch["smem_bytes"] == 0
    assert parsed.metadata["rtol"] == 1e-5
    assert (
        parsed.metadata["selected_kernel"]["entry"]
        == "sample_kernel"
    )


def test_explicit_metadata_path(tmp_path, monkeypatch):
    fake_emmy(monkeypatch)

    path = write_ir(tmp_path)
    meta = tmp_path / "other.json"

    meta.write_text(
        json.dumps(
            make_metadata(arg_order=("a", "out"))
        ),
        encoding="utf-8",
    )

    parsed = EmmyIRParser(
        path,
        meta_path=meta,
        require_meta=True,
    ).parse()

    assert parsed.arg_order == ("a", "out")


def test_missing_arg_order(tmp_path, monkeypatch):
    fake_emmy(monkeypatch)

    path = write_ir(tmp_path)

    (tmp_path / "meta.json").write_text(
        json.dumps(
            make_metadata(include_arg_order=False)
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="arg_order"):
        EmmyIRParser(
            path,
            require_meta=True,
        ).parse()


@pytest.mark.parametrize(
    "order",
    [
        ["a", "a"],
        ["a", 3],
        "a",
    ],
)
def test_invalid_arg_order(
    tmp_path, monkeypatch, order
):
    fake_emmy(monkeypatch)

    path = write_ir(tmp_path)

    (tmp_path / "meta.json").write_text(
        json.dumps(
            make_metadata(arg_order=order)
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="arg_order"):
        EmmyIRParser(
            path,
            require_meta=True,
        ).parse()



def test_nonobject_json(tmp_path):
    with pytest.raises(
        ValueError,
        match="Expected a JSON object",
    ):
        EmmyIRParser(
            write_ir(tmp_path, [])
        ).parse()



def make_metadata(
    *,
    arg_order=("a", "out"),
    include_arg_order=True,
    **extra,
):
    """Build official-style Emmy metadata for the fake kernel."""

    record = {
        "node": "node0",
        "entry": "sample_kernel",
        "grid": [1, 1, 1],
        "block": [128, 1, 1],
        "smem_bytes": 0,
    }

    if include_arg_order:
        # Preserve invalid test inputs without converting them.
        record["arg_order"] = arg_order

    return {
        "id": "test_problem",
        "kernels": [record],
        **extra,
    }

