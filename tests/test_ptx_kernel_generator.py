
import pytest

from compile import EmmyKernelIR, PTXKernelGenerator


@pytest.fixture
def kernel():
    """Construct the metadata shape of r01_array_sum."""

    selected = {
        "node": "sum_1",
        "entry": "k_sum_1_reduce",
        "arg_order": ["x", "sum_1"],
        "grid": [1, 1, 1],
        "block": [128, 1, 1],
        "smem_bytes": 0,
        "args": [
            {
                "name": "x",
                "dtype": "f32",
                "shape": [1024],
                "seed": 0,
            },
            {
                "name": "sum_1",
                "dtype": "f32",
                "shape": [1],
                "output": True,
            },
        ],
    }

    return EmmyKernelIR(
        name="k_sum_1_reduce",
        body=(),
        graph=object(),
        node_key="sum_1",
        inputs=("x",),
        outputs=("sum_1",),
        arg_order=("x", "sum_1"),
        launch={
            "grid": [1, 1, 1],
            "block": [128, 1, 1],
            "smem_bytes": 0,
        },
        metadata={"selected_kernel": selected},
    )


def test_kernel_signature(kernel):
    generator = PTXKernelGenerator(kernel)

    ptx = generator.render_entry()

    assert ".visible .entry k_sum_1_reduce(" in ptx
    assert ".param .u64 param_x" in ptx
    assert ".param .u64 param_sum_1" in ptx
    assert ptx.endswith("    ret;\n}\n")


def test_parameter_order(kernel):
    generator = PTXKernelGenerator(kernel)
    ptx = generator.render_entry()

    assert ptx.index("param_x") < ptx.index("param_sum_1")


def test_parameter_lookup(kernel):
    generator = PTXKernelGenerator(kernel)

    assert generator.parameter("x") == "param_x"
    assert generator.parameter("sum_1") == "param_sum_1"

    with pytest.raises(KeyError):
        generator.parameter("unknown")


def test_register_allocation(kernel):
    generator = PTXKernelGenerator(kernel)

    r0 = generator.register(".u32")
    r1 = generator.register(".u32")
    f0 = generator.register(".f32")
    p0 = generator.register(".pred")

    assert (r0, r1, f0, p0) == (
        "%r0", "%r1", "%f0", "%p0"
    )

    ptx = generator.render_entry()

    assert ".reg .u32 %r0;" in ptx
    assert ".reg .u32 %r1;" in ptx
    assert ".reg .f32 %f0;" in ptx
    assert ".reg .pred %p0;" in ptx


def test_64bit_register(kernel):
    generator = PTXKernelGenerator(kernel)

    reg = generator.register(".u64")

    assert reg == "%rd0"
    assert ".reg .u64 %rd0;" in generator.render_entry()


def test_shared_memory_declaration(kernel):
    generator = PTXKernelGenerator(kernel)

    generator.shared(
        "scratch",
        ".f32",
        length=32,
        align=16,
    )

    ptx = generator.render_entry()

    assert ".shared .align 16 .f32 scratch[32];" in ptx


@pytest.mark.parametrize("length", [0, -1, 1.5, True])
def test_invalid_shared_length(kernel, length):
    generator = PTXKernelGenerator(kernel)

    with pytest.raises(ValueError):
        generator.shared("scratch", ".f32", length=length)


def test_thread_index_1d(kernel):
    generator = PTXKernelGenerator(kernel)

    index = generator.thread_index_1d()
    ptx = generator.render_entry()

    assert index == "%r4"
    assert "mov.u32 %r0, %tid.x;" in ptx
    assert "mov.u32 %r1, %ctaid.x;" in ptx
    assert "mov.u32 %r2, %ntid.x;" in ptx
    assert "mul.lo.u32 %r3, %r1, %r2;" in ptx
    assert "add.u32 %r4, %r3, %r0;" in ptx


def test_emit_instruction(kernel):
    generator = PTXKernelGenerator(kernel)

    r0 = generator.register(".f32")
    r1 = generator.register(".f32")
    r2 = generator.register(".f32")

    generator.emit(f"add.f32 {r2}, {r0}, {r1};")

    ptx = generator.render_entry()

    assert "add.f32 %f2, %f0, %f1;" in ptx


@pytest.mark.parametrize(
    "instruction",
    [
        "",
        "add.f32 %f0, %f1, %f2",
        "ret;\ntrap;",
        "mov.u32 %r0, 1; ret;",
        "ret;\n",
    ],
)
def test_invalid_instruction(kernel, instruction):
    generator = PTXKernelGenerator(kernel)

    with pytest.raises(ValueError):
        generator.emit(instruction)


def test_no_duplicate_ret(kernel):
    generator = PTXKernelGenerator(kernel)

    generator.emit("ret;")

    ptx = generator.render_entry()

    assert ptx.count("ret;") == 1


def test_no_duplicate_ptx_header(kernel):
    generator = PTXKernelGenerator(kernel)

    ptx = generator.render_entry()

    assert ".version" not in ptx
    assert ".target" not in ptx


def test_missing_metadata(kernel):
    kernel.metadata.clear()

    with pytest.raises(ValueError, match="selected_kernel"):
        PTXKernelGenerator(kernel)


def test_inconsistent_arguments(kernel):
    kernel.metadata["selected_kernel"]["args"] = [
        {"name": "x", "dtype": "f32", "shape": [1024]}
    ]

    with pytest.raises(ValueError, match="do not match"):
        PTXKernelGenerator(kernel)
