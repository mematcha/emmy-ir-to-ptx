"""PTX ISA 9.4 chapter 9: PTXInstructionSet unit tests.

Place PTXInstructionSet in compile.py, import it here, and run:
    python -m pytest tests/test_ptx_instruction_set.py -v

These are code-generation/validation tests, not GPU or ptxas tests.
"""

import pytest

from compile import PTXInstructionSet, PTXOperands, PTXType


@pytest.fixture
def isa():
    return PTXInstructionSet(target="sm_89", ptx_version="9.4")


def reg(name, dtype):
    return PTXOperands.Register(name, PTXType(dtype))


class TestCoreArithmetic:
    @pytest.mark.parametrize(
        "dtype,result",
        [
            (".u32", "add.u32 %r1, %r2, 5;"),
            (".s64", "add.s64 %r1, %r2, 5;"),
            (".f32", "add.f32 %r1, %r2, 5;"),
            (".f64", "add.f64 %r1, %r2, 5;"),
        ],
    )
    def test_add(self, isa, dtype, result):
        assert isa.add(dtype, "%r1", "%r2", "5") == result

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            (".s32", "sub.s32 %a, %b, %c;"),
            (".f32", "sub.f32 %a, %b, %c;"),
        ],
    )
    def test_sub(self, isa, dtype, expected):
        assert isa.sub(dtype, "%a", "%b", "%c") == expected

    def test_float_rounding(self, isa):
        assert isa.add(".f32", "%a", "%b", "%c", rounding="rz") == (
            "add.rz.f32 %a, %b, %c;"
        )

    def test_integer_rounding_rejected(self, isa):
        with pytest.raises(ValueError, match="Rounding"):
            isa.add(".u32", "%a", "%b", "%c", rounding="rn")

    def test_reject_8bit_add(self, isa):
        with pytest.raises(ValueError, match="does not support"):
            isa.add(".u8", "%a", "%b", "%c")

    def test_reject_packed_add_for_sm89(self, isa):
        with pytest.raises(ValueError):
            isa.add(".u16x2", "%d", "%a", "%b")

    def test_mul_default_int_low_half(self, isa):
        assert isa.mul(".u32", "%r1", "%r2", "%r3") == (
            "mul.lo.u32 %r1, %r2, %r3;"
        )

    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("hi", "mul.hi.s32 %r1, %r2, %r3;"),
            ("lo", "mul.lo.s32 %r1, %r2, %r3;"),
        ],
    )
    def test_integer_mul_modes(self, isa, mode, expected):
        assert isa.mul(".s32", "%r1", "%r2", "%r3", mode=mode) == expected

    def test_mul_wide(self, isa):
        assert isa.mul(".u32", "%rd1", "%r2", "%r3", mode="wide") == (
            "mul.wide.u32 %rd1, %r2, %r3;"
        )

    def test_mul_wide_typed_destination(self, isa):
        result = isa.mul(
            ".u32", reg("%rd1", ".u64"), reg("%r2", ".u32"),
            reg("%r3", ".u32"), mode="wide",
        )
        assert result == "mul.wide.u32 %rd1, %r2, %r3;"

    def test_mul_wide_wrong_destination_type(self, isa):
        with pytest.raises(ValueError, match="expected"):
            isa.mul(".u32", reg("%r1", ".u32"), "%r2", "%r3", mode="wide")

    def test_mul_wide_64_disallowed(self, isa):
        with pytest.raises(ValueError, match="16/32"):
            isa.mul(".u64", "%rd1", "%a", "%b", mode="wide")

    def test_float_mul(self, isa):
        assert isa.mul(".f32", "%f1", "%f2", "%f3", rounding="rn") == (
            "mul.rn.f32 %f1, %f2, %f3;"
        )

    def test_float_mul_reject_integer_modes(self, isa):
        with pytest.raises(ValueError):
            isa.mul(".f32", "%f1", "%f2", "%f3", mode="lo")

    def test_fma_requires_float_and_four_operands(self, isa):
        assert isa.fma(".f32", "%d", "%a", "%b", "%c") == (
            "fma.rn.f32 %d, %a, %b, %c;"
        )
        with pytest.raises(ValueError):
            isa.fma(".s32", "%d", "%a", "%b", "%c")

    def test_fma_invalid_rounding(self, isa):
        with pytest.raises(ValueError):
            isa.fma(".f64", "%d", "%a", "%b", "%c", rounding="bogus")

    def test_typed_register_validation(self, isa):
        assert isa.add(
            PTXType(".f32"), reg("%f1", ".f32"),
            reg("%f2", ".f32"), reg("%f3", ".f32")
        ) == "add.f32 %f1, %f2, %f3;"
        with pytest.raises(ValueError, match="expected"):
            isa.add(".f32", reg("%r1", ".u64"), "%f2", "%f3")

    def test_invalid_destination_literal(self, isa):
        with pytest.raises(ValueError):
            isa.add(".u32", "7", "%r2", "%r3")

    def test_reject_unsafe_operand(self, isa):
        with pytest.raises(ValueError):
            isa.add(".u32", "%r1", "%r2; trap;", "%r3")


class TestMovesAndConversion:
    def test_mov(self, isa):
        assert isa.mov(".u32", "%r1", "%tid.x") == "mov.u32 %r1, %tid.x;"
        assert isa.mov(".f32", "%f1", "0f3F800000") == "mov.f32 %f1, 0f3F800000;"

    def test_mov_predicate(self, isa):
        assert isa.mov(".pred", "%p1", "1") == "mov.pred %p1, 1;"

    def test_mov_reject_u8(self, isa):
        with pytest.raises(ValueError):
            isa.mov(".u8", "%r1", "1")

    def test_cvt_widen_int(self, isa):
        assert isa.cvt(".u64", ".u32", "%rd1", "%r1") == (
            "cvt.u64.u32 %rd1, %r1;"
        )

    def test_cvt_float_to_int_rounds_toward_zero(self, isa):
        assert isa.cvt(".s32", ".f32", "%r1", "%f1") == (
            "cvt.rzi.s32.f32 %r1, %f1;"
        )

    def test_cvt_int_to_float(self, isa):
        assert isa.cvt(".f32", ".s32", "%f1", "%r1") == (
            "cvt.rn.f32.s32 %f1, %r1;"
        )

    def test_cvt_float_widen(self, isa):
        assert isa.cvt(".f64", ".f32", "%fd1", "%f1") == (
            "cvt.f64.f32 %fd1, %f1;"
        )

    def test_cvt_float_narrow(self, isa):
        assert isa.cvt(".f32", ".f64", "%f1", "%fd1") == (
            "cvt.rn.f32.f64 %f1, %fd1;"
        )

    def test_cvt_wrong_rounding_family(self, isa):
        with pytest.raises(ValueError):
            isa.cvt(".s32", ".f32", "%r1", "%f1", rounding="rn")

    def test_cvt_integer_rounding_invalid(self, isa):
        with pytest.raises(ValueError):
            isa.cvt(".s64", ".s32", "%r1", "%r2", rounding="rzi")


class TestMemoryAndAddressing:
    def test_load_global(self, isa):
        assert isa.load("global", ".f32", "%f1", PTXOperands.Address("%rd1", 16)) == (
            "ld.global.f32 %f1, [%rd1+16];"
        )

    def test_load_param(self, isa):
        assert isa.load(".param", ".u64", "%rd1", "[pointer_arg]") == (
            "ld.param.u64 %rd1, [pointer_arg];"
        )

    def test_load_const(self, isa):
        assert isa.load("const", ".u32", "%r1", "[constvar]") == (
            "ld.const.u32 %r1, [constvar];"
        )

    def test_load_narrow_to_wide_reg(self, isa):
        assert isa.load("global", ".u8", reg("%r1", ".u32"), "[%rd1]") == (
            "ld.global.u8 %r1, [%rd1];"
        )

    def test_load_wrong_register_type(self, isa):
        with pytest.raises(ValueError, match="expected"):
            isa.load("global", ".f32", reg("%rd1", ".u64"), "[%rd2]")

    def test_store_global(self, isa):
        assert isa.store("global", ".f32", "[%rd1]", "%f1") == (
            "st.global.f32 [%rd1], %f1;"
        )

    def test_store_shared_local(self, isa):
        assert isa.store("shared", ".b32", "[%r1]", "%r2") == (
            "st.shared.b32 [%r1], %r2;"
        )
        assert isa.load("local", ".s32", "%r1", "[%rd1+4]") == (
            "ld.local.s32 %r1, [%rd1+4];"
        )

    @pytest.mark.parametrize("space", ["const", "param", "reg", "sreg"])
    def test_illegal_store_spaces(self, isa, space):
        with pytest.raises(ValueError):
            isa.store(space, ".u32", "[%rd1]", "%r1")

    def test_bad_memory_type(self, isa):
        with pytest.raises(ValueError):
            isa.load("global", ".pred", "%p1", "[%rd1]")

    def test_address_must_be_bracketed(self, isa):
        with pytest.raises(ValueError):
            isa.load("global", ".u32", "%r1", "%rd1")

    def test_address_injection_rejected(self, isa):
        with pytest.raises(ValueError):
            isa.load("global", ".u32", "%r1", "[%rd1]; trap;")

    def test_cvta_global(self, isa):
        assert isa.cvta("global", "%rd1", "%rd2") == (
            "cvta.global.u64 %rd1, %rd2;"
        )

    def test_cvta_to_shared(self, isa):
        assert isa.cvta("shared", "%r1", "%r2", size=32, to=True) == (
            "cvta.to.shared.u32 %r1, %r2;"
        )

    def test_cvta_bad_size(self, isa):
        with pytest.raises(ValueError):
            isa.cvta("shared", "%r1", "%r2", size=16)

    def test_cvta_invalid_param_to(self, isa):
        with pytest.raises(ValueError):
            isa.cvta("param", "%rd1", "%rd2", to=True)

    def test_existing_state_space_objects(self, isa):
        from compile import StateSpace
        assert isa.load(StateSpace(".global"), ".u32", "%r1", "[%rd1]") == (
            "ld.global.u32 %r1, [%rd1];"
        )


class TestConfiguration:
    def test_default_config(self):
        builder = PTXInstructionSet()
        assert builder.target == "sm_89"
        assert builder.ptx_version == "9.4"

    @pytest.mark.parametrize("target", ["89", "sm_", "sm_90a", None, "sm_9x"])
    def test_invalid_target(self, target):
        with pytest.raises(ValueError):
            PTXInstructionSet(target=target)

    def test_invalid_version(self):
        with pytest.raises(ValueError):
            PTXInstructionSet(ptx_version="PTX9.4")

class TestVectorMemory:
    def test_vector_load(self, isa):
        dst = PTXOperands.Vector((
            reg('%f0', '.f32'), reg('%f1', '.f32'),
            reg('%f2', '.f32'), reg('%f3', '.f32'),
        ), destination=True)
        assert isa.load_vector('global', '.f32', dst, '[%rd1]') == (
            'ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd1];'
        )

    def test_vector_store(self, isa):
        src = PTXOperands.Vector((reg('%r1', '.b32'), reg('%r2', '.b32')))
        assert isa.store_vector('shared', '.b32', '[%r0]', src) == (
            'st.shared.v2.b32 [%r0], {%r1, %r2};'
        )

    def test_vector_load_requires_destination_tuple(self, isa):
        src = PTXOperands.Vector((reg('%f0', '.f32'), reg('%f1', '.f32')))
        with pytest.raises(ValueError, match='destination'):
            isa.load_vector('global', '.f32', src, '[%rd1]')

    def test_vector_over_128_bits_rejected_for_sm89(self, isa):
        registers = tuple(reg(f'%rd{i}', '.b64') for i in range(4))
        src = PTXOperands.Vector(registers)
        with pytest.raises(ValueError, match='sm_100'):
            isa.store_vector('global', '.b64', '[%rd1]', src)

    def test_vector_tuple_types_must_match(self, isa):
        src = PTXOperands.Vector((reg('%r1', '.u32'), reg('%r2', '.u32')))
        with pytest.raises(ValueError, match='expected'):
            isa.store_vector('global', '.f64', '[%rd1]', src)


class TestArchitectureRestrictions:
    def test_fma_sm10_rejected(self):
        with pytest.raises(ValueError, match='sm_20'):
            PTXInstructionSet(target='sm_10').fma('.f32', '%d', '%a', '%b', '%c')

    def test_cvta_sm10_rejected(self):
        with pytest.raises(ValueError, match='sm_20'):
            PTXInstructionSet(target='sm_10').cvta('global', '%r1', '%r2')
