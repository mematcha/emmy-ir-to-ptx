
"""Tests for PTX ISA 9.4 Section 5.2 — Types."""

import pytest

from compile import PTXType, PTXTypeInfo


class TestPTXFundamentalTypes:

    @pytest.mark.parametrize(
        "name,bits,category",
        [
            (".s8", 8, "signed"),
            (".s16", 16, "signed"),
            (".s32", 32, "signed"),
            (".s64", 64, "signed"),
            (".u8", 8, "unsigned"),
            (".u16", 16, "unsigned"),
            (".u32", 32, "unsigned"),
            (".u64", 64, "unsigned"),
            (".f16", 16, "float"),
            (".f16x2", 32, "float"),
            (".f32", 32, "float"),
            (".f64", 64, "float"),
            (".b8", 8, "bits"),
            (".b16", 16, "bits"),
            (".b32", 32, "bits"),
            (".b64", 64, "bits"),
            (".b128", 128, "bits"),
            (".pred", 1, "predicate"),
        ],
    )
    def test_fundamental_types(self, name, bits, category):
        dtype = PTXType(name)

        assert dtype.bits == bits
        assert dtype.category == category
        assert dtype.fundamental is True
        assert dtype.require_fundamental() == name

    def test_normalization(self):
        assert PTXType("f32").name == ".f32"

    def test_predicate_width(self):
        assert PTXType(".pred").bits == 1

    def test_f16x2_fundamental(self):
        dtype = PTXType(".f16x2")

        assert dtype.elements == 2
        assert dtype.element_type == ".f16"
        assert dtype.register_type == ".f16x2"


class TestPTXAlternateFormats:

    @pytest.mark.parametrize(
        "name,register_type",
        [
            (".bf16", ".b16"),
            (".tf32", ".b32"),
            (".e4m3", ".b8"),
            (".e5m2", ".b8"),
            (".e2m1", ".b8"),
            (".e2m3", ".b8"),
            (".e3m2", ".b8"),
            (".ue8m0", ".b8"),
            (".ue4m3", ".b8"),
            (".ue5m3", ".b8"),
        ],
    )
    def test_alternate_float_storage(
        self, name, register_type
    ):
        dtype = PTXType(name)

        assert dtype.fundamental is False
        assert dtype.register_type == register_type

    def test_fixed_point(self):
        dtype = PTXType(".s2f6")

        assert dtype.bits == 8
        assert dtype.category == "fixed_point"
        assert dtype.register_type == ".b8"

    def test_nonfundamental_rejected_for_declaration(self):
        with pytest.raises(ValueError):
            PTXType(".bf16").require_fundamental()


class TestPTXPackedTypes:

    @pytest.mark.parametrize(
        "name,elements,bits,register_type",
        [
            (".f32x2", 2, 64, ".b64"),
            (".bf16x2", 2, 32, ".b32"),
            (".e4m3x2", 2, 16, ".b16"),
            (".e2m1x2", 2, 8, ".b8"),
            (".e4m3x4", 4, 32, ".b32"),
            (".e2m1p4x4", 4, 32, ".b32"),
            (".u16x2", 2, 32, ".b32"),
            (".s16x2", 2, 32, ".b32"),
            (".u8x4", 4, 32, ".b32"),
            (".s8x4", 4, 32, ".b32"),
        ],
    )
    def test_packed_types(
        self, name, elements, bits, register_type
    ):
        dtype = PTXType(name)

        assert dtype.elements == elements
        assert dtype.bits == bits
        assert dtype.register_type == register_type
        assert dtype.fundamental is False


class TestPTXCompatibility:

    @pytest.mark.parametrize(
        "left,right,expected",
        [
            (".s32", ".u32", True),
            (".u64", ".s64", True),
            (".b32", ".f32", True),
            (".b32", ".s32", True),
            (".f32", ".f32", True),
            (".f32", ".f64", False),
            (".s16", ".s32", False),
            (".pred", ".u8", False),
            (".bf16", ".b16", False),
        ],
    )
    def test_compatibility(self, left, right, expected):
        assert (
            PTXType(left).compatible_with(PTXType(right))
            is expected
        )

    def test_invalid_compatibility_argument(self):
        with pytest.raises(TypeError):
            PTXType(".f32").compatible_with(".f32")


class TestPTXRestrictions:

    @pytest.mark.parametrize(
        "opcode",
        ["ld.global.u8", "st.global.u8",
         "add.u8", "cvt.u8.u32"],
    )
    def test_allowed_u8_operations(self, opcode):
        assert PTXType(".u8").allowed_for_instruction(opcode)

    @pytest.mark.parametrize(
        "opcode",
        ["mul.u8", "div.u8", "sqrt.u8"],
    )
    def test_disallowed_u8_operations(self, opcode):
        assert not PTXType(".u8").allowed_for_instruction(
            opcode
        )

    def test_half_precision_requires_specific_rules(self):
        with pytest.raises(NotImplementedError):
            PTXType(".f16").allowed_for_instruction("add.f16")

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            PTXType(".imaginary")

    def test_invalid_input(self):
        with pytest.raises(TypeError):
            PTXType(32)

    def test_repr(self):
        assert repr(PTXType("f32")) == "PTXType('.f32')"

    def test_metadata_is_frozen(self):
        from dataclasses import FrozenInstanceError

        dtype = PTXType(".f32")

        with pytest.raises(FrozenInstanceError):
            dtype.info.bits = 64

    def test_metadata_type(self):
        assert isinstance(PTXType(".f32").info, PTXTypeInfo)
