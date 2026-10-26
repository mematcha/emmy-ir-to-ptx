"""Unit tests for PTXConstants (PTX ISA 9.4 constant formatting)."""

import math

import pytest

from compile import PTXConstants


@pytest.fixture
def constants():
    """Provide a fresh constants formatter for every test."""
    return PTXConstants()


class TestPTXIntegerConstants:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0"),
            (1, "1"),
            (42, "42"),
            (100, "100"),
            (-1, "-1"),
            (-42, "-42"),
        ],
    )
    def test_decimal(self, constants, value, expected):
        assert constants.integer(value) == expected

    @pytest.mark.parametrize(
        ("value", "base", "expected"),
        [
            (255, 16, "0xFF"),
            (42, 16, "0x2A"),
            (8, 8, "010"),
            (42, 8, "052"),
            (5, 2, "0b101"),
            (42, 2, "0b101010"),
            (-5, 16, "-0x5"),
            (-5, 2, "-0b101"),
        ],
    )
    def test_numeric_bases(self, constants, value, base, expected):
        assert constants.integer(value, base=base) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0U"),
            (42, "42U"),
            (255, "255U"),
            (2**63, "9223372036854775808U"),
            (2**64 - 1, "18446744073709551615U"),
        ],
    )
    def test_explicit_unsigned(self, constants, value, expected):
        assert constants.integer(value, unsigned=True) == expected

    def test_signed_64_bit_bounds(self, constants):
        assert constants.integer(-(2**63)) == str(-(2**63))
        assert constants.integer(2**63 - 1) == str(2**63 - 1)

    def test_auto_unsigned_past_signed_maximum(self, constants):
        assert constants.integer(2**63) == f"{2**63}U"

    @pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
    def test_out_of_range(self, constants, value):
        with pytest.raises(ValueError):
            constants.integer(value)

    @pytest.mark.parametrize("value", [-1, -42, -(2**63)])
    def test_negative_explicit_unsigned_rejected(self, constants, value):
        with pytest.raises(ValueError):
            constants.integer(value, unsigned=True)

    @pytest.mark.parametrize("value", [True, False, 3.14, "42", None, [42]])
    def test_invalid_integer_types(self, constants, value):
        with pytest.raises(TypeError):
            constants.integer(value)

    @pytest.mark.parametrize("base", [0, 3, 7, 12, 32])
    def test_invalid_base(self, constants, base):
        with pytest.raises(ValueError):
            constants.integer(42, base=base)

    def test_returns_string(self, constants):
        assert isinstance(constants.integer(42), str)

    def test_configured_ptx_version(self, constants):
        assert constants.version == "9.4"

    def test_integer_does_not_modify_version(self, constants):
        constants.integer(42)
        assert constants.version == "9.4"


class TestPTXFloatingConstants:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (3.14, "3.14"),
            (-2.5, "-2.5"),
            (1.5e10, "15000000000.0"),
        ],
    )
    def test_decimal_floating(self, constants, value, expected):
        assert constants.floating(value) == expected

    @pytest.mark.parametrize(
        ("value", "bits", "expected"),
        [
            (1.0, 32, "0f3F800000"),
            (-1.0, 32, "0fBF800000"),
            (0.0, 32, "0f00000000"),
            (-0.0, 32, "0f80000000"),
            (1.0, 64, "0d3FF0000000000000"),
            (-1.0, 64, "0dBFF0000000000000"),
        ],
    )
    def test_exact_ieee754_bits(self, constants, value, bits, expected):
        assert constants.float_bits(value, bits=bits) == expected

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_nonfinite_decimal_rejected(self, constants, value):
        with pytest.raises(ValueError):
            constants.floating(value)

    @pytest.mark.parametrize(
        ("value", "bits", "expected"),
        [
            (math.inf, 32, "0f7F800000"),
            (-math.inf, 32, "0fFF800000"),
            (math.inf, 64, "0d7FF0000000000000"),
            (-math.inf, 64, "0dFFF0000000000000"),
        ],
    )
    def test_infinity_exact_bits(self, constants, value, bits, expected):
        assert constants.float_bits(value, bits=bits) == expected

    @pytest.mark.parametrize("bits", [32, 64])
    def test_nan_exact_bits(self, constants, bits):
        literal = constants.float_bits(math.nan, bits=bits)
        assert literal.startswith("0f" if bits == 32 else "0d")
        assert len(literal) == (10 if bits == 32 else 18)
        payload = int(literal[2:], 16)
        if bits == 32:
            assert payload & 0x7F800000 == 0x7F800000
            assert payload & 0x007FFFFF != 0
        else:
            assert payload & 0x7FF0000000000000 == 0x7FF0000000000000
            assert payload & 0x000FFFFFFFFFFFFF != 0

    @pytest.mark.parametrize("bits", [0, 16, 128])
    def test_invalid_float_width(self, constants, bits):
        with pytest.raises(ValueError):
            constants.float_bits(1.0, bits=bits)

    @pytest.mark.parametrize("value", [1, "3.14", None, True])
    def test_invalid_floating_type(self, constants, value):
        with pytest.raises(TypeError):
            constants.floating(value)

    @pytest.mark.parametrize("value", [1, "1.0", None, False])
    def test_invalid_float_bits_type(self, constants, value):
        with pytest.raises(TypeError):
            constants.float_bits(value)

    def test_f32_overflow_rejected(self, constants):
        with pytest.raises(ValueError):
            constants.float_bits(1e100, bits=32)

    def test_default_bits_is_f32(self, constants):
        assert constants.float_bits(1.0) == "0f3F800000"


class TestPTXPredicateConstants:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, "0"), (1, "1"), (-1, "1"), (5, "1"), (-100, "1")],
    )
    def test_predicate_truth_values(self, constants, value, expected):
        assert constants.predicate(value) == expected

    @pytest.mark.parametrize("value", [True, False, 3.14, "1", None])
    def test_invalid_predicate_types(self, constants, value):
        with pytest.raises(TypeError):
            constants.predicate(value)

    @pytest.mark.parametrize("value", [2**64, -(2**63) - 1])
    def test_predicate_range_validation(self, constants, value):
        with pytest.raises(ValueError):
            constants.predicate(value)
