"""Integration tests for PTXOperands defined in compile.py."""
import pytest
from compile import PTXOperands, PTXType, PTXEmitter


def test_register():
    r = PTXOperands.Register('%f1', PTXType('.f32'))
    assert str(r) == '%f1'


def test_immediate():
    assert str(PTXOperands.Immediate('1.0', PTXType('.f32'))) == '1.0'


def test_address():
    assert str(PTXOperands.Address('%rd1', offset=16)) == '[%rd1+16]'
    assert str(PTXOperands.Address('%rd1', offset=-8)) == '[%rd1-8]'
    assert str(PTXOperands.Address('buffer', kind='symbol')) == '[buffer]'


def test_vector():
    a = PTXOperands.Register('%r1', PTXType('.b32'))
    b = PTXOperands.Register('%r2', PTXType('.b32'))
    assert str(PTXOperands.Vector((a, b), destination=True)) == '{%r1, %r2}'


def test_array_component_symbol():
    assert str(PTXOperands.ArrayElement('array', 2)) == 'array[2]'
    assert str(PTXOperands.Component('vector', 'x')) == 'vector.x'
    assert PTXOperands.Symbol('loop', 'label').valid_for('bra')


def test_types_and_alignment():
    assert PTXOperands.compatible_types(PTXType('.s32'), PTXType('.u32'))
    assert PTXOperands.access_alignment_ok(16, 4)
    assert not PTXOperands.access_alignment_ok(18, 4)
    assert PTXOperands.conversion_opcode(PTXType('.f32'), PTXType('.f64'), rounding='.rn') == 'cvt.rn.f32.f64'


def test_reject_invalid_address():
    with pytest.raises(ValueError):
        PTXOperands.Address('%rd1', offset=1 << 32)


def test_emitter_integration():
    e = PTXEmitter()
    e._emit_instruction('ld.global.f32',
        PTXOperands.Register('%f1', PTXType('.f32')),
        PTXOperands.Address('%rd1', 16))
    assert 'ld.global.f32 %f1, [%rd1+16];' in e.ptx_code
