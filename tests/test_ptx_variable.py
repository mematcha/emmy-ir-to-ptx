"""Tests for the progressive Section 5.4 PTXVariable implementation."""
import pytest
from compile import PTXVariable


@pytest.mark.parametrize('space, typ, name, expected', [
    ('reg', 'f32', '%f1', '.reg .f32 %f1;'),
    ('.global', '.u32', 'counter', '.global .u32 counter;'),
    ('.shared', '.b8', 'scratch', '.shared .b8 scratch;'),
    ('.local', '.u64', 'tmp', '.local .u64 tmp;'),
    ('.param', '.b64', 'arg0', '.param .b64 arg0;'),
    ('.reg', '.pred', '%p0', '.reg .pred %p0;'),
])
def test_scalar(space, typ, name, expected):
    assert PTXVariable(name, space, typ).declaration() == expected


@pytest.mark.parametrize('space', ['global', 'shared', 'local', 'const', 'param'])
def test_predicate_is_register_only(space):
    with pytest.raises(ValueError, match='Predicate'):
        PTXVariable('flag', space, 'pred')


def test_vector():
    assert PTXVariable('v', 'global', 'f32', vector=4).declaration() == '.global .v4 .f32 v;'


def test_vector_too_wide():
    with pytest.raises(ValueError, match='128'):
        PTXVariable('v', 'reg', 'f64', vector=4)


def test_predicate_vector_rejected():
    with pytest.raises(ValueError):
        PTXVariable('p', 'reg', 'pred', vector=2)


@pytest.mark.parametrize('length', [0, 3, -1, True])
def test_invalid_vector_length(length):
    with pytest.raises(ValueError):
        PTXVariable('v', 'reg', 'f32', vector=length)


def test_arrays():
    assert PTXVariable('tile', 'shared', 'u16', dimensions=(19, 19)).declaration() == '.shared .u16 tile[19][19];'


def test_inferred_dimension():
    assert PTXVariable('a', 'global', 'u32', dimensions=(None,), initializer=[1, 2, 3]).declaration() == '.global .u32 a[] = {1, 2, 3};'


def test_incomplete_initializer():
    assert PTXVariable('vals', 'const', 'f32', dimensions=(4,), initializer=[0.25, 0.5]).declaration() == '.const .f32 vals[4] = {0.25, 0.5};'


def test_multidimensional_initializer():
    assert PTXVariable('x', 'global', 's32', dimensions=(3, 2), initializer=[[1, 2], [3]]).declaration() == '.global .s32 x[3][2] = {{1, 2}, {3}};'


def test_scalar_initializer():
    assert PTXVariable('answer', 'global', 'u32', initializer=42).declaration() == '.global .u32 answer = 42;'


@pytest.mark.parametrize('space', ['reg', 'local', 'shared', 'param'])
def test_illegal_initializer_space(space):
    with pytest.raises(ValueError, match='Only .global'):
        PTXVariable('x', space, 'u32', initializer=1)


@pytest.mark.parametrize('typ', ['f16', 'f16x2'])
def test_disallowed_initializer_type(typ):
    with pytest.raises(ValueError, match='Initializer not supported'):
        PTXVariable('x', 'global', typ, initializer=1)


@pytest.mark.parametrize('alignment', [0, 3, -4, True])
def test_invalid_align(alignment):
    with pytest.raises(ValueError, match='power of two'):
        PTXVariable('x', 'global', 'b8', align=alignment)


def test_alignment():
    assert PTXVariable('x', 'shared', 'b8', dimensions=(128,), align=16).declaration() == '.shared .align 16 .b8 x[128];'


def test_register_alignment_rejected():
    with pytest.raises(ValueError, match='addressable'):
        PTXVariable('%r0', 'reg', 'u32', align=16)


def test_parameterized_names():
    assert PTXVariable('%r', 'reg', 'b32', count=8).declaration() == '.reg .b32 %r<8>;'


@pytest.mark.parametrize('options', [{'dimensions': (8,)}, {'initializer': 1}, {'vector': 2}])
def test_parameterized_names_restrictions(options):
    with pytest.raises(ValueError):
        PTXVariable('values', 'global', 'u32', count=8, **options)


def test_managed():
    assert PTXVariable('g', 'global', 's32', attribute='managed').declaration() == '.global .attribute(.managed) .s32 g;'


def test_unified_not_available_on_sm89():
    with pytest.raises(ValueError, match='sm_90'):
        PTXVariable('g', 'global', 'u32', attribute='unified')


def test_managed_requires_global():
    with pytest.raises(ValueError, match='.global'):
        PTXVariable('g', 'shared', 'u32', attribute='managed')


def test_bad_identifier():
    with pytest.raises(ValueError):
        PTXVariable('invalid-name', 'reg', 'u32')


def test_nonfundamental_type():
    with pytest.raises(ValueError, match='fundamental'):
        PTXVariable('x', 'reg', 'bf16')


def test_initializer_too_long():
    with pytest.raises(ValueError):
        PTXVariable('a', 'global', 'u32', dimensions=(2,), initializer=[1, 2, 3])


def test_integer_initializer_out_of_range():
    with pytest.raises(ValueError):
        PTXVariable('x', 'global', 'u8', initializer=256)
