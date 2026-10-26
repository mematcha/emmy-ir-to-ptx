"""Tests for PTX Chapter 8 helpers integrated into compile.py."""

import pytest
from compile import PTXMemoryConsistency


def test_default_target():
    m = PTXMemoryConsistency()
    assert m.target == 'sm_89'


def test_fences():
    m = PTXMemoryConsistency()
    assert m.fence() == 'fence.acq_rel.gpu;'
    assert m.fence('cta', 'sc') == 'fence.sc.cta;'
    assert m.fence('.sys', '.sc') == 'fence.sc.sys;'


@pytest.mark.parametrize('scope', ['cluster', 'warp', 'invalid'])
def test_unsupported_scopes_on_sm89(scope):
    with pytest.raises(ValueError):
        PTXMemoryConsistency().fence(scope)


@pytest.mark.parametrize('semantics', ['acquire', 'release'])
def test_newer_fences_rejected_on_sm89(semantics):
    with pytest.raises(ValueError):
        PTXMemoryConsistency().fence(semantics=semantics)


def test_cluster_supported_sm90():
    assert PTXMemoryConsistency('sm_90').fence('cluster') == 'fence.acq_rel.cluster;'
    assert PTXMemoryConsistency('sm_90').fence('cluster', 'release') == 'fence.release.cluster;'


def test_membar():
    m = PTXMemoryConsistency()
    assert m.membar('gl') == 'membar.gl;'
    assert m.membar('cta') == 'membar.cta;'
    with pytest.raises(ValueError):
        m.membar('cluster')


def test_proxy_fence():
    m = PTXMemoryConsistency()
    assert m.proxy_fence() == 'fence.proxy.alias;'
    with pytest.raises(ValueError):
        m.proxy_fence('async')


def test_load_store_weak():
    m = PTXMemoryConsistency()
    assert m.load('%r1', '%rd1') == 'ld.global.u32 %r1, [%rd1];'
    assert m.store('%rd1', '%r1') == 'st.global.u32 [%rd1], %r1;'


def test_ordered_load_store():
    m = PTXMemoryConsistency()
    assert m.load('%r1', '%rd1', order='acquire', scope='gpu') == 'ld.acquire.gpu.global.u32 %r1, [%rd1];'
    assert m.store('%rd1', '%r1', order='release', scope='cta') == 'st.release.cta.global.u32 [%rd1], %r1;'
    assert m.load('%r2', '%rd2+8', '.f32', space='shared', order='relaxed', scope='cta') == 'ld.relaxed.cta.shared.f32 %r2, [%rd2+8];'


def test_volatile_load():
    assert PTXMemoryConsistency().load('%r1', '%rd1', order='volatile') == 'ld.volatile.global.u32 %r1, [%rd1];'


@pytest.mark.parametrize('order', ['release', 'acq_rel'])
def test_invalid_load_order(order):
    with pytest.raises(ValueError):
        PTXMemoryConsistency().load('%r1', '%rd1', order=order)


@pytest.mark.parametrize('order', ['acquire', 'acq_rel'])
def test_invalid_store_order(order):
    with pytest.raises(ValueError):
        PTXMemoryConsistency().store('%rd1', '%r1', order=order)


def test_rejects_unsafe_address_string():
    with pytest.raises(ValueError):
        PTXMemoryConsistency().load('%r1', '%rd1]; trap;')


def test_overlap_and_conflict():
    m = PTXMemoryConsistency()
    assert m.overlaps(0, 8, 4, 8)
    assert not m.overlaps(0, 8, 8, 8)
    assert not m.conflicts(0, 8, False, 4, 8, False)
    assert m.conflicts(0, 8, True, 4, 8, False)


def test_invalid_target():
    with pytest.raises(ValueError):
        PTXMemoryConsistency('sm_60')
