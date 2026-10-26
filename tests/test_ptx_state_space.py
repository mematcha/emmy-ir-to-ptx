
import pytest

from compile import StateSpace


class TestStateSpaceProperties:

    @pytest.mark.parametrize(
        "name, expected",
        [
            (".reg", False),
            (".sreg", False),
            (".const", True),
            (".global", True),
            (".local", True),
            (".shared", True),
            (".param", None),
        ],
    )
    def test_addressable(self, name, expected):
        assert StateSpace(name).addressable is expected

    @pytest.mark.parametrize(
        "name, expected",
        [
            (".reg", False),
            (".sreg", True),
            (".const", True),
            (".global", False),
            (".local", False),
            (".param", None),
        ],
    )
    def test_read_only(self, name, expected):
        assert StateSpace(name).read_only is expected

    def test_special_registers_predefined(self):
        assert StateSpace(".sreg").predefined is True

    def test_normal_registers_not_predefined(self):
        assert StateSpace(".reg").predefined is False

    @pytest.mark.parametrize(
        "name, expected",
        [
            (".reg", "thread"),
            (".local", "thread"),
            (".global", "context"),
            (".const", "context"),
            (".shared", "cta_cluster"),
            (".param", "parameter"),
        ],
    )
    def test_thread_scope(self, name, expected):
        assert StateSpace(name).thread_scope == expected

    def test_legacy_texture_deprecated(self):
        assert StateSpace(".tex").deprecated is True

    def test_properties_are_immutable(self):
        from dataclasses import FrozenInstanceError

        state = StateSpace(".reg")

        with pytest.raises(FrozenInstanceError):
            state.properties.addressable = True


class TestStateSpace:

    @pytest.mark.parametrize(
        "name",
        [
            ".reg", ".sreg", ".const",
            ".global", ".local", ".param",
            ".shared", ".tex", ".surf",
        ],
    )
    def test_valid_state_spaces(self, name):
        state_space = StateSpace(name)
        assert str(state_space) == name
        assert state_space.name == name

    def test_normalization(self):
        assert str(StateSpace("reg")) == ".reg"
        assert str(StateSpace("shared")) == ".shared"

    def test_invalid_state_space(self):
        with pytest.raises(ValueError):
            StateSpace(".unknown")

    @pytest.mark.parametrize("value", [None, 123, True, []])
    def test_invalid_input_type(self, value):
        with pytest.raises(TypeError):
            StateSpace(value)

    def test_repr(self):
        assert repr(StateSpace("reg")) == "StateSpace('.reg')"
