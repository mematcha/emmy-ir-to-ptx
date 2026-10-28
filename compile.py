
import argparse
import logging
import sys
from pathlib import Path
import re
import math
import struct
from typing import Any
from collections.abc import Mapping
import json

logger = logging.getLogger("emmy-ptx")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

PTX_VERSION = "8.0"
TARGET_ARCH = "sm_89"
ADDRESS_SIZE = 64

_MISSING = object()

PTX_DIRECTIVES = frozenset("""
    .address_size .explicitcluster .maxnreg .section
    .alias .extern .maxntid .shared
    .align .file .minnctapersm .sreg
    .branchtargets .func .noreturn .target
    .callprototype .global .param .tex
    .calltargets .loc .pragma .version
    .common .local .reg .visible
    .const .maxclusterrank .reqnctapercluster .weak
    .entry .maxnctapersm .reqntid
""".split())

PTX_INSTRUCTIONS = frozenset("""
    abs activemask add addc alloca and applypriority atom
    bar barrier bfe bfi bfind bmsk bra brev brkpt brx
    call clmad clz cnot copysign cos clusterlaunchcontrol
    cp createpolicy cvt cvta
    discard div dp2a dp4a elect ex2 exit fence fma fns
    getctarank griddepcontrol isspacep istypep
    ld ldmatrix ldu lg2 lop3
    mad mad24 madc mapa match max mbarrier membar min mma
    mov movmatrix mul mul24 multimem
    nanosleep neg not or pmevent popc prefetch prefetchu
    prmt rcp red redux rem ret rsqrt sad selp
    set setmaxnreg setp shf shfl shl shr sin slct
    spcompress spdecompress sqrt st stackrestore stacksave
    stmatrix sub subc suld suq sured sust szext
    tanh tcgen05 tensormap testp tex tld4 trap txq
    vabsdiff vabsdiff2 vabsdiff4 vadd vadd2 vadd4
    vavrg2 vavrg4 vmad vmax vmax2 vmax4
    vmin vmin2 vmin4 vote vset vset2 vset4
    vshl vshr vsub vsub2 vsub4 wgmma wmma xor
""".split())

PTX_PREDEFINED = frozenset("""
    %aggr_smem_size %dynamic_smem_size
    %clock %clock64 %globaltimer %globaltimer_hi
    %globaltimer_lo %gridid %smid %nsmid
    %tid %ntid %ctaid %nctaid
    %cluster_ctaid %cluster_ctarank %cluster_nctaid
    %cluster_nctarank %clusterid %nclusterid
    %is_explicit_cluster %laneid %warpid %nwarpid
    %lanemask_eq %lanemask_ge %lanemask_gt
    %lanemask_le %lanemask_lt
    %current_graph_exec %total_smem_size
    %reserved_smem_offset_begin
    %reserved_smem_offset_end
    %reserved_smem_offset_cap
    WARP_SZ
""".split())


from dataclasses import dataclass


@dataclass(frozen=True)
class StateSpaceProperties:
    """Static properties of a PTX state space."""

    addressable: bool | None
    read_only: bool | None
    predefined: bool
    thread_scope: str
    deprecated: bool = False

class StateSpace:
    """Represent a PTX state space and its properties."""

    PROPERTIES = {
        ".reg": StateSpaceProperties(
            addressable=False,
            read_only=False,
            predefined=False,
            thread_scope="thread",
        ),
        ".sreg": StateSpaceProperties(
            addressable=False,
            read_only=True,
            predefined=True,
            thread_scope="special",
        ),
        ".const": StateSpaceProperties(
            addressable=True,
            read_only=True,
            predefined=False,
            thread_scope="context",
        ),
        ".global": StateSpaceProperties(
            addressable=True,
            read_only=False,
            predefined=False,
            thread_scope="context",
        ),
        ".local": StateSpaceProperties(
            addressable=True,
            read_only=False,
            predefined=False,
            thread_scope="thread",
        ),
        ".param": StateSpaceProperties(
            addressable=None,
            read_only=None,
            predefined=False,
            thread_scope="parameter",
        ),
        ".shared": StateSpaceProperties(
            addressable=True,
            read_only=False,
            predefined=False,
            thread_scope="cta_cluster",
        ),
        ".tex": StateSpaceProperties(
            addressable=None,
            read_only=True,
            predefined=False,
            thread_scope="context",
            deprecated=True,
        ),
        ".surf": StateSpaceProperties(
            addressable=None,
            read_only=None,
            predefined=False,
            thread_scope="context",
            deprecated=True,
        ),
    }

    VALID_SPACES = frozenset(PROPERTIES)

    def __init__(self, name: str):
        if not isinstance(name, str):
            raise TypeError(
                "State space name must be a string."
            )

        normalized = (
            name if name.startswith(".") else f".{name}"
        )

        if normalized not in self.VALID_SPACES:
            raise ValueError(
                f"Invalid PTX state space: {name!r}"
            )

        self._name = normalized

    @property
    def name(self) -> str:
        return self._name

    
    @property
    def properties(self) -> StateSpaceProperties:
        return self.PROPERTIES[self._name]

    @property
    def addressable(self) -> bool | None:
        return self.properties.addressable

    @property
    def read_only(self) -> bool | None:
        return self.properties.read_only

    @property
    def predefined(self) -> bool:
        return self.properties.predefined

    @property
    def thread_scope(self) -> str:
        return self.properties.thread_scope

    @property
    def deprecated(self) -> bool:
        return self.properties.deprecated


    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"StateSpace({self._name!r})"

@dataclass(frozen=True)
class PTXTypeInfo:
    """Metadata for a PTX type or instruction data format."""

    category: str
    bits: int
    fundamental: bool
    register_type: str
    elements: int = 1
    element_type: str | None = None

class PTXType:
    """PTX ISA 9.4 Section 5.2 type metadata and validation."""

    # Section 5.2.1 — fundamental types (Table 8)
    FUNDAMENTAL = {
        ".s8": ("signed", 8),
        ".s16": ("signed", 16),
        ".s32": ("signed", 32),
        ".s64": ("signed", 64),

        ".u8": ("unsigned", 8),
        ".u16": ("unsigned", 16),
        ".u32": ("unsigned", 32),
        ".u64": ("unsigned", 64),

        ".f16": ("float", 16),
        ".f16x2": ("float", 32),
        ".f32": ("float", 32),
        ".f64": ("float", 64),

        ".b8": ("bits", 8),
        ".b16": ("bits", 16),
        ".b32": ("bits", 32),
        ".b64": ("bits", 64),
        ".b128": ("bits", 128),

        ".pred": ("predicate", 1),
    }

    # Section 5.2.3 — alternate floating-point formats.
    # These are formats, not fundamental register types.
    ALTERNATE_FLOATS = {
        ".bf16": (16, ".b16"),
        ".e4m3": (8, ".b8"),
        ".e5m2": (8, ".b8"),
        ".tf32": (32, ".b32"),
        ".e2m1": (4, ".b8"),
        ".e2m3": (6, ".b8"),
        ".e3m2": (6, ".b8"),
        ".ue8m0": (8, ".b8"),
        ".ue4m3": (7, ".b8"),
        ".ue5m3": (8, ".b8"),
    }

    # Section 5.2.4 — fixed-point data format.
    FIXED_POINT = {
        ".s2f6": (8, ".b8"),
    }

    # Section 5.2.5 — packed instruction data types.
    # Values: (element_type, element_count, total_bits,
    #          required register declaration type).
    PACKED = {
        ".f32x2": (".f32", 2, 64, ".b64"),
        ".bf16x2": (".bf16", 2, 32, ".b32"),
        ".e4m3x2": (".e4m3", 2, 16, ".b16"),
        ".e5m2x2": (".e5m2", 2, 16, ".b16"),
        ".e2m3x2": (".e2m3", 2, 16, ".b16"),
        ".e3m2x2": (".e3m2", 2, 16, ".b16"),
        ".ue8m0x2": (".ue8m0", 2, 16, ".b16"),
        ".ue5m3x2": (".ue5m3", 2, 16, ".b16"),
        ".s2f6x2": (".s2f6", 2, 16, ".b16"),
        ".e2m1x2": (".e2m1", 2, 8, ".b8"),

        ".e4m3x4": (".e4m3", 4, 32, ".b32"),
        ".e5m2x4": (".e5m2", 4, 32, ".b32"),
        ".e2m3x4": (".e2m3", 4, 32, ".b32"),
        ".e3m2x4": (".e3m2", 4, 32, ".b32"),
        ".e2m1p4x4": (".e2m1", 4, 32, ".b32"),
        ".ue8m0x4": (".ue8m0", 4, 32, ".b32"),
        ".e2m1x4": (".e2m1", 4, 16, ".b16"),

        ".u16x2": (".u16", 2, 32, ".b32"),
        ".s16x2": (".s16", 2, 32, ".b32"),
        ".u8x4": (".u8", 4, 32, ".b32"),
        ".s8x4": (".s8", 4, 32, ".b32"),
    }

    SUBWORD_8_INSTRUCTIONS = frozenset({
        "ld", "st", "add", "sub", "min", "max",
        "neg", "cvt",
    })

    def __init__(self, name: str):
        if not isinstance(name, str):
            raise TypeError("PTX type name must be a string.")

        normalized = name if name.startswith(".") else f".{name}"
        self._name = normalized
        self._info = self._resolve(normalized)

    @classmethod
    def _resolve(cls, name: str) -> PTXTypeInfo:
        if name in cls.FUNDAMENTAL:
            category, bits = cls.FUNDAMENTAL[name]
            elements = 2 if name == ".f16x2" else 1
            element_type = ".f16" if elements == 2 else None

            return PTXTypeInfo(
                category=category,
                bits=bits,
                fundamental=True,
                register_type=name,
                elements=elements,
                element_type=element_type,
            )

        if name in cls.ALTERNATE_FLOATS:
            bits, register_type = cls.ALTERNATE_FLOATS[name]

            return PTXTypeInfo(
                category="alternate_float",
                bits=bits,
                fundamental=False,
                register_type=register_type,
            )

        if name in cls.FIXED_POINT:
            bits, register_type = cls.FIXED_POINT[name]

            return PTXTypeInfo(
                category="fixed_point",
                bits=bits,
                fundamental=False,
                register_type=register_type,
            )

        if name in cls.PACKED:
            element_type, elements, bits, register_type = (
                cls.PACKED[name]
            )

            return PTXTypeInfo(
                category="packed",
                bits=bits,
                fundamental=False,
                register_type=register_type,
                elements=elements,
                element_type=element_type,
            )

        raise ValueError(f"Unsupported PTX type or format: {name!r}")

    @property
    def name(self) -> str:
        return self._name

    @property
    def info(self) -> PTXTypeInfo:
        return self._info

    @property
    def bits(self) -> int:
        return self._info.bits

    @property
    def category(self) -> str:
        return self._info.category

    @property
    def fundamental(self) -> bool:
        return self._info.fundamental

    @property
    def register_type(self) -> str:
        """Type required for a basic register declaration."""
        return self._info.register_type

    @property
    def elements(self) -> int:
        return self._info.elements

    @property
    def element_type(self) -> str | None:
        return self._info.element_type

    def require_fundamental(self) -> str:
        """Validate use as a register declaration type."""
        if not self.fundamental:
            raise ValueError(
                f"{self.name} is not a fundamental PTX type; "
                f"use {self.register_type} for register storage."
            )
        return self.name

    def compatible_with(self, other: "PTXType") -> bool:
        """
        Fundamental type compatibility from Section 5.2.1.

        This does not imply that an arbitrary instruction
        accepts both operand types.
        """
        if not isinstance(other, PTXType):
            raise TypeError("Expected a PTXType instance.")

        if not self.fundamental or not other.fundamental:
            return False

        if self.bits != other.bits:
            return False

        if self.name == other.name:
            return True

        if "bits" in (self.category, other.category):
            return True

        integer_categories = {"signed", "unsigned"}
        if (
            self.category in integer_categories
            and other.category in integer_categories
        ):
            return True

        return False

    def allowed_for_instruction(self, opcode: str) -> bool:
        """
        Check only the explicit restricted-type rules
        from Section 5.2.2.

        True for other types means 'not rejected by this
        limited check', not full instruction validation.
        """
        if not isinstance(opcode, str):
            raise TypeError("Opcode must be a string.")

        base = opcode.split(".", 1)[0]

        if not base:
            raise ValueError("Opcode cannot be empty.")

        if self.name in {".s8", ".u8", ".b8"}:
            return base in self.SUBWORD_8_INSTRUCTIONS

        # Detailed floating-point and packed instruction
        # restrictions require instruction-specific rules.
        if self.name in {".f16", ".f16x2"}:
            raise NotImplementedError(
                "Half-precision instruction validation "
                "requires instruction-specific rules."
            )

        return True

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"PTXType({self.name!r})"

class PTXOperands:
    """Reusable PTX operand constructors and shape-level checks (PTX chapter 6).

    This validates operand *syntax*, not complete instruction legality.
    """

    _NAME = re.compile(r"(?:[A-Za-z][A-Za-z0-9_$]*|[_$%][A-Za-z0-9_$]+)\Z", re.ASCII)
    _ROUNDING = frozenset({".rn", ".rna", ".rz", ".rm", ".rp", ".rs",
                           ".rni", ".rzi", ".rmi", ".rpi"})

    @staticmethod
    def _identifier(name: str) -> str:
        if not isinstance(name, str) or not PTXOperands._NAME.fullmatch(name):
            raise ValueError(f"Invalid PTX identifier: {name!r}")
        return name

    @staticmethod
    def _dtype(dtype) -> str:
        # Reuse the compiler's PTXType objects, rather than type-name strings.
        if not isinstance(dtype, PTXType):
            raise TypeError("dtype must be a PTXType instance")
        return dtype.name

    @dataclass(frozen=True)
    class Register:
        name: str
        dtype: object

        def __post_init__(self):
            PTXOperands._identifier(self.name)
            PTXOperands._dtype(self.dtype)

        def __str__(self):
            return self.name

    @dataclass(frozen=True)
    class Immediate:
        literal: str
        dtype: object

        def __post_init__(self):
            PTXOperands._dtype(self.dtype)
            if (not isinstance(self.literal, str) or not self.literal or
                    not self.literal.isascii() or
                    any(c in self.literal for c in ';\r\n{}[]')):
                raise ValueError("Invalid PTX immediate string")
            # Pass values produced by PTXConstants; this is not a literal parser.

        def __str__(self):
            return self.literal

    @dataclass(frozen=True)
    class Address:
        base: str | int
        offset: int = 0
        kind: str = "register"  # register, symbol, absolute

        def __post_init__(self):
            if self.kind not in {"register", "symbol", "absolute"}:
                raise ValueError("Invalid address kind")
            if (isinstance(self.offset, bool) or not isinstance(self.offset, int)
                    or not -(1 << 31) <= self.offset < (1 << 31)):
                raise ValueError("Address offset must be a signed 32-bit integer")
            if self.kind == "absolute":
                if (isinstance(self.base, bool) or not isinstance(self.base, int)
                        or not 0 <= self.base < (1 << 32) or self.offset):
                    raise ValueError("Absolute address must be a u32 with no offset")
            else:
                PTXOperands._identifier(self.base)

        def __str__(self):
            if self.kind == "absolute":
                return f"[{self.base}]"
            suffix = f"+{self.offset}" if self.offset > 0 else str(self.offset) if self.offset < 0 else ""
            return f"[{self.base}{suffix}]"

    @dataclass(frozen=True)
    class ArrayElement:
        array: str
        index: int | str
        offset: int = 0

        def __post_init__(self):
            PTXOperands._identifier(self.array)
            if isinstance(self.index, bool) or not isinstance(self.index, (int, str)):
                raise TypeError("Array index must be an int or register name")
            if isinstance(self.index, str):
                PTXOperands._identifier(self.index)
            if isinstance(self.offset, bool) or not isinstance(self.offset, int):
                raise TypeError("Array offset must be an int")
            if isinstance(self.index, int) and self.offset:
                raise ValueError("Use a combined integer index")

        def __str__(self):
            suffix = (f"+{self.offset}" if self.offset > 0 else str(self.offset)) if self.offset else ""
            return f"{self.array}[{self.index}{suffix}]"

    @dataclass(frozen=True)
    class Vector:
        registers: tuple
        destination: bool = False

        def __post_init__(self):
            if not isinstance(self.registers, tuple) or len(self.registers) not in (2, 4):
                raise ValueError("Vector operand must have 2 or 4 elements")
            if not all(isinstance(r, PTXOperands.Register) for r in self.registers):
                raise TypeError("Vector entries must be PTXOperands.Register")
            if len({r.dtype.name for r in self.registers}) != 1:
                raise ValueError("Vector entries must share a type")
            if self.destination and len({r.name for r in self.registers}) != len(self.registers):
                raise ValueError("Destination registers must be distinct")

        def __str__(self):
            return "{" + ", ".join(map(str, self.registers)) + "}"

    @dataclass(frozen=True)
    class Component:
        name: str
        component: str

        def __post_init__(self):
            PTXOperands._identifier(self.name)
            if self.component not in {"x", "y", "z", "w", "r", "g", "b", "a"}:
                raise ValueError("Unsupported vector component")

        def __str__(self):
            return f"{self.name}.{self.component}"

    @dataclass(frozen=True)
    class Symbol:
        name: str
        kind: str  # label or function

        def __post_init__(self):
            PTXOperands._identifier(self.name)
            if self.kind not in {"label", "function"}:
                raise ValueError("Symbol kind must be label or function")

        def valid_for(self, opcode: str) -> bool:
            base = opcode.split(".", 1)[0]
            return base in ({"bra", "brx"} if self.kind == "label" else {"call", "mov"})

        def __str__(self):
            return self.name

    @staticmethod
    def compatible_types(source, target) -> bool:
        PTXOperands._dtype(source)
        PTXOperands._dtype(target)
        if source.name == target.name:
            return True
        if source.bits != target.bits:
            return False
        if "bits" in (source.category, target.category):
            return True
        return source.category in {"signed", "unsigned"} and target.category in {"signed", "unsigned"}

    @staticmethod
    def access_alignment_ok(byte_address: int, access_size_bytes: int) -> bool:
        for n in (byte_address, access_size_bytes):
            if isinstance(n, bool) or not isinstance(n, int):
                raise TypeError("Address and access size must be integers")
        if byte_address < 0 or access_size_bytes <= 0:
            raise ValueError("Address must be nonnegative and size positive")
        return byte_address % access_size_bytes == 0

    @staticmethod
    def conversion_opcode(destination, source, *, rounding: str | None = None) -> str:
        dest = PTXOperands._dtype(destination)
        src = PTXOperands._dtype(source)
        if rounding is not None and rounding not in PTXOperands._ROUNDING:
            raise ValueError(f"Unsupported rounding modifier: {rounding}")
        # Construct spelling only; detailed cvt legality is checked elsewhere.
        return f"cvt{rounding or ''}{dest}{src}"

class PTXVariable:
    """A validated PTX variable declaration; no instructions are emitted."""

    _DECLARABLE_SPACES = frozenset({
        '.reg', '.const', '.global', '.local', '.param', '.shared'
    })
    _IDENTIFIER = re.compile(r'(?:[A-Za-z][A-Za-z0-9_$]*|[_$%][A-Za-z0-9_$]+)\Z', re.ASCII)

    def __init__(
        self,
        name: str,
        state_space: 'StateSpace | str',
        dtype: 'PTXType | str',
        *,
        vector: int = 1,
        dimensions: tuple[int | None, ...] = (),
        initializer: Any = _MISSING,
        align: int | None = None,
        count: int | None = None,
        attribute: str | None = None,
        target: str = 'sm_89',
    ) -> None:
        self.state_space = state_space if isinstance(state_space, StateSpace) else StateSpace(state_space)
        self.dtype = dtype if isinstance(dtype, PTXType) else PTXType(dtype)
        self.name = self._validate_name(name)
        self.vector = vector
        self.dimensions = dimensions
        self.initializer = initializer
        self.align = align
        self.count = count
        self.attribute = attribute
        self.target = target
        self._validate()

    @classmethod
    def _validate_name(cls, name: str) -> str:
        if not isinstance(name, str) or not cls._IDENTIFIER.fullmatch(name):
            raise ValueError(f'Invalid PTX variable identifier: {name!r}')
        # Reserve instruction names and built-in names when the existing
        # module-level identifier tables are available.
        if name in PTX_INSTRUCTIONS or name in PTX_PREDEFINED:
            raise ValueError(f'Reserved PTX identifier: {name!r}')
        return name

    def _validate(self) -> None:
        space = self.state_space.name
        typ = self.dtype.name
        if space not in self._DECLARABLE_SPACES:
            raise ValueError(f'Unsupported user-defined variable space: {space}')
        if not self.dtype.fundamental:
            raise ValueError(f'Variable type must be fundamental: {typ}')
        if typ == '.pred' and space != '.reg':
            raise ValueError('Predicate variables are only allowed in .reg')
        if type(self.vector) is not int or self.vector not in (1, 2, 4):
            raise ValueError('Vector length must be 1, 2, or 4')
        if self.vector != 1:
            if typ == '.pred' or self.vector * self.dtype.bits > 128:
                raise ValueError('Vector must be non-predicate and at most 128 bits')
        if not isinstance(self.dimensions, tuple):
            raise TypeError('dimensions must be a tuple')
        for index, size in enumerate(self.dimensions):
            if size is None:
                if index != 0 or self.initializer is _MISSING:
                    raise ValueError('Only the first dimension may be inferred, with initializer')
            elif type(size) is not int or size <= 0:
                raise ValueError('Array dimensions must be positive integers')
        if self.dimensions and space == '.reg':
            raise ValueError('Use a register name group, not a register array')
        if self.dimensions and self.vector != 1:
            raise ValueError('Arrays of vectors are not yet supported')
        if self.align is not None:
            if type(self.align) is not int or self.align <= 0 or self.align & (self.align - 1):
                raise ValueError('Alignment must be a positive power of two')
            if not self.state_space.addressable:
                raise ValueError('Alignment requires addressable storage')
        if self.count is not None:
            if type(self.count) is not int or self.count <= 0:
                raise ValueError('Parameterized-name count must be positive')
            if self.dimensions or self.initializer is not _MISSING or self.vector != 1:
                raise ValueError('Parameterized names cannot use arrays, vectors, or initializers')
        if self.initializer is not _MISSING:
            if space not in ('.global', '.const'):
                raise ValueError('Only .global and .const variables support initializers')
            if typ in ('.pred', '.f16', '.f16x2'):
                raise ValueError(f'Initializer not supported for {typ}')
            self._validate_initializer(self.initializer, 0)
        if self.attribute not in (None, 'managed', 'unified'):
            raise ValueError('Unsupported variable attribute')
        if self.attribute is not None:
            if space != '.global':
                raise ValueError('Variable attributes require .global space')
            if self.attribute == 'unified':
                raise ValueError('.unified requires sm_90+ and is unsupported for sm_89')
            if self.target != 'sm_89':
                raise ValueError('This declaration model currently targets sm_89')
            if self.align is not None:
                raise ValueError('Combining .attribute and .align is not yet supported')
        if self.target != 'sm_89':
            raise ValueError('Current PTXVariable model targets sm_89 only')

    def _validate_initializer(self, value: Any, depth: int) -> None:
        # For vectors, one brace-enclosed sequence is required.
        shape = self.dimensions or ((self.vector,) if self.vector != 1 else ())
        if depth < len(shape):
            if not isinstance(value, (list, tuple)):
                raise TypeError('Aggregate initializers must use nested lists or tuples')
            bound = shape[depth]
            if len(value) == 0 or (bound is not None and len(value) > bound):
                raise ValueError('Initializer extent exceeds declared dimensions or is empty')
            if depth == 0 and bound is None and len(value) == 0:
                raise ValueError('Inferred array dimension must be nonempty')
            for element in value:
                self._validate_initializer(element, depth + 1)
            return
        if isinstance(value, (list, tuple)):
            raise TypeError('Scalar initializer must be a single numeric value')
        typ = self.dtype.name
        if typ.startswith('.f'):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Floating initializer must be finite and numeric')
        else:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError('Integer initializer must be an integer')
            bits = self.dtype.bits
            if typ.startswith('.s'):
                low, high = -(1 << (bits-1)), (1 << (bits-1)) - 1
            else:
                low, high = 0, (1 << bits) - 1
            if not low <= value <= high:
                raise ValueError(f'Initializer {value} outside {typ} range')

    def _format_scalar(self, value: int | float) -> str:
        if self.dtype.name.startswith('.f'):
            return repr(float(value))
        return str(value)

    def _format_initializer(self, value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return '{' + ', '.join(self._format_initializer(item) for item in value) + '}'
        return self._format_scalar(value)

    def declaration(self) -> str:
        """Return one PTX declaration statement ending in a semicolon."""
        parts = [self.state_space.name]
        if self.align is not None:
            parts += ['.align', str(self.align)]
        if self.attribute is not None:
            parts.append(f'.attribute(.{self.attribute})')
        if self.vector != 1:
            parts.append(f'.v{self.vector}')
        parts.append(self.dtype.name)
        name = self.name
        if self.count is not None:
            name += f'<{self.count}>'
        for dim in self.dimensions:
            name += f'[{"" if dim is None else dim}]'
        parts.append(name)
        line = ' '.join(parts)
        if self.initializer is not _MISSING:
            line += ' = ' + self._format_initializer(self.initializer)
        return line + ';'

    def __str__(self) -> str:
        return self.declaration()

    def __repr__(self) -> str:
        return f'PTXVariable({self.name!r}, {self.state_space.name!r}, {self.dtype.name!r})'

class PTXConstants:
    """Manage PTX constant values and expressions."""

    def __init__(self, version: str = PTX_VERSION):
        self.version = version
   
    def integer(
        self,
        value: int,
        unsigned: bool = False,
        base: int = 10,
    ) -> str:
        """Format a PTX integer literal using PTX ISA rules."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Integer constant must be an int.")

        if base not in (2, 8, 10, 16):
            raise ValueError("Supported bases: 2, 8, 10, 16.")

        S64_MIN = -(1 << 63)
        S64_MAX = (1 << 63) - 1
        U64_MAX = (1 << 64) - 1

        if unsigned:
            if not 0 <= value <= U64_MAX:
                raise ValueError("Unsigned integer outside u64 range.")
        elif not S64_MIN <= value <= U64_MAX:
            raise ValueError("Integer outside PTX 64-bit range.")

        # Values above signed maximum require unsigned representation.
        is_unsigned = unsigned or value > S64_MAX

        magnitude = abs(value)

        if base == 10:
            literal = str(magnitude)
        elif base == 16:
            literal = f"0x{magnitude:X}"
        elif base == 8:
            literal = f"0{magnitude:o}"
        else:
            literal = f"0b{magnitude:b}"

        if is_unsigned:
            literal += "U"

        if value < 0:
            literal = "-" + literal

        logger.debug(
            "Formatted PTX integer: value=%s, base=%s, unsigned=%s, literal=%s",
            value,
            base,
            is_unsigned,
            literal,
        )

        return literal
   
    def floating(self, value: float) -> str:
        """Format a finite PTX floating-point literal (f64)."""

        if not isinstance(value, float):
            raise TypeError("Floating-point constant must be a float.")

        if not math.isfinite(value):
            raise ValueError(
                "Non-finite values require exact IEEE 754 notation."
            )

        literal = repr(value)

        logger.debug(
            "Formatted PTX floating constant: %s", literal
        )

        return literal


    def float_bits(self, value: float, bits: int = 32) -> str:
        """Format exact IEEE 754 floating-point bit notation."""

        if not isinstance(value, float):
            raise TypeError("Floating-point constant must be a float.")

        if bits not in (32, 64):
            raise ValueError("Floating-point bits must be 32 or 64.")

        if bits == 32:
            # Round Python's binary64 value into IEEE 754 binary32.
            try:
                packed = struct.pack(">f", value)
            except OverflowError as exc:
                raise ValueError(
                    "Value exceeds the finite f32 range."
                ) from exc

            literal = f"0f{int.from_bytes(packed, 'big'):08X}"

        else:
            packed = struct.pack(">d", value)
            literal = f"0d{int.from_bytes(packed, 'big'):016X}"

        logger.debug(
            "Formatted PTX exact floating constant: %s", literal
        )

        return literal


    def predicate(self, value: int) -> str:
        """Format a PTX predicate-compatible integer constant."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("Predicate constant must be an integer.")

        # Reuse our integer literal validator.
        self.integer(value)

        literal = "0" if value == 0 else "1"

        logger.debug(
            "Formatted PTX predicate: input=%s, result=%s",
            value,
            literal,
        )

        return literal

class PTXABI:
    """PTX ISA 9.4 Section 7: device-function ABI syntax and stack helpers.

    This formats and validates basic constructs. It does not choose the physical
    calling convention or perform liveness/stack allocation.
    """

    _IDENT = re.compile(r"(?:[A-Za-z][A-Za-z0-9_$]*|[_$%][A-Za-z0-9_$]+)\Z", re.ASCII)
    _DTYPE = re.compile(r"\.(?:[sub](?:8|16|32|64)|b128|f16|f16x2|f32|f64|pred)\Z")
    _REG_ARG = re.compile(r"(?:[A-Za-z][A-Za-z0-9_$]*|[_$%][A-Za-z0-9_$]+)\Z", re.ASCII)
    _INT_ARG = re.compile(r"-?(?:0|[1-9][0-9]*)(?:U)?\Z")

    @dataclass(frozen=True)
    class Parameter:
        name: str
        dtype: str
        space: str = ".reg"
        size: int | None = None  # byte-array length only
        align: int | None = None  # alignment for .param arrays

        def __post_init__(self):
            PTXABI._identifier(self.name)
            if self.space not in (".reg", ".param"):
                raise ValueError("Device-function formal parameters must be .reg or .param")
            if not isinstance(self.dtype, str) or not PTXABI._DTYPE.fullmatch(self.dtype):
                raise ValueError(f"Unsupported formal parameter type: {self.dtype!r}")
            if self.space == ".reg":
                if self.size is not None or self.align is not None:
                    raise ValueError(".reg formal cannot have size/align")
                # Under ABI, subword values need promotion.
                bits = 1 if self.dtype == ".pred" else (32 if self.dtype == ".f16x2" else int(re.search(r"\d+", self.dtype).group()))
                if bits < 32:
                    raise ValueError("ABI .reg formal parameters must be at least 32 bits")
            else:
                if self.size is not None and (type(self.size) is not int or self.size <= 0):
                    raise ValueError("Byte-array size must be a positive integer")
                if self.size is not None and self.dtype != ".b8":
                    raise ValueError("This helper supports .param byte arrays only (.b8)")
                if self.align is not None and self.align not in (1, 2, 4, 8, 16):
                    raise ValueError("ABI .param alignment must be 1, 2, 4, 8, or 16")

        def declaration(self) -> str:
            fields = [self.space]
            if self.align is not None:
                fields.extend((".align", str(self.align)))
            fields.extend((self.dtype, self.name))
            text = " ".join(fields)
            return text + (f"[{self.size}]" if self.size is not None else "")

    @staticmethod
    def _identifier(value: str) -> str:
        if not isinstance(value, str) or not PTXABI._IDENT.fullmatch(value):
            raise ValueError(f"Invalid PTX identifier: {value!r}")
        return value

    @classmethod
    def _parameters(cls, values):
        params = tuple(values)
        if not all(isinstance(p, cls.Parameter) for p in params):
            raise TypeError("Expected PTXABI.Parameter objects")
        names = [p.name for p in params]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate formal parameter names")
        return params

    @classmethod
    def function(cls, name: str, *, inputs=(), returns=(), body: str | None = None) -> str:
        """Format a device .func declaration or definition (modern sm_20+ ABI)."""
        cls._identifier(name)
        ins, outs = cls._parameters(inputs), cls._parameters(returns)
        if len(outs) > 1:
            raise ValueError("Modern PTX ABI permits at most one return parameter")
        if name in {p.name for p in (*ins, *outs)}:
            raise ValueError("Function name conflicts with formal parameter")
        if len({p.name for p in (*ins, *outs)}) != len(ins) + len(outs):
            raise ValueError("Input and return parameter names must be unique")
        out_text = f" ({outs[0].declaration()})" if outs else ""
        in_text = f" ({', '.join(p.declaration() for p in ins)})" if ins else ""
        signature = f".func{out_text} {name}{in_text}"
        if body is None:
            return signature + ";"
        if not isinstance(body, str):
            raise TypeError("Function body must be text")
        body.encode("ascii")
        return signature + "\n{\n" + ("\n".join("    " + line if line else "" for line in body.splitlines()) + "\n" if body else "") + "}"

    @classmethod
    def call(cls, name: str, *, arguments=(), returns=()) -> str:
        """Format a call. Caller must separately verify types and staging rules."""
        cls._identifier(name)
        args, ret = tuple(arguments), tuple(returns)
        if len(ret) > 1:
            raise ValueError("Modern PTX ABI permits at most one return value")
        for operand in (*args, *ret):
            if not isinstance(operand, str) or not (cls._REG_ARG.fullmatch(operand) or cls._INT_ARG.fullmatch(operand)):
                raise ValueError(f"Unsupported call operand: {operand!r}")
        result = f"call ({ret[0]}), {name}" if ret else f"call {name}"
        if args:
            result += ", (" + ", ".join(args) + ")"
        return result + ";"

    @classmethod
    def stack_alloc(cls, destination: str, size: int, *, alignment: int = 8) -> str:
        """Build alloca.u64 for a per-thread local stack allocation."""
        cls._identifier(destination)
        if type(size) is not int or size <= 0:
            raise ValueError("Allocation size must be a positive integer")
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("Requested alignment must be a power of two")
        if alignment > (1 << 23):
            raise ValueError("PTX alloca alignment cannot exceed 2^23")
        return f"alloca.u64 {destination}, {size}, {alignment};"

    @classmethod
    def stack_save(cls, destination: str) -> str:
        cls._identifier(destination)
        return f"stacksave.u64 {destination};"

    @classmethod
    def stack_restore(cls, source: str) -> str:
        cls._identifier(source)
        return f"stackrestore.u64 {source};"

class PTXMemoryConsistency:
    """PTX Chapter 8 memory-ordering helpers for a selected GPU target.

    Generates a conservative subset of real PTX instructions. It does NOT
    prove that a multi-threaded program is free of data races.
    """

    SCOPES = frozenset({'cta', 'cluster', 'gpu', 'sys'})
    ORDERS = frozenset({'weak', 'relaxed', 'acquire', 'release', 'acq_rel', 'volatile'})
    SPACES = frozenset({'global', 'shared'})
    _IDENTIFIER = re.compile(r'(?:[A-Za-z][A-Za-z0-9_$]*|[_$%][A-Za-z0-9_$]+)\Z', re.ASCII)

    def __init__(self, target: str = TARGET_ARCH):
        if not isinstance(target, str) or re.fullmatch(r'sm_(\d{2,3})(?:[af])?', target) is None:
            raise ValueError('Expected target architecture such as sm_89')
        self.target = target
        self.compute_capability = int(re.match(r'sm_(\d+)', target).group(1))
        if self.compute_capability < 70:
            raise ValueError('Chapter 8 memory model requires sm_70 or newer')

    @classmethod
    def _normalize(cls, value: str, allowed: frozenset[str], field: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f'{field} must be a string')
        normalized = value.removeprefix('.')
        if normalized not in allowed:
            raise ValueError(f'Unsupported {field}: {value!r}')
        return normalized

    def scope(self, value: str) -> str:
        result = self._normalize(value, self.SCOPES, 'scope')
        if result == 'cluster' and self.compute_capability < 90:
            raise ValueError('.cluster scope requires sm_90 or newer')
        return result

    @classmethod
    def _register(cls, value: str) -> str:
        if not isinstance(value, str) or not cls._IDENTIFIER.fullmatch(value):
            raise ValueError(f'Invalid PTX register: {value!r}')
        return value

    @classmethod
    def _address(cls, value: str) -> str:
        """Conservative address syntax: identifier with optional byte offset."""
        if not isinstance(value, str):
            raise TypeError('Address must be a string')
        if re.fullmatch(r'(?:[A-Za-z][A-Za-z0-9_$]*|[_$%][A-Za-z0-9_$]+)(?:[+-]\d+)?', value, re.ASCII) is None:
            raise ValueError(f'Invalid PTX address expression: {value!r}')
        return f'[{value}]'

    def fence(self, scope: str = 'gpu', semantics: str = 'acq_rel') -> str:
        """Generate fence.acq_rel or fence.sc (the forms available on sm_89)."""
        scope = self.scope(scope)
        semantics = self._normalize(semantics, frozenset({'sc', 'acq_rel', 'acquire', 'release'}), 'fence semantics')
        if semantics in {'acquire', 'release'} and self.compute_capability < 90:
            raise ValueError('fence.acquire/release requires sm_90 or newer')
        return f'fence.{semantics}.{scope};'

    def membar(self, level: str = 'gl') -> str:
        """Legacy fence syntax; on sm_70+ its semantics correspond to fence.sc."""
        level = self._normalize(level, frozenset({'cta', 'gl', 'sys'}), 'membar level')
        return f'membar.{level};'

    def proxy_fence(self, proxy: str = 'alias') -> str:
        """Supported unidirectional-free proxy fences for sm_89."""
        if proxy != 'alias':
            raise ValueError('Only fence.proxy.alias is supported by this conservative sm_89 helper')
        return 'fence.proxy.alias;'

    def load(self, dst: str, address: str, dtype: str = '.u32', *,
             space: str = 'global', order: str = 'weak', scope: str = 'gpu') -> str:
        """Generate a subset of scalar ld forms; inputs must be declared elsewhere."""
        return self._memory('ld', dst, address, dtype, space, order, scope)

    def store(self, address: str, src: str, dtype: str = '.u32', *,
              space: str = 'global', order: str = 'weak', scope: str = 'gpu') -> str:
        """Generate a subset of scalar st forms; inputs must be declared elsewhere."""
        return self._memory('st', src, address, dtype, space, order, scope)

    def _memory(self, op: str, reg: str, address: str, dtype: str,
                space: str, order: str, scope: str) -> str:
        reg = self._register(reg)
        addr = self._address(address)
        space = self._normalize(space, self.SPACES, 'state space')
        order = self._normalize(order, self.ORDERS, 'ordering')
        if not isinstance(dtype, str) or dtype not in {'.b32', '.b64', '.u32', '.u64', '.s32', '.s64', '.f32', '.f64'}:
            raise ValueError('Only basic 32/64-bit scalar types are supported here')
        if op == 'ld' and order in {'release', 'acq_rel'}:
            raise ValueError('Loads cannot use release/acq_rel ordering')
        if op == 'st' and order in {'acquire', 'acq_rel'}:
            raise ValueError('Stores cannot use acquire/acq_rel ordering')
        if order == 'volatile':
            # Volatile is not a replacement for acquire/release synchronization.
            mnemonic = f'{op}.volatile.{space}{dtype}'
        elif order == 'weak':
            mnemonic = f'{op}.{space}{dtype}'
        else:
            scope = self.scope(scope)
            mnemonic = f'{op}.{order}.{scope}.{space}{dtype}'
        if op == 'ld':
            return f'{mnemonic} {reg}, {addr};'
        return f'{mnemonic} {addr}, {reg};'

    @staticmethod
    def overlaps(start_a: int, size_a: int, start_b: int, size_b: int) -> bool:
        """Test overlap of known byte ranges; ignores virtual-address aliasing."""
        for value in (start_a, size_a, start_b, size_b):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError('Address and size values must be integers')
        if min(start_a, start_b) < 0 or min(size_a, size_b) <= 0:
            raise ValueError('Addresses must be nonnegative and sizes positive')
        return start_a < start_b + size_b and start_b < start_a + size_a

    @classmethod
    def conflicts(cls, start_a: int, size_a: int, write_a: bool,
                  start_b: int, size_b: int, write_b: bool) -> bool:
        if not isinstance(write_a, bool) or not isinstance(write_b, bool):
            raise TypeError('write flags must be bool')
        return cls.overlaps(start_a, size_a, start_b, size_b) and (write_a or write_b)

class PTXInstructionSet:
    """Build a validated, deliberately narrow set of PTX instructions.

    Does not track register declarations, pointer provenance, bounds,
    alignment at runtime, control flow, or data-flow semantics.
    All returned strings are complete, single PTX instructions.
    """

    _INTEGER = frozenset({
        '.s16', '.u16', '.s32', '.u32', '.s64', '.u64'
    })
    _FLOAT = frozenset({'.f32', '.f64'})
    _ROUND = frozenset({'rn', 'rz', 'rm', 'rp'})
    _LOAD_STORE_TYPES = frozenset({
        '.b8', '.b16', '.b32', '.b64',
        '.u8', '.u16', '.u32', '.u64',
        '.s8', '.s16', '.s32', '.s64', '.f32', '.f64'
    })
    _LOAD_SPACES = frozenset({'global', 'shared', 'local', 'const', 'param'})
    _STORE_SPACES = frozenset({'global', 'shared', 'local'})
    _CVT_TYPES = frozenset({
        '.s16', '.u16', '.s32', '.u32', '.s64', '.u64', '.f32', '.f64'
    })
    _SIMPLE_IDENTIFIER = re.compile(
        r'(?:[A-Za-z][A-Za-z0-9_$]*|[_$%][A-Za-z0-9_$]+)\Z', re.ASCII
    )
    _SPECIAL_IDENTIFIER = re.compile(
        r'%[A-Za-z_$][A-Za-z0-9_$]*\.[xyzw]\Z', re.ASCII
    )
    _INTEGER_LITERAL = re.compile(
        r'[+-]?(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]*|[1-9][0-9]*)(?:[uU])?\Z',
        re.ASCII,
    )
    _FLOAT_LITERAL = re.compile(
        r'[+-]?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)(?:[eE][+-]?[0-9]+)?)\Z',
        re.ASCII,
    )
    _FLOAT_BITS = re.compile(
        r'(?:0[fF][0-9a-fA-F]{8}|0[dD][0-9a-fA-F]{16})\Z', re.ASCII
    )
    _ADDRESS_TEXT = re.compile(
        r'\[(?:[A-Za-z][A-Za-z0-9_$]*|[_$%][A-Za-z0-9_$]+|[0-9]+)(?:[+-][0-9]+)?\]\Z',
        re.ASCII,
    )

    def __init__(self, target: str = 'sm_89', ptx_version: str = '9.4'):
        match = re.fullmatch(r'sm_([0-9]{2,3})', target) if isinstance(target, str) else None
        if not match:
            raise ValueError('Expected target such as sm_89 or sm_90')
        if not isinstance(ptx_version, str) or not re.fullmatch(r'[0-9]+\.[0-9]+', ptx_version):
            raise ValueError('Expected PTX version such as 9.4')
        self.target = target
        self.compute_capability = int(match.group(1))
        self.ptx_version = ptx_version

    @staticmethod
    def _dtype(dtype: 'PTXType | str') -> 'PTXType':
        if isinstance(dtype, PTXType):
            return dtype
        if not isinstance(dtype, str):
            raise TypeError('dtype must be PTXType or PTX type string')
        return PTXType(dtype)

    @classmethod
    def _operand(cls, operand, *, destination=False) -> str:
        """Check operand spelling and basic role, not full PTX semantics."""
        if isinstance(operand, PTXOperands.Register):
            return operand.name
        if isinstance(operand, PTXOperands.Immediate):
            if destination:
                raise ValueError('Immediate cannot be a destination')
            text = operand.literal
        elif isinstance(operand, str):
            text = operand
        else:
            raise TypeError('Operand must be a string, PTXOperands.Register, or Immediate')

        if not isinstance(text, str) or not text.isascii() or not text:
            raise ValueError('Invalid PTX operand')
        if cls._SIMPLE_IDENTIFIER.fullmatch(text) or cls._SPECIAL_IDENTIFIER.fullmatch(text):
            return text
        if not destination and (
            cls._INTEGER_LITERAL.fullmatch(text) or
            cls._FLOAT_LITERAL.fullmatch(text) or
            cls._FLOAT_BITS.fullmatch(text)
        ):
            return text
        raise ValueError(f'Unsupported/unsafe PTX operand: {text!r}')

    @classmethod
    def _address(cls, operand) -> str:
        if isinstance(operand, PTXOperands.Address):
            return str(operand)
        if isinstance(operand, str) and cls._ADDRESS_TEXT.fullmatch(operand):
            return operand
        raise ValueError('Address must be PTXOperands.Address or a simple [base+offset] string')

    @staticmethod
    def _render(opcode: str, *operands: str) -> str:
        return f"{opcode} {', '.join(operands)};"

    @staticmethod
    def _register_type(operand):
        """Typed register metadata, if supplied; bare strings are untyped."""
        return operand.dtype if isinstance(operand, PTXOperands.Register) else None

    @classmethod
    def _check_register(cls, operand, expected: 'PTXType | str', *, narrow_load=False):
        actual = cls._register_type(operand)
        if actual is None:
            return
        expected = cls._dtype(expected)
        if narrow_load and expected.name in {'.s8', '.u8', '.b8', '.s16', '.u16', '.b16'}:
            if actual.name in {'.s32', '.u32', '.b32'}:
                return
        if actual.name == expected.name:
            return
        if not actual.compatible_with(expected):
            raise ValueError(
                f'Register {operand.name} has type {actual.name}, expected {expected.name}'
            )

    @classmethod
    def _check_value(cls, operand, expected: 'PTXType | str', *, narrow_load=False):
        if isinstance(operand, PTXOperands.Immediate):
            cls._check_immediate_type(operand, expected)
        else:
            cls._check_register(operand, expected, narrow_load=narrow_load)

    @classmethod
    def _check_immediate_type(cls, operand, expected):
        actual = operand.dtype
        expected = cls._dtype(expected)
        # Same bit-width integers are interchangeable; bit types are compatible.
        if actual.name != expected.name and not actual.compatible_with(expected):
            raise ValueError(f'Immediate type {actual.name} does not match {expected.name}')

    @classmethod
    def _rounding(cls, rounding: str | None, *, required=False) -> str:
        if rounding is None:
            if required:
                raise ValueError('Explicit rounding mode is required')
            return ''
        if rounding not in cls._ROUND:
            raise ValueError(f'Unsupported rounding mode: {rounding!r}')
        return f'.{rounding}'

    def _binary(self, opcode, dtype, dst, left, right, *, rounding=None):
        typ = self._dtype(dtype)
        if typ.name not in self._INTEGER | self._FLOAT:
            raise ValueError(f'{opcode} does not support {typ.name} in this implementation')
        if typ.name in self._INTEGER and rounding is not None:
            raise ValueError('Rounding modifier cannot be used with integer arithmetic')
        modifier = self._rounding(rounding) if typ.name in self._FLOAT else ''
        for op in (dst, left, right):
            self._check_value(op, typ)
        return self._render(
            f'{opcode}{modifier}{typ.name}',
            self._operand(dst, destination=True), self._operand(left), self._operand(right),
        )

    def add(self, dtype, dst, left, right, *, rounding=None):
        return self._binary('add', dtype, dst, left, right, rounding=rounding)

    def sub(self, dtype, dst, left, right, *, rounding=None):
        return self._binary('sub', dtype, dst, left, right, rounding=rounding)

    def mul(self, dtype, dst, left, right, *, mode=None, rounding=None):
        """Integer mul defaults to .lo; .wide doubles destination width."""
        typ = self._dtype(dtype)
        if typ.name in self._FLOAT:
            if mode is not None:
                raise ValueError('Floating-point mul has no .lo/.hi/.wide mode')
            return self._binary('mul', typ, dst, left, right, rounding=rounding)
        if typ.name not in self._INTEGER:
            raise ValueError(f'Unsupported integer multiply type: {typ.name}')
        if rounding is not None:
            raise ValueError('Integer mul cannot specify rounding')
        mode = 'lo' if mode is None else mode
        if mode not in {'lo', 'hi', 'wide'}:
            raise ValueError('Integer mul mode must be lo, hi, or wide')
        if mode == 'wide' and typ.bits not in (16, 32):
            raise ValueError('mul.wide only supports 16/32-bit integer operands')
        destination_type = (
            ('.s' if typ.name.startswith('.s') else '.u') + str(typ.bits * 2)
            if mode == 'wide' else typ.name
        )
        self._check_register(dst, destination_type)
        self._check_value(left, typ)
        self._check_value(right, typ)
        return self._render(
            f'mul.{mode}{typ.name}', self._operand(dst, destination=True),
            self._operand(left), self._operand(right),
        )

    def fma(self, dtype, dst, a, b, c, *, rounding='rn'):
        typ = self._dtype(dtype)
        if typ.name not in self._FLOAT:
            raise ValueError('fma currently supports .f32 and .f64')
        if self.compute_capability < 20 and typ.name == '.f32':
            raise ValueError('fma.f32 requires sm_20 or later')
        if self.compute_capability < 13 and typ.name == '.f64':
            raise ValueError('fma.f64 requires sm_13 or later')
        mode = self._rounding(rounding, required=True)
        for operand in (dst, a, b, c):
            self._check_value(operand, typ)
        return self._render(
            f'fma{mode}{typ.name}', self._operand(dst, destination=True),
            self._operand(a), self._operand(b), self._operand(c),
        )

    def mov(self, dtype, dst, src):
        typ = self._dtype(dtype)
        if typ.name not in {
            '.pred', '.b16', '.b32', '.b64', '.u16', '.u32', '.u64',
            '.s16', '.s32', '.s64', '.f32', '.f64'
        }:
            raise ValueError(f'mov unsupported for type {typ.name}')
        self._check_register(dst, typ)
        self._check_value(src, typ)
        return self._render(
            f'mov{typ.name}', self._operand(dst, destination=True), self._operand(src)
        )

    def cvt(self, dst_type, src_type, dst, src, *, rounding=None):
        dest_type = self._dtype(dst_type)
        source_type = self._dtype(src_type)
        if dest_type.name not in self._CVT_TYPES or source_type.name not in self._CVT_TYPES:
            raise ValueError('cvt supports s/u16/32/64 and f32/f64 in this phase')
        self._check_register(dst, dest_type)
        self._check_value(src, source_type)

        dest_float = dest_type.name in self._FLOAT
        src_float = source_type.name in self._FLOAT
        if dest_float and src_float:
            # Widening needs no rounding; narrowing must specify rounding.
            if dest_type.bits < source_type.bits:
                rounding = 'rn' if rounding is None else rounding
            elif rounding is not None:
                raise ValueError('Do not specify rounding on widening float conversion')
        elif dest_float:
            rounding = 'rn' if rounding is None else rounding
        elif src_float:
            # PTX's .rzi = integer rounding toward zero, unlike .rz.
            rounding = 'rzi' if rounding is None else rounding
        elif rounding is not None:
            raise ValueError('Integer-to-integer cvt does not accept a rounding mode here')

        if dest_float or src_float:
            permitted = {'rni', 'rzi', 'rmi', 'rpi'} if (src_float and not dest_float) else self._ROUND
            if rounding is not None and rounding not in permitted:
                raise ValueError(f'Invalid conversion rounding: {rounding!r}')
        suffix = f'.{rounding}' if rounding else ''
        return self._render(
            f'cvt{suffix}{dest_type.name}{source_type.name}',
            self._operand(dst, destination=True), self._operand(src),
        )

    def _space(self, space, *, store=False):
        if isinstance(space, StateSpace):
            space = space.name
        if not isinstance(space, str):
            raise TypeError('State space must be StateSpace or str')
        space = space.removeprefix('.')
        allowed = self._STORE_SPACES if store else self._LOAD_SPACES
        if space not in allowed:
            raise ValueError(f'Unsupported {"store" if store else "load"} space: {space!r}')
        return space

    def load(self, space, dtype, dst, address):
        """Scalar weak load; address is presumed valid and suitably aligned."""
        ss = self._space(space)
        typ = self._dtype(dtype)
        if typ.name not in self._LOAD_STORE_TYPES:
            raise ValueError(f'Unsupported scalar load type: {typ.name}')
        self._check_register(dst, typ, narrow_load=True)
        return self._render(
            f'ld.{ss}{typ.name}', self._operand(dst, destination=True), self._address(address),
        )

    def store(self, space, dtype, address, src):
        """Scalar weak store; .const and .param excluded deliberately."""
        ss = self._space(space, store=True)
        typ = self._dtype(dtype)
        if typ.name not in self._LOAD_STORE_TYPES:
            raise ValueError(f'Unsupported scalar store type: {typ.name}')
        self._check_value(src, typ)
        return self._render(f'st.{ss}{typ.name}', self._address(address), self._operand(src))

    def _vector(self, operand, typ, *, destination=False):
        if not isinstance(operand, PTXOperands.Vector):
            raise TypeError('Vector load/store requires PTXOperands.Vector')
        if destination and not operand.destination:
            raise ValueError('Mark a vector destination with destination=True')
        count = len(operand.registers)
        if count not in (2, 4):
            raise ValueError('Only v2/v4 vector instructions are implemented')
        if count * typ.bits > 128 and self.compute_capability < 100:
            raise ValueError('Vector transfers over 128 bits require sm_100+')
        for register in operand.registers:
            self._check_register(register, typ, narrow_load=destination)
        return f'.v{count}', str(operand)

    def load_vector(self, space, dtype, dst, address):
        """Load a v2/v4 tuple of scalar elements into registers."""
        ss = self._space(space)
        typ = self._dtype(dtype)
        if typ.name not in self._LOAD_STORE_TYPES:
            raise ValueError(f'Unsupported vector load type: {typ.name}')
        vec, text = self._vector(dst, typ, destination=True)
        return self._render(f'ld.{ss}{vec}{typ.name}', text, self._address(address))

    def store_vector(self, space, dtype, address, src):
        """Store a v2/v4 tuple of scalar register elements."""
        ss = self._space(space, store=True)
        typ = self._dtype(dtype)
        if typ.name not in self._LOAD_STORE_TYPES:
            raise ValueError(f'Unsupported vector store type: {typ.name}')
        vec, text = self._vector(src, typ)
        return self._render(f'st.{ss}{vec}{typ.name}', self._address(address), text)

    def cvta(self, space, dst, src, *, size=64, to=False):
        """State-space address -> generic (or reverse with to=True)."""
        if self.compute_capability < 20:
            raise ValueError('cvta requires sm_20 or later')
        ss = self._space(space)
        if isinstance(size, bool) or size not in (32, 64):
            raise ValueError('cvta address size must be 32 or 64')
        if ss == 'param' and to:
            # The .param::entry special variant is not exposed in this phase.
            raise ValueError('cvta.to.param needs an explicit ::entry form; not implemented')
        expected = f'.u{size}'
        self._check_register(dst, expected)
        self._check_register(src, expected)
        return self._render(
            f"cvta.{('to.' if to else '')}{ss}.u{size}",
            self._operand(dst, destination=True), self._operand(src),
        )

@dataclass(frozen=True)
class EmmyKernelIR:
    """Validated kernel view. `graph` and `body` remain Emmy-native objects."""
    name: str
    body: object
    graph: object
    node_key: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    arg_order: tuple[str, ...] | None
    launch: dict
    metadata: dict

class EmmyIRParser:
    """Read an Emmy Graph without using Emmy's CUDA lowering/rendering.

    `meta.json` is optional during exploration but REQUIRED for a
    submission-ready signature and launch-contract validation.
    """

    def __init__(self, input_path, *, meta_path=None, require_meta=False):
        self.input_path = Path(input_path)
        self.meta_path = Path(meta_path) if meta_path is not None else None
        self.require_meta = require_meta

    @staticmethod
    def _read_json(path):
        try:
            with Path(path).open('r', encoding='utf-8') as stream:
                result = json.load(stream)
        except OSError as exc:
            raise ValueError(f'Cannot read {path}: {exc}') from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSON in {path}: {exc}') from exc

        if not isinstance(result, dict):
            raise ValueError(f'Expected a JSON object in {path}')
        return result

    def _metadata_path(self):
        if self.meta_path is not None:
            return self.meta_path
        return self.input_path.parent / 'meta.json'

    @staticmethod
    def _string_names(value, field):
        if value is None:
            return ()

        if not isinstance(value, (tuple, list)):
            raise ValueError(f'{field} must be a list/tuple of names')

        result = tuple(value)

        if any(not isinstance(name, str) or not name for name in result):
            raise ValueError(f'{field} contains an invalid name')

        if len(result) != len(set(result)):
            raise ValueError(f'{field} contains duplicate names')

        return result

    def parse(self):
        payload = self._read_json(self.input_path)

        try:
            from emmy.compiler.graph import Graph
        except ImportError as exc:
            raise RuntimeError(
                'Install emmy-ml==0.3.11 to parse Kernel IR'
            ) from exc

        try:
            graph = Graph.from_dict(payload)
        except Exception as exc:
            raise ValueError(
                f'Emmy could not deserialize {self.input_path}: {exc}'
            ) from exc

        nodes = getattr(graph, 'nodes', None)

        if not isinstance(nodes, Mapping):
            raise ValueError(
                'Deserialized Emmy Graph has no nodes mapping'
            )

        kernels = []

        for key, node in nodes.items():
            op = getattr(node, 'op', None)

            if op is not None and type(op).__name__ == 'KernelOp':
                kernels.append((key, op))

        if len(kernels) != 1:
            raise ValueError(
                f'Expected exactly one KernelOp, found {len(kernels)}'
            )

        key, op = kernels[0]
        name = getattr(op, 'name', None)
        body = getattr(op, 'body', None)

        if not isinstance(name, str) or not name:
            raise ValueError('KernelOp.name is missing or invalid')

        if body is None:
            raise ValueError('KernelOp.body is missing')

        # Node inputs/outputs are informative only.
        # meta.json defines ABI argument order.
        node = nodes[key]

        inputs = self._string_names(
            getattr(node, 'inputs', None),
            'node.inputs',
        )

        outputs = self._string_names(
            [
                tensor.name
                for tensor in getattr(node, "outputs", ())
            ],
            "node.outputs",
        )

        metadata_path = self._metadata_path()

        if metadata_path.is_file():
            metadata = self._read_json(metadata_path)
        elif self.require_meta:
            raise ValueError(
                f'Missing required metadata file: {metadata_path}'
            )
        else:
            metadata = {}

        
        # Metadata is organized per kernel, not at the top level.
        # Select by both graph node and PTX entry name.
        kernel_records = metadata.get("kernels")

        if kernel_records is None:
            if self.require_meta:
                raise ValueError(
                    "meta.json is missing the 'kernels' array"
                )
            kernel_meta = {}

        else:
            if not isinstance(kernel_records, list):
                raise ValueError(
                    "meta.kernels must be a list"
                )

            matches = [
                record
                for record in kernel_records
                if isinstance(record, dict)
                and record.get("node") == str(key)
                and record.get("entry") == name
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one metadata record for "
                    f"node={key!r}, entry={name!r}; "
                    f"found {len(matches)}"
                )

            kernel_meta = matches[0]

        order = kernel_meta.get("arg_order")

        if order is not None:
            order = self._string_names(
                order,
                "meta.kernels[].arg_order",
            )

            if not order:
                raise ValueError(
                    "Kernel arg_order cannot be empty"
                )

        elif self.require_meta:
            raise ValueError(
                f"Missing arg_order for kernel {name!r}"
            )

        launch = {
            field: kernel_meta[field]
            for field in (
                "grid",
                "block",
                "smem_bytes",
            )
            if field in kernel_meta
        }

        # Preserve both problem-level and per-kernel metadata.
        metadata = {
            **metadata,
            "selected_kernel": kernel_meta,
        }

        return EmmyKernelIR(
            name=name,
            body=body,
            graph=graph,
            node_key=str(key),
            inputs=inputs,
            outputs=outputs,
            arg_order=order,
            launch=launch,
            metadata=metadata,
        )

class PTXKernelGenerator:
    """Construct a PTX .entry function from parsed Emmy kernel metadata.

    This stage manages kernel structure, parameter declarations,
    virtual registers, shared storage, and the basic thread index.

    IR statement lowering and GPU correctness belong to Stage 3.
    """

    def __init__(self, kernel: EmmyKernelIR):
        if not isinstance(kernel, EmmyKernelIR):
            raise TypeError("Expected an EmmyKernelIR instance")

        self.kernel = kernel
        self.name = PTXABI._identifier(kernel.name)

        if not kernel.arg_order:
            raise ValueError("Kernel requires metadata arg_order")

        selected = kernel.metadata.get("selected_kernel")

        if not isinstance(selected, dict):
            raise ValueError(
                "Missing selected_kernel metadata"
            )

        if selected.get("entry") != kernel.name:
            raise ValueError(
                "Kernel entry does not match metadata"
            )

        if selected.get("node") != kernel.node_key:
            raise ValueError(
                "Kernel node does not match metadata"
            )

        args = selected.get("args")

        if not isinstance(args, list):
            raise ValueError(
                "Kernel metadata requires an args list"
            )

        self.arg_order = tuple(kernel.arg_order)

        if len(set(self.arg_order)) != len(self.arg_order):
            raise ValueError("Duplicate kernel arguments")

        self.arguments = {}

        for arg in args:
            if not isinstance(arg, dict):
                raise ValueError("Invalid argument metadata")

            name = arg.get("name")
            PTXABI._identifier(name)

            if name in self.arguments:
                raise ValueError(
                    f"Duplicate argument metadata: {name}"
                )

            self.arguments[name] = arg

        if set(self.arguments) != set(self.arg_order):
            raise ValueError(
                "arg_order and args metadata do not match"
            )

        self._params = {}
        self._declarations = []
        self._instructions = []
        self._register_counts = {}

        # All current hackathon kernel arguments are buffer pointers.
        # Do not silently treat unsupported scalar ABI arguments
        # as pointers if their representation changes.
        for arg_name in self.arg_order:
            self._params[arg_name] = (
                self._parameter_name(arg_name)
            )


    
    def label(self, name: str) -> None:
        """Append a PTX branch target label."""
        PTXABI._identifier(name)

        if any(line == f"{name}:" for line in self._instructions):
            raise ValueError(f"Duplicate PTX label: {name}")

        self._instructions.append(f"{name}:")

    @staticmethod
    def _parameter_name(name: str) -> str:
        PTXABI._identifier(name)
        return "param_" + name

    @staticmethod
    def _validate_line(line: str) -> str:
        if not isinstance(line, str):
            raise TypeError("PTX instruction must be text")

        if (
            not line.isascii()
            or "\n" in line
            or "\r" in line
            or not line.strip()
            or not line.strip().endswith(";")
        ):
            raise ValueError(
                "Expected one ASCII PTX instruction"
            )

        text = line.strip()

        # Avoid allowing callers to inject a second statement.
        if text.count(";") != 1:
            raise ValueError(
                "Expected exactly one PTX instruction"
            )

        return text

    def parameter(self, name: str) -> str:
        """Get the PTX parameter symbol for an IR buffer."""
        if name not in self._params:
            raise KeyError(f"Unknown kernel argument: {name}")
        return self._params[name]

    def register(self, dtype: str = ".u32") -> str:
        """Allocate and declare a fresh virtual register."""

        typ = PTXType(dtype)
        typ.require_fundamental()

        if typ.name == ".b128":
            raise ValueError(
                "128-bit scalar registers are not supported here"
            )

        prefix = {
            ".pred": "%p",
            ".f32": "%f",
            ".f64": "%fd",
            ".u64": "%rd",
            ".s64": "%rd",
            ".b64": "%rd",
        }.get(typ.name, "%r")

        index = self._register_counts.get(prefix, 0)
        self._register_counts[prefix] = index + 1

        name = f"{prefix}{index}"

        declaration = PTXVariable(
            name=name,
            state_space=".reg",
            dtype=typ,
        ).declaration()

        self._declarations.append(declaration)

        return name

    def shared(
        self,
        name: str,
        dtype: str,
        *,
        length: int,
        align: int | None = None,
    ) -> str:
        """Declare a fixed-size shared-memory array."""

        if type(length) is not int or length <= 0:
            raise ValueError(
                "Shared-memory length must be positive"
            )

        declaration = PTXVariable(
            name=name,
            state_space=".shared",
            dtype=dtype,
            dimensions=(length,),
            align=align,
        ).declaration()

        if declaration in self._declarations:
            raise ValueError(
                f"Duplicate shared declaration: {name}"
            )

        self._declarations.append(declaration)
        return name

    def emit(self, instruction: str) -> None:
        """Append a complete instruction from PTXInstructionSet
        or from the future IR lowering stage.
        """
        self._instructions.append(
            self._validate_line(instruction)
        )

    def thread_index_1d(self) -> str:
        """Compute blockIdx.x * blockDim.x + threadIdx.x.

        Returns a fresh .u32 register holding the global
        flattened x-dimension thread index.

        Not sufficient by itself for multidimensional Tile
        decoding or strided reduction loops.
        """

        thread_id = self.register(".u32")
        block_id = self.register(".u32")
        block_size = self.register(".u32")
        offset = self.register(".u32")
        index = self.register(".u32")

        self.emit(f"mov.u32 {thread_id}, %tid.x;")
        self.emit(f"mov.u32 {block_id}, %ctaid.x;")
        self.emit(f"mov.u32 {block_size}, %ntid.x;")

        self.emit(
            f"mul.lo.u32 {offset}, {block_id}, {block_size};"
        )
        self.emit(
            f"add.u32 {index}, {offset}, {thread_id};"
        )

        return index

    def render_entry(self) -> str:
        """Render one .visible .entry function.

        Deliberately does not generate module-level
        .version/.target directives.
        """

        params = [
            f"    .param .u64 {self._params[name]}"
            for name in self.arg_order
        ]

        signature = (
            f".visible .entry {self.name}(\n"
            + ",\n".join(params)
            + "\n)\n"
        )

        lines = [signature.rstrip(), "{"]

        for declaration in self._declarations:
            lines.append(f"    {declaration}")

        if self._declarations and self._instructions:
            lines.append("")

        for instruction in self._instructions:
            lines.append(f"    {instruction}")

        if (
            not self._instructions
            or self._instructions[-1] != "ret;"
        ):
            lines.append("    ret;")

        lines.append("}")

        return "\n".join(lines) + "\n"




class EmmyIRLowering:
    """Lower 1D f32 pointwise Emmy Kernel IR directly to PTX.

    Stage 3A supports:
      Tile, Load, Assign, Write
      Var, Literal, BinaryExpr (+, -, *)
      add, subtract, multiply

    Unsupported IR fails explicitly.
    """

    def __init__(self, kernel: EmmyKernelIR):
        self.kernel = kernel
        self.gen = PTXKernelGenerator(kernel)
        self.iset = PTXInstructionSet(
            target=TARGET_ARCH,
            ptx_version=PTX_VERSION,
        )

        self.values = {}
        self.axes = {}
        self.pointers = {}
        self.shapes = {}

        selected = kernel.metadata["selected_kernel"]

        for arg in selected["args"]:
            name = arg["name"]
            dtype = arg["dtype"]
            shape = arg["shape"]

            if dtype != "f32" or len(shape) not in (1, 2):
                raise NotImplementedError(
                    f"Unsupported tensor dtype or rank: "
                    f"{name}, dtype={dtype}, shape={shape}"
                )

            self.shapes[name] = tuple(shape)

        self._label_counter = 0

        # State used by the general multi-axis lowering path.
        self.carried = set()        # names declared by Init (updated in place)
        self.smem = {}              # shared array -> {"base", "dims"}
        self.module_decls = []      # module-scope declarations (dynamic smem)
        self._live_inits = set()    # accumulators seeded by an enclosing loop
        self._smem_pool_used = 0
        self._smem_dynamic = False
        self.tid = None
        self.bid = None

    def _new_label(self):
        self._label_counter += 1
        return f"L_pointwise_end_{self._label_counter}"

    def _emit(self, instruction):
        self.gen.emit(instruction)

    def _u32(self, value):
        """Create a u32 register containing an integer constant."""
        reg = self.gen.register(".u32")
        self._emit(self.iset.mov(".u32", reg, str(value)))
        return reg

    def _expr(self, expr):
        """Lower an Emmy integer index expression."""
        kind = type(expr).__name__

        if kind == "Var":
            if expr.name in self.axes:
                return self.axes[expr.name]
            raise NotImplementedError(
                f"Unknown index variable: {expr.name}"
            )

        if kind == "Literal":
            value = expr.value
            if type(value) is not int or value < 0:
                raise NotImplementedError(
                    "Only nonnegative integer indices supported"
                )
            return self._u32(value)

        if kind == "BinaryExpr":
            left = self._expr(expr.left)
            right = self._expr(expr.right)
            dst = self.gen.register(".u32")

            if expr.op == "+":
                instruction = self.iset.add(
                    ".u32", dst, left, right
                )
            elif expr.op == "-":
                instruction = self.iset.sub(
                    ".u32", dst, left, right
                )
            elif expr.op == "*":
                instruction = self.iset.mul(
                    ".u32", dst, left, right, mode="lo"
                )
            else:
                raise NotImplementedError(
                    f"Index operator: {expr.op}"
                )

            self._emit(instruction)
            return dst

        raise NotImplementedError(
            f"Index expression: {kind}"
        )

    def _buffer_address(self, name, index_reg):
        """Compute pointer + index * sizeof(float32)."""
        if name not in self.pointers:
            raise ValueError(f"Unknown buffer: {name}")

        offset = self.gen.register(".u64")

        self._emit(
            self.iset.mul(
                ".u32",
                offset,
                index_reg,
                "4",
                mode="wide",
            )
        )

        address = self.gen.register(".u64")

        self._emit(
            self.iset.add(
                ".u64",
                address,
                self.pointers[name],
                offset,
            )
        )

        return address

    def _flat_index(self, index):
        if len(index) != 1:
            raise NotImplementedError(
                "Stage 3A requires one index coordinate"
            )
        return self._expr(index[0])

    def _load(self, stmt):
        if len(stmt.names) != 1:
            raise NotImplementedError(
                "Vector loads are not implemented yet"
            )

        if stmt.dtype.name != "f32":
            raise NotImplementedError(
                "Only f32 loads supported"
            )

        index = self._flat_index(stmt.index)
        address = self._buffer_address(stmt.input, index)
        reg = self.gen.register(".f32")

        self._emit(
            self.iset.load("global", ".f32", reg, f"[{address}]")
        )

        self.values[stmt.names[0]] = reg

    @staticmethod
    def _operation_name(op):
        # Emmy's ElementwiseImpl is displayed as
        # ElementwiseImpl('add'). Support the named attributes
        # when available, otherwise inspect this representation.
        for field in ("name", "op"):
            value = getattr(op, field, None)
            if isinstance(value, str):
                return value

        match = re.fullmatch(
            r"ElementwiseImpl\(['\"]([^'\"]+)['\"]\)",
            repr(op),
        )
        if match:
            return match.group(1)

        raise NotImplementedError(
            f"Unsupported elementwise descriptor: {op!r}"
        )

    def _assign(self, stmt):
        """Lower supported unary and binary f32 elementwise operations."""

        # Combine-state programs (WarpShuffle / TreeHalve) carry dtype=None;
        # every value in these kernels is f32, so treat None as f32.
        if stmt.dtype is not None and stmt.dtype.name != "f32":
            raise NotImplementedError(
                "Stage 3A requires typed f32 assignments"
            )

        op = self._operation_name(stmt.op).lower()
        args = tuple(stmt.args)

        for name in args:
            if name not in self.values:
                raise ValueError(
                    f"Undefined elementwise input: {name}"
                )

        # An Init-declared name is carried state: update it in place.
        if stmt.name in self.carried and stmt.name in self.values:
            dst = self.values[stmt.name]
        else:
            dst = self.gen.register(".f32")

        # -----------------------------
        # Unary operations
        # -----------------------------
        if len(args) == 1:
            src = self.values[args[0]]

            if op in ("exp", "exponential"):
                # exp(x) = 2 ** (x * log2(e)); PTX has ex2, not exp.
                scaled = self.gen.register(".f32")
                self._emit(f"mul.f32 {scaled}, {src}, 0f3fb8aa3b;")
                self._emit(f"ex2.approx.f32 {dst}, {scaled};")

            elif op in ("reciprocal", "recip"):
                self._emit(f"rcp.rn.f32 {dst}, {src};")

            elif op in ("log", "ln"):
                # ln(x) = log2(x) * ln(2)
                scaled = self.gen.register(".f32")
                self._emit(f"lg2.approx.f32 {scaled}, {src};")
                self._emit(f"mul.f32 {dst}, {scaled}, 0f3f317218;")

            elif op in ("square", "sqr"):
                self._emit(f"mul.f32 {dst}, {src}, {src};")

            elif op in ("relu", "relu_", "positive"):
                self._emit(
                    f"max.f32 {dst}, {src}, 0f00000000;"
                )

            elif op in ("neg", "negative"):
                self._emit(
                    f"neg.f32 {dst}, {src};"
                )

            elif op in ("identity", "clone", "copy"):
                self._emit(
                    self.iset.mov(".f32", dst, src)
                )

            elif op in ("abs", "absolute"):
                self._emit(
                    f"abs.f32 {dst}, {src};"
                )

            elif op in ("sqrt", "square_root"):
                self._emit(
                    f"sqrt.rn.f32 {dst}, {src};"
                )

            elif op in ("rsqrt", "reciprocal_sqrt"):
                self._emit(
                    f"rsqrt.approx.f32 {dst}, {src};"
                )

            else:
                raise NotImplementedError(
                    f"Unsupported unary elementwise operation: {op}"
                )

        # -----------------------------
        # Binary operations
        # -----------------------------
        elif len(args) == 2:
            left = self.values[args[0]]
            right = self.values[args[1]]

            if op in ("add", "sum"):
                instruction = self.iset.add(
                    ".f32", dst, left, right
                )

            elif op in ("subtract", "sub"):
                instruction = self.iset.sub(
                    ".f32", dst, left, right
                )

            elif op in ("multiply", "mul"):
                instruction = self.iset.mul(
                    ".f32", dst, left, right
                )

            elif op in ("maximum", "max"):
                instruction = (
                    f"max.f32 {dst}, {left}, {right};"
                )

            elif op in ("minimum", "min"):
                instruction = (
                    f"min.f32 {dst}, {left}, {right};"
                )

            elif op in ("divide", "div", "true_divide"):
                instruction = (
                    f"div.rn.f32 {dst}, {left}, {right};"
                )
            
            else:
                raise NotImplementedError(
                    f"Unsupported binary elementwise operation: {op}"
                )

            self._emit(instruction)

        else:
            raise NotImplementedError(
                f"Unsupported arity {len(args)} "
                f"for elementwise operation {op}"
            )

        self.values[stmt.name] = dst

    def _write(self, stmt):
        if stmt.atomic:
            raise NotImplementedError(
                "Atomic writes belong to a later phase"
            )

        if stmt.value_dtype.name != "f32":
            raise NotImplementedError(
                "Only f32 writes supported"
            )

        # A grouped write stores successive elements.
        # Correct scalar equivalent of a vector store.
        base_index = self._flat_index(stmt.index)

        for offset, name in enumerate(stmt.values):
            if name not in self.values:
                raise ValueError(
                    f"Write references undefined value: {name}"
                )

            index = base_index
            if offset:
                index = self.gen.register(".u32")
                self._emit(
                    self.iset.add(
                        ".u32",
                        index,
                        base_index,
                        str(offset),
                    )
                )

            address = self._buffer_address(
                stmt.output, index
            )

            self._emit(
                self.iset.store(
                    "global",
                    ".f32",
                    f"[{address}]",
                    self.values[name],
                )
            )

    def _tile(self, tile):
        if len(tile.axes) != 1:
            raise NotImplementedError(
                "Stage 3A requires a single Tile axis"
            )

        axis = tile.axes[0]
        extent = getattr(axis.extent, "value", axis.extent)

        if type(extent) is not int or extent <= 0:
            raise ValueError(
                "Tile extent must be a positive integer"
            )

        thread = self.gen.thread_index_1d()
        self.axes[axis.name] = thread

        pred = self.gen.register(".pred")
        end_label = self._new_label()

        # Skip lanes outside the logical Tile extent.
        self._emit(
            f"setp.ge.u32 {pred}, {thread}, {extent};"
        )
        self._emit(f"@{pred} bra {end_label};")

        self._body(tile.body)
        self.gen.label(end_label)

    def _body(self, statements):
        for stmt in statements:
            kind = type(stmt).__name__

            if kind == "Tile":
                self._tile(stmt)
            elif kind == "Load":
                self._load(stmt)
            elif kind == "Assign":
                self._assign(stmt)
            elif kind == "Write":
                self._write(stmt)
            else:
                raise NotImplementedError(
                    f"Stage 3A does not support {kind}"
                )

    def _prologue(self):
        """Read argument pointers from the PTX entry parameters."""
        for name in self.kernel.arg_order:
            reg = self.gen.register(".u64")
            param = self.gen.parameter(name)

            self._emit(
                self.iset.load(
                    "param", ".u64", reg, f"[{param}]"
                )
            )

            # PTX global loads accept generic addresses.
            self.pointers[name] = reg

    def _lower_sum_reduction(self, tile):
        """Lower a cooperative 1D f32 sum reduction.

        Supported pattern:
          StridedLoop(Load, Accum(add))
          WarpShuffle(add)
          Smem
          Cond(lane==0, Write(shared))
          Sync
          TreeHalve(add)
          Cond(thread==0, Write(output))

        Unsupported variants are rejected.
        """
        stmts = tuple(tile.body)
        kinds = tuple(type(s).__name__ for s in stmts)

        # Scalar reduction: one tile axis, one block.
        # Row reduction: row axis + thread axis, one block per row.
        is_row_reduction = len(tile.axes) == 2

        if len(tile.axes) not in (1, 2):
            raise NotImplementedError(
                "Expected 1D scalar or 2D row reduction Tile"
            )

        thread_axis = tile.axes[-1]
        row_axis = tile.axes[0] if is_row_reduction else None
        
        expected = (
            "StridedLoop",
            "WarpShuffle",
            "Smem",
            "Cond",
            "Sync",
            "TreeHalve",
            "Cond",
        )

        if kinds != expected:
            raise NotImplementedError(
                f"Unsupported reduction structure: {kinds}"
            )

        loop, shuffle, smem, store_cond, sync, tree, output_cond = stmts


        if len(loop.body) != 2 or tuple(
            type(s).__name__ for s in loop.body
        ) != ("Load", "Accum"):
            raise NotImplementedError(
                "Expected Load followed by Accum"
            )

        load, accum = loop.body

        if (
            len(load.names) != 1
            or load.dtype.name != "f32"
            or accum.dtype.name != "f32"
            or self._operation_name(accum.op) != "add"
            or accum.value != load.names[0]
        ):
            raise NotImplementedError(
                "Only scalar f32 sum accumulation supported"
            )

        if (
            loop.end is not None
            or not loop.seed
            or type(loop.step).__name__ != "Literal"
        ):
            raise NotImplementedError(
                "Unsupported reduction loop configuration"
            )

        extent = getattr(loop.axis.extent, "value", loop.axis.extent)
        tile_extent = getattr(
            thread_axis.extent, "value", thread_axis.extent
        )
        step = loop.step.value

        if not all(
            type(v) is int and v > 0
            for v in (extent, tile_extent, step)
        ):
            raise ValueError("Reduction extents and stride must be positive integers")

        if type(loop.start).__name__ != "Var":
            raise NotImplementedError("Expected thread-based loop start")

        if loop.start.name != thread_axis.name:
            raise NotImplementedError("Loop start must use Tile thread axis")


        if is_row_reduction:
            if (
                len(load.index) != 2
                or type(load.index[0]).__name__ != "Var"
                or load.index[0].name != row_axis.name
                or type(load.index[1]).__name__ != "Var"
                or load.index[1].name != loop.axis.name
            ):
                raise NotImplementedError(
                    "Expected matrix load x[row, column]"
                )
        else:
            if (
                len(load.index) != 1
                or type(load.index[0]).__name__ != "Var"
                or load.index[0].name != loop.axis.name
            ):
                raise NotImplementedError(
                    "Expected vector reduction load"
                )

        if (
            len(shuffle.state) != 1
            or shuffle.state[0] != accum.name
            or len(shuffle.combine_states) != 1
            or self._operation_name(shuffle.combine_states[0].op) != "add"
            or len(shuffle.state_b) != 1
        ):
            raise NotImplementedError("Unsupported warp combine")

        warp_width = shuffle.length

        if type(warp_width) is not int or warp_width != 32:
            raise NotImplementedError("Only 32-lane warp sum supported")

        block = self.kernel.launch.get("block")
        grid = self.kernel.launch.get("grid")

        
        expected_grid_x = 1

        if is_row_reduction:
            expected_grid_x = getattr(
                row_axis.extent,
                "value",
                row_axis.extent,
            )

        if (
            tuple(block) != (tile_extent, 1, 1)
            or tuple(grid) != (expected_grid_x, 1, 1)
            or step != tile_extent
            or tile_extent % warp_width != 0
        ):
            raise NotImplementedError(
                "Unsupported cooperative reduction launch geometry"
            )


        warps = tile_extent // warp_width

        if (
            len(smem.extents) != 1
            or smem.extents[0] != warps
            or smem.dtype != "float"
            or tree.length != warps
            or tree.tid_var != "warp"
            or tuple(tree.bufs) != (smem.name,)
            or tuple(tree.state) != (accum.name,)
            or len(tree.combine_states) != 1
            or self._operation_name(tree.combine_states[0].op) != "add"
            or warps & (warps - 1)
        ):
            raise NotImplementedError(
                "Unsupported cross-warp reduction layout"
            )

        if (
            type(store_cond.cond).__name__ != "BinaryExpr"
            or store_cond.cond.op != "=="
            or type(store_cond.cond.left).__name__ != "Var"
            or store_cond.cond.left.name != "lane"
            or type(store_cond.cond.right).__name__ != "Literal"
            or store_cond.cond.right.value != 0
            or store_cond.else_body
            or len(store_cond.body) != 1
        ):
            raise NotImplementedError("Unsupported warp partial store condition")

        partial_write = store_cond.body[0]

        if (
            type(partial_write).__name__ != "Write"
            or partial_write.output != smem.name
            or tuple(partial_write.values) != (accum.name,)
            or len(partial_write.index) != 1
            or type(partial_write.index[0]).__name__ != "Var"
            or partial_write.index[0].name != "warp"
            or partial_write.atomic
        ):
            raise NotImplementedError("Unsupported warp partial write")

        if sync.warp or sync.barrier_id != 0:
            raise NotImplementedError("Only CTA barrier 0 supported")


        # Final writer must be thread zero.
        cond = output_cond.cond

        if (
            type(cond).__name__ != "BinaryExpr"
            or cond.op != "=="
            or type(cond.left).__name__ != "Var"
            or cond.left.name != thread_axis.name
            or type(cond.right).__name__ != "Literal"
            or cond.right.value != 0
            or output_cond.else_body
        ):
            raise NotImplementedError(
                "Unsupported final output condition"
            )

        final_body = tuple(output_cond.body)
        final_kinds = tuple(
            type(stmt).__name__ for stmt in final_body
        )

        if final_kinds == ("Write",):
            # Ordinary sum reduction.
            final_mode = "sum"
            output_write = final_body[0]

            if tuple(output_write.values) != (accum.name,):
                raise NotImplementedError(
                    "Unexpected sum reduction output value"
                )

        elif final_kinds == ("Load", "Assign", "Write"):
            # Mean reduction: sum / reduction count.
            final_mode = "mean"
            count_load, divide_stmt, output_write = final_body

            if (
                len(count_load.names) != 1
                or count_load.dtype.name != "f32"
                or len(count_load.index) != 1
                or type(count_load.index[0]).__name__ != "Literal"
                or count_load.index[0].value != 0
                or count_load.input not in self.pointers
            ):
                raise NotImplementedError(
                    "Unsupported mean denominator load"
                )

            if (
                divide_stmt.dtype is None
                or divide_stmt.dtype.name != "f32"
                or self._operation_name(divide_stmt.op) != "divide"
                or tuple(divide_stmt.args)
                    != (accum.name, count_load.names[0])
                or tuple(output_write.values)
                    != (divide_stmt.name,)
            ):
                raise NotImplementedError(
                    "Unsupported mean division pattern"
                )

        else:
            raise NotImplementedError(
                f"Unsupported final reduction body: {final_kinds}"
            )

        
        # A scalar output uses index=().
        # A length-one tensor may use index=(Literal(0),).
        output_index = tuple(output_write.index)

        if is_row_reduction:
            valid_index = (
                len(output_index) == 1
                and type(output_index[0]).__name__ == "Var"
                and output_index[0].name == row_axis.name
            )
        else:
            valid_index = (
                len(output_index) == 0
                or (
                    len(output_index) == 1
                    and (
                        (
                            type(output_index[0]).__name__ == "Literal"
                            and output_index[0].value == 0
                        )
                        or (
                            type(output_index[0]) is int
                            and output_index[0] == 0
                        )
                    )
                )
            )

        if (
            type(output_write).__name__ != "Write"
            or not valid_index
            or output_write.atomic
            or output_write.value_dtype.name != "f32"
            or output_write.output not in self.pointers
        ):
            raise NotImplementedError(
                "Unsupported reduction output write"
            )
        

        input_shape = None

        for arg in self.kernel.metadata["selected_kernel"]["args"]:
            if arg["name"] == load.input:
                input_shape = tuple(arg["shape"])
                break

        if input_shape is None:
            raise ValueError(
                f"Missing tensor metadata for {load.input}"
            )

        if is_row_reduction:
            if (
                len(input_shape) != 2
                or input_shape[0] != expected_grid_x
                or input_shape[1] != extent
            ):
                raise NotImplementedError(
                    "Row reduction requires a contiguous "
                    "row-major f32 input"
                )
        elif len(input_shape) != 1:
            raise NotImplementedError(
                "Scalar sum expects a 1D input"
            )

        # --------------------------------------------------
        # 1. Each thread accumulates strided input elements.
        # --------------------------------------------------
        # A cooperative reduction uses threadIdx.x, not a
        # grid-wide linear thread index.
        tid = self.gen.register(".u32")
        self._emit(f"mov.u32 {tid}, %tid.x;")

        self.axes[thread_axis.name] = tid

        row = None

        if is_row_reduction:
            row = self.gen.register(".u32")
            self._emit(f"mov.u32 {row}, %ctaid.x;")
            self.axes[row_axis.name] = row


        index = self.gen.register(".u32")
        self._emit(f"mov.u32 {index}, {tid};")

        acc = self.gen.register(".f32")
        self._emit(f"mov.f32 {acc}, 0f00000000;")
        self.values[accum.name] = acc

        loop_label = self._new_label()
        exit_label = self._new_label()

        self.gen.label(loop_label)

        predicate = self.gen.register(".pred")
        self._emit(
            f"setp.ge.u32 {predicate}, {index}, {extent};"
        )
        self._emit(f"@{predicate} bra {exit_label};")

        if is_row_reduction:
            # Row-major linear index:
            # row * number_of_columns + column.
            linear_index = self.gen.register(".u32")

            self._emit(
                f"mad.lo.u32 {linear_index}, "
                f"{row}, {input_shape[1]}, {index};"
            )
        else:
            linear_index = index

        address = self._buffer_address(
            load.input,
            linear_index,
        )
        value = self.gen.register(".f32")

        self._emit(
            self.iset.load(
                "global", ".f32", value, f"[{address}]"
            )
        )
        self._emit(
            self.iset.add(".f32", acc, acc, value)
        )

        self._emit(
            f"add.u32 {index}, {index}, {step};"
        )
        self._emit(f"bra {loop_label};")

        self.gen.label(exit_label)

        # --------------------------------------------------
        # 2. Reduce within each warp using XOR butterflies.
        # --------------------------------------------------
        bits_in = self.gen.register(".b32")
        bits_out = self.gen.register(".b32")
        shuffled = self.gen.register(".f32")

        for mask in (16, 8, 4, 2, 1):
            self._emit(f"mov.b32 {bits_in}, {acc};")
            self._emit(
                f"shfl.sync.bfly.b32 {bits_out}, "
                f"{bits_in}, {mask}, 0x1f, 0xffffffff;"
            )
            self._emit(
                f"mov.b32 {shuffled}, {bits_out};"
            )
            self._emit(
                self.iset.add(
                    ".f32", acc, acc, shuffled
                )
            )

        # --------------------------------------------------
        # 3. Warp leaders store their partial sums.
        # --------------------------------------------------
        self.gen.shared(
            smem.name, ".f32", length=warps, align=4
        )

        lane = self.gen.register(".u32")
        warp = self.gen.register(".u32")

        self._emit(f"and.b32 {lane}, {tid}, 31;")
        self._emit(f"shr.u32 {warp}, {tid}, 5;")

        skip_store = self._new_label()
        not_leader = self.gen.register(".pred")

        self._emit(
            f"setp.ne.u32 {not_leader}, {lane}, 0;"
        )
        self._emit(
            f"@{not_leader} bra {skip_store};"
        )

        shared_base = self.gen.register(".u32")
        shared_offset = self.gen.register(".u32")
        shared_addr = self.gen.register(".u32")

        self._emit(
            f"mov.u32 {shared_base}, {smem.name};"
        )
        self._emit(
            f"shl.b32 {shared_offset}, {warp}, 2;"
        )
        self._emit(
            f"add.u32 {shared_addr}, "
            f"{shared_base}, {shared_offset};"
        )
        self._emit(
            f"st.shared.f32 [{shared_addr}], {acc};"
        )

        self.gen.label(skip_store)

        # All threads must reach the barrier.
        self._emit("bar.sync 0;")

        # --------------------------------------------------
        # 4. Thread zero combines warp sums in registers.
        # --------------------------------------------------
        done = self._new_label()
        not_zero = self.gen.register(".pred")

        self._emit(f"setp.ne.u32 {not_zero}, {tid}, 0;")
        self._emit(f"@{not_zero} bra {done};")

        base = self.gen.register(".u32")
        self._emit(f"mov.u32 {base}, {smem.name};")

        partials = []

        for i in range(warps):
            reg = self.gen.register(".f32")

            offset = f"+{i * 4}" if i else ""
            self._emit(
                f"ld.shared.f32 {reg}, [{base}{offset}];"
            )
            partials.append(reg)

        # Balanced pairwise reduction.
        while len(partials) > 1:
            next_level = []

            for i in range(0, len(partials), 2):
                dst = self.gen.register(".f32")
                self._emit(
                    self.iset.add(
                        ".f32",
                        dst,
                        partials[i],
                        partials[i + 1],
                    )
                )
                next_level.append(dst)

            partials = next_level

        
        # --------------------------------------------------
        # 5. Final reduction result: sum or mean.
        # --------------------------------------------------

        result = partials[0]

        if final_mode == "mean":
            # Emmy IR:
            # Load mean_keepdim_count[0]
            # Assign v0 = acc0 / in1
            # Write mean = v0

            count_ptr = self.pointers[count_load.input]
            divisor = self.gen.register(".f32")

            self._emit(
                self.iset.load(
                    "global",
                    ".f32",
                    divisor,
                    f"[{count_ptr}]",
                )
            )

            mean_result = self.gen.register(".f32")

            # Correctly rounded floating-point division.
            self._emit(
                f"div.rn.f32 {mean_result}, "
                f"{result}, {divisor};"
            )

            result = mean_result

        # Both sum and mean write one float32 result.
        
        if is_row_reduction:
            # sum_1[row] — one f32 result per CUDA block.
            output_address = self._buffer_address(
                output_write.output,
                row,
            )
        else:
            output_address = self.pointers[
                output_write.output
            ]

        self._emit(
            self.iset.store(
                "global",
                ".f32",
                f"[{output_address}]",
                result,
            )
        )


        self.gen.label(done)

    def _lower_l2norm(self, tile):
        """Lower a row-wise f32 L2 normalization kernel."""

        stmts = tuple(tile.body)
        kinds = tuple(type(s).__name__ for s in stmts)

        expected = (
            "StridedLoop",
            "WarpShuffle",
            "Smem",
            "Cond",
            "Sync",
            "TreeHalve",
            "Assign",
            "Assign",
            "StridedLoop",
        )

        if kinds != expected or len(tile.axes) != 2:
            raise NotImplementedError(
                f"Unsupported L2 normalization structure: {kinds}"
            )

        (
            reduce_loop,
            shuffle,
            smem,
            store_cond,
            sync,
            tree,
            sqrt_stmt,
            reciprocal_stmt,
            output_loop,
        ) = stmts

        row_axis, thread_axis = tile.axes

        reduce_body = tuple(reduce_loop.body)
        output_body = tuple(output_loop.body)

        if (
            tuple(type(s).__name__ for s in reduce_body)
            != ("Load", "Assign", "Accum")
            or tuple(type(s).__name__ for s in output_body)
            != ("Load", "Assign", "Write")
        ):
            raise NotImplementedError(
                "Expected square-accumulate and multiply-write loops"
            )

        load, square, accum = reduce_body
        out_load, out_mul, out_write = output_body

        op = lambda stmt: self._operation_name(stmt.op).lower()

        if (
            op(square) not in ("multiply", "mul")
            or tuple(square.args) != (load.names[0], load.names[0])
            or op(accum) != "add"
            or accum.value != square.name
            or op(sqrt_stmt) != "sqrt"
            or tuple(sqrt_stmt.args) != (accum.name,)
            or op(reciprocal_stmt) != "reciprocal"
            or tuple(reciprocal_stmt.args) != (sqrt_stmt.name,)
            or op(out_mul) not in ("multiply", "mul")
            or tuple(out_mul.args)
            != (out_load.names[0], reciprocal_stmt.name)
            or tuple(out_write.values) != (out_mul.name,)
        ):
            raise NotImplementedError(
                "Unsupported L2 normalization arithmetic"
            )

        if (
            len(load.names) != 1
            or len(out_load.names) != 1
            or load.input != out_load.input
            or len(load.index) != 2
            or len(out_load.index) != 2
            or len(out_write.index) != 2
            or out_write.atomic
            or out_write.output not in self.pointers
        ):
            raise NotImplementedError(
                "Unsupported L2 normalization buffers"
            )

        def index_names(index):
            return tuple(
                v.name if type(v).__name__ == "Var" else None
                for v in index
            )

        if (
            index_names(load.index)
            != (row_axis.name, reduce_loop.axis.name)
            or index_names(out_load.index)
            != (row_axis.name, output_loop.axis.name)
            or index_names(out_write.index)
            != (row_axis.name, output_loop.axis.name)
        ):
            raise NotImplementedError(
                "Unsupported L2 normalization tensor indexing"
            )

        rows = getattr(row_axis.extent, "value", row_axis.extent)
        threads = getattr(
            thread_axis.extent, "value", thread_axis.extent
        )
        columns = getattr(
            reduce_loop.axis.extent, "value", reduce_loop.axis.extent
        )
        out_columns = getattr(
            output_loop.axis.extent, "value", output_loop.axis.extent
        )

        if (
            type(rows) is not int
            or type(threads) is not int
            or type(columns) is not int
            or min(rows, threads, columns) <= 0
            or columns != out_columns
            or threads % 32 != 0
        ):
            raise NotImplementedError(
                "Unsupported L2 normalization dimensions"
            )

        warps = threads // 32

        if (
            tuple(self.kernel.launch["grid"]) != (rows, 1, 1)
            or tuple(self.kernel.launch["block"]) != (threads, 1, 1)
            or warps <= 0
            or warps & (warps - 1)
        ):
            raise NotImplementedError(
                "Unsupported L2 normalization launch"
            )

        def valid_loop(loop):
            return (
                loop.end is None
                and loop.seed
                and type(loop.start).__name__ == "Var"
                and loop.start.name == thread_axis.name
                and type(loop.step).__name__ == "Literal"
                and loop.step.value == threads
            )

        if not valid_loop(reduce_loop) or not valid_loop(output_loop):
            raise NotImplementedError(
                "Unsupported L2 normalization loop configuration"
            )

        if (
            shuffle.length != 32
            or tuple(shuffle.state) != (accum.name,)
            or len(shuffle.combine_states) != 1
            or op(shuffle.combine_states[0]) != "add"
            or tuple(smem.extents) != (warps,)
            or smem.dtype != "float"
            or tuple(tree.bufs) != (smem.name,)
            or tuple(tree.state) != (accum.name,)
            or tree.length != warps
            or len(tree.combine_states) != 1
            or op(tree.combine_states[0]) != "add"
            or sync.warp
            or sync.barrier_id != 0
        ):
            raise NotImplementedError(
                "Unsupported L2 normalization reduction"
            )

        if (
            type(store_cond.cond).__name__ != "BinaryExpr"
            or store_cond.cond.op != "=="
            or getattr(store_cond.cond.left, "name", None) != "lane"
            or getattr(store_cond.cond.right, "value", None) != 0
            or store_cond.else_body
            or len(store_cond.body) != 1
        ):
            raise NotImplementedError(
                "Unsupported warp leader condition"
            )

        partial_write = store_cond.body[0]

        if (
            type(partial_write).__name__ != "Write"
            or partial_write.output != smem.name
            or tuple(partial_write.values) != (accum.name,)
            or index_names(partial_write.index) != ("warp",)
            or partial_write.atomic
        ):
            raise NotImplementedError(
                "Unsupported warp partial write"
            )

        args = {
            a["name"]: a
            for a in self.kernel.metadata["selected_kernel"]["args"]
        }

        if (
            load.input not in args
            or out_write.output not in args
            or args[load.input]["shape"] != [rows, columns]
            or args[out_write.output]["shape"] != [rows, columns]
            or args[load.input]["dtype"] != "f32"
            or args[out_write.output]["dtype"] != "f32"
        ):
            raise NotImplementedError(
                "Expected matching contiguous row-major f32 matrices"
            )

        # --------------------------------------------------
        # Thread and row coordinates
        # --------------------------------------------------

        tid = self.gen.register(".u32")
        row = self.gen.register(".u32")

        self._emit(f"mov.u32 {tid}, %tid.x;")
        self._emit(f"mov.u32 {row}, %ctaid.x;")

        self.axes[thread_axis.name] = tid
        self.axes[row_axis.name] = row

        # Shared array stores four warp partial sums and,
        # later, the broadcast normalization factor.
        self.gen.shared(smem.name, ".f32", length=warps, align=4)

        # --------------------------------------------------
        # First loop: sum of squares
        # --------------------------------------------------

        col = self.gen.register(".u32")
        acc = self.gen.register(".f32")

        self._emit(f"mov.u32 {col}, {tid};")
        self._emit(f"mov.f32 {acc}, 0f00000000;")

        first_loop = self._new_label()
        first_end = self._new_label()

        self.gen.label(first_loop)

        stop = self.gen.register(".pred")
        self._emit(f"setp.ge.u32 {stop}, {col}, {columns};")
        self._emit(f"@{stop} bra {first_end};")

        linear = self.gen.register(".u32")
        self._emit(
            f"mad.lo.u32 {linear}, {row}, {columns}, {col};"
        )

        address = self._buffer_address(load.input, linear)
        value = self.gen.register(".f32")
        squared = self.gen.register(".f32")

        self._emit(
            self.iset.load("global", ".f32", value, f"[{address}]")
        )
        self._emit(
            self.iset.mul(".f32", squared, value, value)
        )
        self._emit(
            self.iset.add(".f32", acc, acc, squared)
        )
        self._emit(f"add.u32 {col}, {col}, {threads};")
        self._emit(f"bra {first_loop};")

        self.gen.label(first_end)

        # --------------------------------------------------
        # Warp reduction
        # --------------------------------------------------

        bits = self.gen.register(".b32")
        shuffled_bits = self.gen.register(".b32")
        other = self.gen.register(".f32")

        for mask in (16, 8, 4, 2, 1):
            self._emit(f"mov.b32 {bits}, {acc};")
            self._emit(
                f"shfl.sync.bfly.b32 {shuffled_bits}, "
                f"{bits}, {mask}, 0x1f, 0xffffffff;"
            )
            self._emit(f"mov.b32 {other}, {shuffled_bits};")
            self._emit(self.iset.add(".f32", acc, acc, other))

        # --------------------------------------------------
        # One lane per warp writes shared memory
        # --------------------------------------------------

        lane = self.gen.register(".u32")
        warp = self.gen.register(".u32")

        self._emit(f"and.b32 {lane}, {tid}, 31;")
        self._emit(f"shr.u32 {warp}, {tid}, 5;")

        shared_base = self.gen.register(".u32")
        self._emit(f"mov.u32 {shared_base}, {smem.name};")

        skip_partial = self._new_label()
        not_leader = self.gen.register(".pred")

        self._emit(
            f"setp.ne.u32 {not_leader}, {lane}, 0;"
        )
        self._emit(f"@{not_leader} bra {skip_partial};")

        shared_offset = self.gen.register(".u32")
        shared_address = self.gen.register(".u32")

        self._emit(
            f"shl.b32 {shared_offset}, {warp}, 2;"
        )
        self._emit(
            f"add.u32 {shared_address}, "
            f"{shared_base}, {shared_offset};"
        )
        self._emit(
            f"st.shared.f32 [{shared_address}], {acc};"
        )

        self.gen.label(skip_partial)

        self._emit("bar.sync 0;")

        # --------------------------------------------------
        # Thread zero calculates inverse L2 norm
        # --------------------------------------------------

        skip_combine = self._new_label()
        not_zero = self.gen.register(".pred")

        self._emit(f"setp.ne.u32 {not_zero}, {tid}, 0;")
        self._emit(f"@{not_zero} bra {skip_combine};")

        partials = []

        for i in range(warps):
            p = self.gen.register(".f32")
            offset = f"+{i * 4}" if i else ""

            self._emit(
                f"ld.shared.f32 {p}, [{shared_base}{offset}];"
            )
            partials.append(p)

        while len(partials) > 1:
            next_level = []
            for i in range(0, len(partials), 2):
                dst = self.gen.register(".f32")
                self._emit(
                    self.iset.add(
                        ".f32", dst, partials[i], partials[i + 1]
                    )
                )
                next_level.append(dst)
            partials = next_level

        norm = self.gen.register(".f32")
        inverse_norm = self.gen.register(".f32")

        self._emit(
            f"sqrt.rn.f32 {norm}, {partials[0]};"
        )
        self._emit(
            f"div.rn.f32 {inverse_norm}, "
            f"0f3f800000, {norm};"
        )

        # Reuse shared element zero to broadcast the result.
        self._emit(
            f"st.shared.f32 [{shared_base}], {inverse_norm};"
        )

        self.gen.label(skip_combine)

        # Every thread waits until thread zero has published
        # the normalization factor.
        self._emit("bar.sync 0;")

        factor = self.gen.register(".f32")
        self._emit(
            f"ld.shared.f32 {factor}, [{shared_base}];"
        )

        # --------------------------------------------------
        # Second loop: normalize and write every element
        # --------------------------------------------------

        out_col = self.gen.register(".u32")
        self._emit(f"mov.u32 {out_col}, {tid};")

        second_loop = self._new_label()
        second_end = self._new_label()

        self.gen.label(second_loop)

        stop_out = self.gen.register(".pred")

        self._emit(
            f"setp.ge.u32 {stop_out}, {out_col}, {columns};"
        )
        self._emit(
            f"@{stop_out} bra {second_end};"
        )

        out_linear = self.gen.register(".u32")

        self._emit(
            f"mad.lo.u32 {out_linear}, "
            f"{row}, {columns}, {out_col};"
        )

        input_address = self._buffer_address(
            out_load.input, out_linear
        )

        output_address = self._buffer_address(
            out_write.output, out_linear
        )

        element = self.gen.register(".f32")
        normalized = self.gen.register(".f32")

        self._emit(
            self.iset.load(
                "global", ".f32", element,
                f"[{input_address}]"
            )
        )
        self._emit(
            self.iset.mul(
                ".f32", normalized, element, factor
            )
        )
        self._emit(
            self.iset.store(
                "global", ".f32",
                f"[{output_address}]", normalized
            )
        )

        self._emit(
            f"add.u32 {out_col}, {out_col}, {threads};"
        )
        self._emit(f"bra {second_loop};")

        self.gen.label(second_end)

    def _lower_rmsnorm(self, tile):
        """Lower row-wise weighted f32 RMSNorm."""

        stmts = tuple(tile.body)
        kinds = tuple(type(s).__name__ for s in stmts)

        expected = (
            "StridedLoop", "WarpShuffle", "Smem",
            "Cond", "Sync", "TreeHalve",
            "Load", "Assign", "Assign",
            "Load", "Assign", "Assign", "Assign",
            "StridedLoop",
        )

        if kinds != expected or len(tile.axes) != 2:
            raise NotImplementedError(
                f"Unsupported RMSNorm IR: {kinds}"
            )

        (
            reduce_loop, shuffle, smem,
            store_cond, sync, tree,
            count_load, reciprocal_count, mean_mul,
            epsilon_load, epsilon_add, sqrt_stmt,
            reciprocal_norm, output_loop,
        ) = stmts

        row_axis, thread_axis = tile.axes

        def op(stmt):
            return self._operation_name(stmt.op).lower()

        def names(index):
            return tuple(
                getattr(expr, "name", None)
                for expr in index
            )

        def extent(axis):
            return getattr(axis.extent, "value", axis.extent)

        def scalar_zero(load):
            return (
                len(load.names) == 1
                and len(load.index) == 1
                and type(load.index[0]).__name__ == "Literal"
                and load.index[0].value == 0
                and load.dtype.name == "f32"
            )

        reduce_body = tuple(reduce_loop.body)
        output_body = tuple(output_loop.body)

        if (
            tuple(type(s).__name__ for s in reduce_body)
            != ("Load", "Assign", "Accum")
            or tuple(type(s).__name__ for s in output_body)
            != ("Load", "Assign", "Load", "Assign", "Write")
        ):
            raise NotImplementedError(
                "Unsupported RMSNorm loop bodies"
            )

        load_x, square, accum = reduce_body
        out_x, mul_factor, load_w, mul_weight, write = output_body

        if (
            op(square) not in ("multiply", "mul")
            or tuple(square.args)
                != (load_x.names[0], load_x.names[0])
            or op(accum) != "add"
            or accum.value != square.name
            or op(reciprocal_count) != "reciprocal"
            or tuple(reciprocal_count.args)
                != (count_load.names[0],)
            or op(mean_mul) not in ("multiply", "mul")
            or tuple(mean_mul.args)
                != (accum.name, reciprocal_count.name)
            or op(epsilon_add) != "add"
            or tuple(epsilon_add.args)
                != (epsilon_load.names[0], mean_mul.name)
            or op(sqrt_stmt) != "sqrt"
            or tuple(sqrt_stmt.args) != (epsilon_add.name,)
            or op(reciprocal_norm) != "reciprocal"
            or tuple(reciprocal_norm.args)
                != (sqrt_stmt.name,)
            or op(mul_factor) not in ("multiply", "mul")
            or tuple(mul_factor.args)
                != (out_x.names[0], reciprocal_norm.name)
            or op(mul_weight) not in ("multiply", "mul")
            or tuple(mul_weight.args)
                != (load_w.names[0], mul_factor.name)
            or tuple(write.values) != (mul_weight.name,)
        ):
            raise NotImplementedError(
                "Unsupported RMSNorm arithmetic"
            )

        if not scalar_zero(count_load) or not scalar_zero(epsilon_load):
            raise NotImplementedError(
                "Expected scalar count and epsilon loads"
            )

        if (
            len(load_x.names) != 1
            or len(out_x.names) != 1
            or len(load_w.names) != 1
            or load_x.dtype.name != "f32"
            or out_x.dtype.name != "f32"
            or load_w.dtype.name != "f32"
            or names(load_x.index)
                != (row_axis.name, reduce_loop.axis.name)
            or names(out_x.index)
                != (row_axis.name, output_loop.axis.name)
            or names(load_w.index)
                != (output_loop.axis.name,)
            or names(write.index)
                != (row_axis.name, output_loop.axis.name)
            or load_x.input != out_x.input
            or write.atomic
        ):
            raise NotImplementedError(
                "Unsupported RMSNorm tensor access"
            )

        rows = extent(row_axis)
        threads = extent(thread_axis)
        columns = extent(reduce_loop.axis)

        if (
            not all(
                type(v) is int and v > 0
                for v in (rows, threads, columns)
            )
            or extent(output_loop.axis) != columns
            or threads % 32 != 0
            or tuple(self.kernel.launch["grid"])
                != (rows, 1, 1)
            or tuple(self.kernel.launch["block"])
                != (threads, 1, 1)
        ):
            raise NotImplementedError(
                "Unsupported RMSNorm launch geometry"
            )

        def valid_loop(loop):
            return (
                loop.end is None
                and loop.seed
                and type(loop.start).__name__ == "Var"
                and loop.start.name == thread_axis.name
                and type(loop.step).__name__ == "Literal"
                and loop.step.value == threads
            )

        if not valid_loop(reduce_loop) or not valid_loop(output_loop):
            raise NotImplementedError(
                "Unsupported RMSNorm loop configuration"
            )

        warps = threads // 32

        if (
            shuffle.length != 32
            or warps & (warps - 1)
            or tuple(shuffle.state) != (accum.name,)
            or len(shuffle.combine_states) != 1
            or op(shuffle.combine_states[0]) != "add"
            or tuple(smem.extents) != (warps,)
            or smem.dtype != "float"
            or tuple(tree.bufs) != (smem.name,)
            or tuple(tree.state) != (accum.name,)
            or tree.length != warps
            or len(tree.combine_states) != 1
            or op(tree.combine_states[0]) != "add"
            or sync.warp
            or sync.barrier_id != 0
        ):
            raise NotImplementedError(
                "Unsupported RMSNorm cooperative reduction"
            )

        if (
            type(store_cond.cond).__name__ != "BinaryExpr"
            or store_cond.cond.op != "=="
            or getattr(store_cond.cond.left, "name", None) != "lane"
            or getattr(store_cond.cond.right, "value", None) != 0
            or store_cond.else_body
            or len(store_cond.body) != 1
        ):
            raise NotImplementedError(
                "Unsupported RMSNorm warp leader condition"
            )

        partial_write = store_cond.body[0]

        if (
            type(partial_write).__name__ != "Write"
            or partial_write.output != smem.name
            or names(partial_write.index) != ("warp",)
            or tuple(partial_write.values) != (accum.name,)
            or partial_write.atomic
        ):
            raise NotImplementedError(
                "Unsupported RMSNorm shared-memory write"
            )

        args = {
            arg["name"]: arg
            for arg in self.kernel.metadata["selected_kernel"]["args"]
        }

        required_shapes = {
            load_x.input: [rows, columns],
            load_w.input: [columns],
            count_load.input: [1],
            epsilon_load.input: [1],
            write.output: [rows, columns],
        }

        for buffer, shape in required_shapes.items():
            if (
                buffer not in args
                or args[buffer]["dtype"] != "f32"
                or args[buffer]["shape"] != shape
                or buffer not in self.pointers
            ):
                raise NotImplementedError(
                    f"Unsupported RMSNorm tensor: {buffer}"
                )

        # --------------------------------------------------
        # 1. Block-local thread and row indices
        # --------------------------------------------------

        tid = self.gen.register(".u32")
        row = self.gen.register(".u32")

        self._emit(f"mov.u32 {tid}, %tid.x;")
        self._emit(f"mov.u32 {row}, %ctaid.x;")

        self.axes[thread_axis.name] = tid
        self.axes[row_axis.name] = row

        # Four warp partials. Slot zero will later
        # contain the inverse RMS factor.
        self.gen.shared(
            smem.name, ".f32", length=warps, align=4
        )

        # --------------------------------------------------
        # 2. First loop: accumulate x * x
        # --------------------------------------------------

        col = self.gen.register(".u32")
        acc = self.gen.register(".f32")

        self._emit(f"mov.u32 {col}, {tid};")
        self._emit(f"mov.f32 {acc}, 0f00000000;")

        loop_start = self._new_label()
        loop_end = self._new_label()
        stop = self.gen.register(".pred")

        self.gen.label(loop_start)
        self._emit(
            f"setp.ge.u32 {stop}, {col}, {columns};"
        )
        self._emit(f"@{stop} bra {loop_end};")

        linear = self.gen.register(".u32")
        self._emit(
            f"mad.lo.u32 {linear}, {row}, {columns}, {col};"
        )

        x_address = self._buffer_address(load_x.input, linear)

        value = self.gen.register(".f32")
        squared = self.gen.register(".f32")

        self._emit(
            self.iset.load(
                "global", ".f32", value, f"[{x_address}]"
            )
        )
        self._emit(
            self.iset.mul(".f32", squared, value, value)
        )
        self._emit(
            self.iset.add(".f32", acc, acc, squared)
        )
        self._emit(f"add.u32 {col}, {col}, {threads};")
        self._emit(f"bra {loop_start};")
        self.gen.label(loop_end)

        # --------------------------------------------------
        # 3. Warp butterfly reduction
        # --------------------------------------------------

        bits = self.gen.register(".b32")
        other_bits = self.gen.register(".b32")
        other = self.gen.register(".f32")

        for mask in (16, 8, 4, 2, 1):
            self._emit(f"mov.b32 {bits}, {acc};")
            self._emit(
                f"shfl.sync.bfly.b32 {other_bits}, "
                f"{bits}, {mask}, 0x1f, 0xffffffff;"
            )
            self._emit(f"mov.b32 {other}, {other_bits};")
            self._emit(
                self.iset.add(".f32", acc, acc, other)
            )

        # --------------------------------------------------
        # 4. Warp leaders publish their partial sums
        # --------------------------------------------------

        lane = self.gen.register(".u32")
        warp = self.gen.register(".u32")
        shared_base = self.gen.register(".u32")

        self._emit(f"and.b32 {lane}, {tid}, 31;")
        self._emit(f"shr.u32 {warp}, {tid}, 5;")
        self._emit(f"mov.u32 {shared_base}, {smem.name};")

        skip_store = self._new_label()
        not_leader = self.gen.register(".pred")

        self._emit(
            f"setp.ne.u32 {not_leader}, {lane}, 0;"
        )
        self._emit(f"@{not_leader} bra {skip_store};")

        offset = self.gen.register(".u32")
        shared_address = self.gen.register(".u32")

        self._emit(f"shl.b32 {offset}, {warp}, 2;")
        self._emit(
            f"add.u32 {shared_address}, "
            f"{shared_base}, {offset};"
        )
        self._emit(
            f"st.shared.f32 [{shared_address}], {acc};"
        )

        self.gen.label(skip_store)
        self._emit("bar.sync 0;")

        # --------------------------------------------------
        # 5. Thread zero computes the inverse RMS factor
        # --------------------------------------------------

        skip_combine = self._new_label()
        not_zero = self.gen.register(".pred")

        self._emit(f"setp.ne.u32 {not_zero}, {tid}, 0;")
        self._emit(f"@{not_zero} bra {skip_combine};")

        partials = []

        for i in range(warps):
            part = self.gen.register(".f32")
            byte_offset = f"+{i * 4}" if i else ""

            self._emit(
                f"ld.shared.f32 {part}, "
                f"[{shared_base}{byte_offset}];"
            )
            partials.append(part)

        while len(partials) > 1:
            next_level = []

            for i in range(0, len(partials), 2):
                dst = self.gen.register(".f32")
                self._emit(
                    self.iset.add(
                        ".f32",
                        dst,
                        partials[i],
                        partials[i + 1],
                    )
                )
                next_level.append(dst)

            partials = next_level

        reduced_sum = partials[0]

        # Load mean_count[0] and add_c1[0]
        count = self.gen.register(".f32")
        epsilon = self.gen.register(".f32")

        self._emit(
            self.iset.load(
                "global", ".f32", count,
                f"[{self.pointers[count_load.input]}]"
            )
        )
        self._emit(
            self.iset.load(
                "global", ".f32", epsilon,
                f"[{self.pointers[epsilon_load.input]}]"
            )
        )

        reciprocal = self.gen.register(".f32")
        mean_square = self.gen.register(".f32")
        stabilized = self.gen.register(".f32")
        root = self.gen.register(".f32")
        inverse_rms = self.gen.register(".f32")

        self._emit(
            f"div.rn.f32 {reciprocal}, "
            f"0f3f800000, {count};"
        )
        self._emit(
            self.iset.mul(
                ".f32",
                mean_square,
                reduced_sum,
                reciprocal,
            )
        )
        self._emit(
            self.iset.add(
                ".f32",
                stabilized,
                epsilon,
                mean_square,
            )
        )
        self._emit(
            f"sqrt.rn.f32 {root}, {stabilized};"
        )
        self._emit(
            f"div.rn.f32 {inverse_rms}, "
            f"0f3f800000, {root};"
        )

        # Publish the normalization factor to other threads.
        self._emit(
            f"st.shared.f32 [{shared_base}], {inverse_rms};"
        )

        self.gen.label(skip_combine)

        # This barrier is essential. Without it, other threads
        # could read a stale warp partial instead of inverse_rms.
        self._emit("bar.sync 0;")

        factor = self.gen.register(".f32")
        self._emit(
            f"ld.shared.f32 {factor}, [{shared_base}];"
        )

        # --------------------------------------------------
        # 6. Second loop: output = weight * (x * factor)
        # --------------------------------------------------

        out_col = self.gen.register(".u32")
        self._emit(f"mov.u32 {out_col}, {tid};")

        output_start = self._new_label()
        output_end = self._new_label()
        stop_out = self.gen.register(".pred")

        self.gen.label(output_start)
        self._emit(
            f"setp.ge.u32 {stop_out}, {out_col}, {columns};"
        )
        self._emit(f"@{stop_out} bra {output_end};")

        out_linear = self.gen.register(".u32")

        self._emit(
            f"mad.lo.u32 {out_linear}, "
            f"{row}, {columns}, {out_col};"
        )

        x_addr = self._buffer_address(
            out_x.input, out_linear
        )
        w_addr = self._buffer_address(
            load_w.input, out_col
        )
        y_addr = self._buffer_address(
            write.output, out_linear
        )

        x_value = self.gen.register(".f32")
        w_value = self.gen.register(".f32")
        scaled = self.gen.register(".f32")
        weighted = self.gen.register(".f32")

        self._emit(
            self.iset.load(
                "global", ".f32", x_value, f"[{x_addr}]"
            )
        )
        self._emit(
            self.iset.mul(
                ".f32", scaled, x_value, factor
            )
        )
        self._emit(
            self.iset.load(
                "global", ".f32", w_value, f"[{w_addr}]"
            )
        )
        self._emit(
            self.iset.mul(
                ".f32", weighted, w_value, scaled
            )
        )
        self._emit(
            self.iset.store(
                "global", ".f32",
                f"[{y_addr}]", weighted
            )
        )

        self._emit(
            f"add.u32 {out_col}, {out_col}, {threads};"
        )
        self._emit(f"bra {output_start};")

        self.gen.label(output_end)

    def _lower_softmax(self, tile):
        """Lower the supported online-softmax IR using stable passes.

        One CUDA block per row, 128 threads per block.
        Supported: contiguous row-major f32 matrices.
        """

        stmts = tuple(tile.body)
        kinds = tuple(type(s).__name__ for s in stmts)

        expected = (
            "StridedLoop", "WarpShuffle",
            "Smem", "Smem", "Cond", "Sync",
            "TreeHalve", "Assign", "StridedLoop",
        )

        if kinds != expected or len(tile.axes) != 2:
            raise NotImplementedError(
                f"Unsupported softmax structure: {kinds}"
            )

        (
            online_loop, shuffle, max_smem, sum_smem,
            partial_store, sync, tree, reciprocal, output_loop,
        ) = stmts

        row_axis, thread_axis = tile.axes

        def extent(axis):
            return getattr(axis.extent, "value", axis.extent)

        def operation(stmt):
            return self._operation_name(stmt.op).lower()

        def index_names(index):
            return tuple(
                getattr(expr, "name", None)
                for expr in index
            )

        rows = extent(row_axis)
        threads = extent(thread_axis)
        columns = extent(online_loop.axis)

        if (
            not all(
                type(n) is int and n > 0
                for n in (rows, threads, columns)
            )
            or threads % 32 != 0
            or extent(output_loop.axis) != columns
            or tuple(self.kernel.launch["grid"])
                != (rows, 1, 1)
            or tuple(self.kernel.launch["block"])
                != (threads, 1, 1)
        ):
            raise NotImplementedError(
                "Unsupported softmax launch geometry"
            )

        def valid_loop(loop):
            return (
                loop.end is None
                and loop.seed
                and getattr(loop.start, "name", None)
                    == thread_axis.name
                and type(loop.step).__name__ == "Literal"
                and loop.step.value == threads
            )

        if not valid_loop(online_loop) or not valid_loop(output_loop):
            raise NotImplementedError(
                "Unsupported softmax loop indexing"
            )

        first_body = tuple(online_loop.body)
        final_body = tuple(output_loop.body)

        first_kinds = tuple(
            type(s).__name__ for s in first_body
        )

        final_kinds = tuple(
            type(s).__name__ for s in final_body
        )

        if first_kinds != (
            "Let", "Load",
            "Assign", "Assign", "Assign", "Assign",
            "Assign", "Assign", "Assign", 
            "Accum", "Accum",
        ):
            raise NotImplementedError(
                f"Unsupported online softmax body: {first_kinds}"
            )

        if final_kinds != (
            "Load", "Assign", "Assign", "Assign", "Write"
        ):
            raise NotImplementedError(
                f"Unsupported softmax output body: {final_kinds}"
            )

        (
            one, input_load,
            running_max, old_delta, old_exp,
            new_delta, new_exp, scaled_sum,
            new_term, 
            sum_accum, max_accum,
        ) = first_body

        # Some Emmy versions may order the online-softmax
        # assignments differently. Validate the actual IR
        # rather than guessing how to translate it.
        if (
            getattr(one, "name", None) != "one"
            or getattr(getattr(one, "value", None), "value", None)
                != 1.0
            or operation(running_max) != "maximum"
            or operation(old_delta) != "subtract"
            or operation(old_exp) != "exp"
            or operation(new_delta) != "subtract"
            or operation(new_exp) != "exp"
            or operation(scaled_sum) != "multiply"
            or operation(new_term) != "multiply"
            or operation(sum_accum) != "add"
            or operation(max_accum) != "maximum"
        ):
            raise NotImplementedError(
                "Unsupported online softmax arithmetic"
            )

        # Check the accumulator dependencies. This ensures
        # our stable-pass replacement is semantically valid.
        if (
            tuple(running_max.args)
                != (max_accum.name, input_load.names[0])
            or tuple(old_delta.args)
                != (max_accum.name, running_max.name)
            or tuple(old_exp.args) != (old_delta.name,)
            or tuple(new_delta.args)
                != (input_load.names[0], running_max.name)
            or tuple(new_exp.args) != (new_delta.name,)
            or tuple(scaled_sum.args)
                != (sum_accum.name, old_exp.name)
            or tuple(new_term.args)
                != (new_exp.name, one.name)
            or sum_accum.base != scaled_sum.name
            or sum_accum.value != new_term.name
            or max_accum.value != input_load.names[0]
        ):
            raise NotImplementedError(
                "Unsupported online softmax dependencies"
            )

        out_load, subtract, exp_stmt, multiply, write = final_body

        if (
            len(input_load.names) != 1
            or len(out_load.names) != 1
            or input_load.dtype.name != "f32"
            or out_load.dtype.name != "f32"
            or input_load.input != out_load.input
            or index_names(input_load.index)
                != (row_axis.name, online_loop.axis.name)
            or index_names(out_load.index)
                != (row_axis.name, output_loop.axis.name)
            or index_names(write.index)
                != (row_axis.name, output_loop.axis.name)
            or operation(reciprocal) != "reciprocal"
            or tuple(reciprocal.args) != (sum_accum.name,)
            or operation(subtract) != "subtract"
            or tuple(subtract.args)
                != (out_load.names[0], max_accum.name)
            or operation(exp_stmt) != "exp"
            or tuple(exp_stmt.args) != (subtract.name,)
            or operation(multiply) != "multiply"
            or tuple(multiply.args)
                != (reciprocal.name, exp_stmt.name)
            or tuple(write.values) != (multiply.name,)
            or write.atomic
        ):
            raise NotImplementedError(
                "Unsupported softmax output expressions"
            )

        warps = threads // 32

        if (
            shuffle.length != 32
            or warps & (warps - 1)
            or tuple(shuffle.state)
                != (max_accum.name, sum_accum.name)
            or tuple(tree.state)
                != (max_accum.name, sum_accum.name)
            or tuple(tree.bufs)
                != (max_smem.name, sum_smem.name)
            or tree.length != warps
            or tuple(max_smem.extents) != (warps,)
            or tuple(sum_smem.extents) != (warps,)
            or max_smem.dtype != "float"
            or sum_smem.dtype != "float"
            or sync.warp
            or sync.barrier_id != 0
        ):
            raise NotImplementedError(
                "Unsupported softmax reduction layout"
            )

        if (
            getattr(partial_store.cond, "op", None) != "=="
            or getattr(partial_store.cond.left, "name", None)
                != "lane"
            or getattr(partial_store.cond.right, "value", None)
                != 0
            or partial_store.else_body
            or len(partial_store.body) != 2
            or tuple(
                getattr(s, "output", None)
                for s in partial_store.body
            ) != (max_smem.name, sum_smem.name)
            or tuple(
                tuple(s.values) for s in partial_store.body
            ) != ((max_accum.name,), (sum_accum.name,))
        ):
            raise NotImplementedError(
                "Unsupported softmax warp partial writes"
            )

        # The two accumulator pairs use max, subtract, exp,
        # multiply, add, and copy to merge running softmax
        # states across lanes and warps.
        merge_ops = (
            "maximum", "subtract", "exp", "subtract",
            "exp", "multiply", "multiply", "add", "copy",
        )

        for merge in (shuffle, tree):
            if tuple(
                operation(s) for s in merge.combine_states
            ) != merge_ops:
                raise NotImplementedError(
                    "Unsupported online softmax merge"
                )

        args = {
            a["name"]: a
            for a in self.kernel.metadata["selected_kernel"]["args"]
        }

        for name in (input_load.input, write.output):
            if (
                name not in args
                or args[name]["dtype"] != "f32"
                or args[name]["shape"] != [rows, columns]
                or name not in self.pointers
            ):
                raise NotImplementedError(
                    f"Unsupported softmax tensor: {name}"
                )

        # -----------------------------------------------
        # CUDA block-local coordinates and shared arrays
        # -----------------------------------------------

        tid = self.gen.register(".u32")
        row = self.gen.register(".u32")

        self._emit(f"mov.u32 {tid}, %tid.x;")
        self._emit(f"mov.u32 {row}, %ctaid.x;")

        self.axes[thread_axis.name] = tid
        self.axes[row_axis.name] = row

        self.gen.shared(
            max_smem.name, ".f32", length=warps, align=4
        )
        self.gen.shared(
            sum_smem.name, ".f32", length=warps, align=4
        )

        # Convert a natural exponential using:
        # exp(x) = 2 ** (x * log2(e)).
        #
        # PTX has ex2.approx.f32, not exp.f32.
        def emit_exp(dst, src):
            exponent = self.gen.register(".f32")
            self._emit(
                f"mul.f32 {exponent}, "
                f"{src}, 0f3fb8aa3b;"
            )
            self._emit(
                f"ex2.approx.f32 {dst}, {exponent};"
            )

        # -----------------------------------------------
        # Cooperative reduction helper:
        # max or sum across all 128 threads.
        # -----------------------------------------------

        def reduce_row(mode, shared_name, previous_max=None):
            if mode not in ("max", "sum"):
                raise ValueError("Unsupported reduction mode")

            col = self.gen.register(".u32")
            acc = self.gen.register(".f32")

            self._emit(f"mov.u32 {col}, {tid};")

            if mode == "max":
                self._emit(
                    f"mov.f32 {acc}, 0fff800000;"
                )
            else:
                self._emit(
                    f"mov.f32 {acc}, 0f00000000;"
                )

            start = self._new_label()
            end = self._new_label()
            stop = self.gen.register(".pred")

            self.gen.label(start)

            self._emit(
                f"setp.ge.u32 {stop}, {col}, {columns};"
            )
            self._emit(f"@{stop} bra {end};")

            linear = self.gen.register(".u32")
            self._emit(
                f"mad.lo.u32 {linear}, "
                f"{row}, {columns}, {col};"
            )

            address = self._buffer_address(
                input_load.input, linear
            )

            value = self.gen.register(".f32")
            self._emit(
                self.iset.load(
                    "global", ".f32", value,
                    f"[{address}]"
                )
            )

            if mode == "max":
                self._emit(
                    f"max.f32 {acc}, {acc}, {value};"
                )
            else:
                delta = self.gen.register(".f32")
                exp_value = self.gen.register(".f32")

                self._emit(
                    f"sub.f32 {delta}, "
                    f"{value}, {previous_max};"
                )
                emit_exp(exp_value, delta)

                self._emit(
                    self.iset.add(
                        ".f32", acc, acc, exp_value
                    )
                )

            self._emit(
                f"add.u32 {col}, {col}, {threads};"
            )
            self._emit(f"bra {start};")

            self.gen.label(end)

            # Warp-level butterfly reduction.
            bits = self.gen.register(".b32")
            shuffled_bits = self.gen.register(".b32")
            other = self.gen.register(".f32")

            for mask in (16, 8, 4, 2, 1):
                self._emit(f"mov.b32 {bits}, {acc};")
                self._emit(
                    f"shfl.sync.bfly.b32 {shuffled_bits}, "
                    f"{bits}, {mask}, 0x1f, 0xffffffff;"
                )
                self._emit(
                    f"mov.b32 {other}, {shuffled_bits};"
                )

                if mode == "max":
                    self._emit(
                        f"max.f32 {acc}, {acc}, {other};"
                    )
                else:
                    self._emit(
                        self.iset.add(
                            ".f32", acc, acc, other
                        )
                    )

            # One partial reduction per warp.
            lane = self.gen.register(".u32")
            warp = self.gen.register(".u32")
            shared_base = self.gen.register(".u32")

            self._emit(f"and.b32 {lane}, {tid}, 31;")
            self._emit(f"shr.u32 {warp}, {tid}, 5;")
            self._emit(
                f"mov.u32 {shared_base}, {shared_name};"
            )

            skip_store = self._new_label()
            not_leader = self.gen.register(".pred")

            self._emit(
                f"setp.ne.u32 {not_leader}, {lane}, 0;"
            )
            self._emit(
                f"@{not_leader} bra {skip_store};"
            )

            offset = self.gen.register(".u32")
            shared_address = self.gen.register(".u32")

            self._emit(
                f"shl.b32 {offset}, {warp}, 2;"
            )
            self._emit(
                f"add.u32 {shared_address}, "
                f"{shared_base}, {offset};"
            )
            self._emit(
                f"st.shared.f32 [{shared_address}], {acc};"
            )

            self.gen.label(skip_store)

            # All warp partials are now ready.
            self._emit("bar.sync 0;")

            skip_combine = self._new_label()
            not_zero = self.gen.register(".pred")

            self._emit(
                f"setp.ne.u32 {not_zero}, {tid}, 0;"
            )
            self._emit(
                f"@{not_zero} bra {skip_combine};"
            )

            partials = []

            for i in range(warps):
                part = self.gen.register(".f32")
                byte_offset = f"+{i * 4}" if i else ""

                self._emit(
                    f"ld.shared.f32 {part}, "
                    f"[{shared_base}{byte_offset}];"
                )
                partials.append(part)

            # Balanced cross-warp reduction in thread zero.
            while len(partials) > 1:
                next_level = []

                for i in range(0, len(partials), 2):
                    combined = self.gen.register(".f32")

                    if mode == "max":
                        self._emit(
                            f"max.f32 {combined}, "
                            f"{partials[i]}, {partials[i + 1]};"
                        )
                    else:
                        self._emit(
                            self.iset.add(
                                ".f32", combined,
                                partials[i], partials[i + 1]
                            )
                        )

                    next_level.append(combined)

                partials = next_level

            # Replace shared slot zero with the final result.
            self._emit(
                f"st.shared.f32 "
                f"[{shared_base}], {partials[0]};"
            )

            self.gen.label(skip_combine)

            # Broadcast the final result to every thread.
            self._emit("bar.sync 0;")

            result = self.gen.register(".f32")
            self._emit(
                f"ld.shared.f32 {result}, "
                f"[{shared_base}];"
            )

            return result

        # -----------------------------------------------
        # Pass 1: maximum of the row
        # -----------------------------------------------

        row_max = reduce_row("max", max_smem.name)

        # -----------------------------------------------
        # Pass 2: sum(exp(x - row_max))
        # -----------------------------------------------

        row_sum = reduce_row(
            "sum", sum_smem.name, previous_max=row_max
        )

        inv_sum = self.gen.register(".f32")

        self._emit(
            f"div.rn.f32 {inv_sum}, "
            f"0f3f800000, {row_sum};"
        )

        # -----------------------------------------------
        # Pass 3: normalized output
        # -----------------------------------------------

        out_col = self.gen.register(".u32")
        self._emit(f"mov.u32 {out_col}, {tid};")

        output_start = self._new_label()
        output_end = self._new_label()
        output_stop = self.gen.register(".pred")

        self.gen.label(output_start)

        self._emit(
            f"setp.ge.u32 {output_stop}, "
            f"{out_col}, {columns};"
        )
        self._emit(
            f"@{output_stop} bra {output_end};"
        )

        output_linear = self.gen.register(".u32")

        self._emit(
            f"mad.lo.u32 {output_linear}, "
            f"{row}, {columns}, {out_col};"
        )

        input_address = self._buffer_address(
            out_load.input, output_linear
        )

        output_address = self._buffer_address(
            write.output, output_linear
        )

        value = self.gen.register(".f32")
        delta = self.gen.register(".f32")
        exponential = self.gen.register(".f32")
        normalized = self.gen.register(".f32")

        self._emit(
            self.iset.load(
                "global", ".f32", value,
                f"[{input_address}]"
            )
        )
        self._emit(
            f"sub.f32 {delta}, {value}, {row_max};"
        )
        emit_exp(exponential, delta)

        self._emit(
            self.iset.mul(
                ".f32", normalized, inv_sum, exponential
            )
        )

        self._emit(
            self.iset.store(
                "global", ".f32",
                f"[{output_address}]", normalized
            )
        )

        self._emit(
            f"add.u32 {out_col}, {out_col}, {threads};"
        )
        self._emit(f"bra {output_start};")

        self.gen.label(output_end)

    # ==================================================================
    # General multi-axis lowering (matmul and attention-style kernels)
    #
    # Walks the statement list in order, one translation per statement
    # kind. Index expressions evaluate to either a Python int (constant
    # folded) or a u32 register; predicates are .pred registers.
    # ==================================================================

    _GENERAL_KINDS = frozenset({
        "Load", "Assign", "Let", "Init", "Accum", "Loop", "StridedLoop",
        "Cond", "Sync", "Smem", "CpAsyncCopy", "CpAsyncCommit",
        "CpAsyncWait", "Write", "WarpShuffle", "TreeHalve",
    })

    _MAX_UNROLL = 64
    _MAX_UNROLL_STATEMENTS = 1024

    @staticmethod
    def _f32_imm(value):
        """PTX hex float literal for an f32 value (handles +-inf)."""
        bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
        return f"0f{bits:08X}"

    @staticmethod
    def _is_pred(value):
        return isinstance(value, str) and value.startswith("%p")

    @staticmethod
    def _imm(value):
        return str(value & 0xFFFFFFFF)

    def _greg(self, value):
        """Materialize an int or register result into a u32 register."""
        if isinstance(value, int):
            reg = self.gen.register(".u32")
            self._emit(f"mov.u32 {reg}, {self._imm(value)};")
            return reg
        return value

    def _gpred(self, value):
        """Convert an int/u32/pred expression result to a pred (or bool)."""
        if isinstance(value, int):
            return bool(value)
        if self._is_pred(value):
            return value
        pred = self.gen.register(".pred")
        self._emit(f"setp.ne.u32 {pred}, {value}, 0;")
        return pred

    def _garith(self, op, left, right):
        """Integer arithmetic over ints/registers with constant folding."""
        if isinstance(left, int) and isinstance(right, int):
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left // right
            if op == "%":
                return left % right
            if op == "&":
                return left & right
            if op == "|":
                return left | right
            if op == "^":
                return left ^ right
            if op == "<<":
                return left << right
            if op == ">>":
                return left >> right
            raise NotImplementedError(f"Index operator: {op}")

        if self._is_pred(left) or self._is_pred(right):
            raise NotImplementedError(
                f"Arithmetic operator {op} applied to a predicate"
            )

        if op in ("+", "*", "&", "|", "^") and isinstance(left, int):
            left, right = right, left

        if op in ("+", "-", "|", "^", "<<", ">>") and right == 0:
            return left
        if op == "*" and isinstance(right, int):
            if right == 0:
                return 0
            if right == 1:
                return left
        if op == "/" and right == 1:
            return left
        if op == "%" and right == 1:
            return 0

        mnemonic = {
            "+": "add.u32",
            "-": "sub.u32",
            "*": "mul.lo.u32",
            "/": "div.u32",
            "%": "rem.u32",
            "&": "and.b32",
            "|": "or.b32",
            "^": "xor.b32",
            "<<": "shl.b32",
            ">>": "shr.u32",
        }.get(op)

        if mnemonic is None:
            raise NotImplementedError(f"Index operator: {op}")

        left = self._greg(left)
        right = self._imm(right) if isinstance(right, int) else right
        dst = self.gen.register(".u32")
        self._emit(f"{mnemonic} {dst}, {left}, {right};")
        return dst

    def _gcompare(self, op, left, right):
        names = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge",
                 "==": "eq", "!=": "ne"}

        if isinstance(left, int) and isinstance(right, int):
            return int({
                "<": left < right, "<=": left <= right,
                ">": left > right, ">=": left >= right,
                "==": left == right, "!=": left != right,
            }[op])

        left = self._greg(left)
        right = self._imm(right) if isinstance(right, int) else right
        pred = self.gen.register(".pred")
        self._emit(f"setp.{names[op]}.u32 {pred}, {left}, {right};")
        return pred

    def _glogic(self, op, left, right):
        left = self._gpred(left)
        right = self._gpred(right)

        if isinstance(left, bool) or isinstance(right, bool):
            if op == "&&":
                if left is False or right is False:
                    return 0
                return right if isinstance(left, bool) else left
            if left is True or right is True:
                return 1
            return right if isinstance(left, bool) else left

        pred = self.gen.register(".pred")
        mnemonic = "and.pred" if op == "&&" else "or.pred"
        self._emit(f"{mnemonic} {pred}, {left}, {right};")
        return pred

    def _gexpr(self, expr):
        """Lower an index/predicate expression to an int or register."""
        kind = type(expr).__name__

        if kind == "Literal":
            value = expr.value
            if isinstance(value, bool):
                return int(value)
            if type(value) is int:
                return value
            raise NotImplementedError(
                f"Non-integer literal in index expression: {value!r}"
            )

        if kind == "Var":
            if expr.name in self.axes:
                return self.axes[expr.name]
            raise NotImplementedError(
                f"Unknown index variable: {expr.name}"
            )

        if kind == "Builtin":
            if expr.name == "threadIdx.x":
                return self.tid
            if expr.name == "blockIdx.x":
                return self.bid
            raise NotImplementedError(f"Builtin: {expr.name}")

        if kind == "BinaryExpr":
            op = expr.op
            left = self._gexpr(expr.left)
            right = self._gexpr(expr.right)

            if op in ("<", "<=", ">", ">=", "==", "!="):
                return self._gcompare(op, left, right)
            if op in ("&&", "||"):
                return self._glogic(op, left, right)
            return self._garith(op, left, right)

        if kind == "TernaryExpr":
            cond = self._gpred(self._gexpr(expr.cond))
            if isinstance(cond, bool):
                return self._gexpr(
                    expr.if_true if cond else expr.if_false
                )

            when_true = self._gexpr(expr.if_true)
            when_false = self._gexpr(expr.if_false)

            if (
                isinstance(when_true, int)
                and isinstance(when_false, int)
                and when_true == when_false
            ):
                return when_true

            dst = self.gen.register(".u32")
            true_op = (
                self._imm(when_true)
                if isinstance(when_true, int) else when_true
            )
            false_op = (
                self._imm(when_false)
                if isinstance(when_false, int) else when_false
            )
            self._emit(
                f"selp.u32 {dst}, {true_op}, {false_op}, {cond};"
            )
            return dst

        raise NotImplementedError(f"Index expression: {kind}")

    def _flatten(self, index, dims):
        """Row-major flat element offset of a coordinate tuple."""
        index = tuple(index)
        dims = tuple(dims)

        if len(index) == 0:
            return 0

        if len(index) != len(dims):
            raise NotImplementedError(
                f"Index rank {len(index)} does not match "
                f"buffer rank {len(dims)}"
            )

        flat = self._gexpr(index[0])
        for coord, dim in zip(index[1:], dims[1:]):
            flat = self._garith("*", flat, dim)
            flat = self._garith("+", flat, self._gexpr(coord))
        return flat

    def _buffer_space(self, name):
        if name in self.smem:
            return "shared", self.smem[name]["dims"]
        if name in self.pointers:
            return "global", self.shapes[name]
        raise ValueError(f"Unknown buffer: {name}")

    def _gmem_ref(self, space, name, flat):
        """Return (base register, constant byte offset) for element flat."""
        base = (
            self.pointers[name] if space == "global"
            else self.smem[name]["base"]
        )

        if isinstance(flat, int):
            return base, 4 * flat

        if space == "global":
            offset = self.gen.register(".u64")
            address = self.gen.register(".u64")
            self._emit(f"mul.wide.u32 {offset}, {flat}, 4;")
            self._emit(f"add.u64 {address}, {base}, {offset};")
        else:
            address = self.gen.register(".u32")
            self._emit(f"mad.lo.u32 {address}, {flat}, 4, {base};")

        return address, 0

    @staticmethod
    def _operand(base, offset):
        return f"[{base}+{offset}]" if offset else f"[{base}]"

    def _new_general_label(self, hint):
        self._label_counter += 1
        return f"L_{hint}_{self._label_counter}"

    @staticmethod
    def _dim_value(dim):
        value = getattr(dim, "value", dim)
        if type(value) is not int or value <= 0:
            raise ValueError("Extents must be positive integers")
        return value

    def _walk(self, statements):
        """Yield every statement in a body, descending into control flow."""
        for stmt in statements:
            yield stmt
            kind = type(stmt).__name__
            if kind in ("Loop", "StridedLoop"):
                yield from self._walk(stmt.body)
            elif kind == "Cond":
                yield from self._walk(stmt.body)
                yield from self._walk(stmt.else_body)

    # --------------------------- statements ---------------------------

    def _gload(self, stmt):
        # Shared-memory loads carry no dtype; every shared array is f32.
        if stmt.dtype is not None and stmt.dtype.name != "f32":
            raise NotImplementedError("Only f32 loads supported")

        space, dims = self._buffer_space(stmt.input)
        flat = self._flatten(stmt.index, dims)
        base, offset = self._gmem_ref(space, stmt.input, flat)

        # A grouped load reads consecutive elements of the last axis.
        for lane, name in enumerate(stmt.names):
            reg = self.gen.register(".f32")
            self._emit(
                f"ld.{space}.f32 {reg}, "
                f"{self._operand(base, offset + 4 * lane)};"
            )
            self.values[name] = reg

    def _gwrite(self, stmt):
        if stmt.value_dtype.name != "f32":
            raise NotImplementedError("Only f32 writes supported")
        if str(stmt.swizzle).upper() != "NONE":
            raise NotImplementedError("Swizzled writes are not supported")

        space, dims = self._buffer_space(stmt.output)

        if stmt.atomic and space != "global":
            raise NotImplementedError("Atomic writes require global memory")

        flat = self._flatten(stmt.index, dims)
        base, offset = self._gmem_ref(space, stmt.output, flat)

        for lane, name in enumerate(stmt.values):
            if name not in self.values:
                raise ValueError(f"Write references undefined value: {name}")

            target = self._operand(base, offset + 4 * lane)
            value = self.values[name]

            if stmt.atomic:
                self._emit(f"red.global.add.f32 {target}, {value};")
            else:
                self._emit(f"st.{space}.f32 {target}, {value};")

    def _glet(self, stmt):
        dtype = stmt.dtype.name if stmt.dtype is not None else "f32"
        value = stmt.value

        if dtype == "f32":
            if type(value).__name__ != "Literal":
                raise NotImplementedError("Let f32 requires a literal")
            reg = self.gen.register(".f32")
            self._emit(f"mov.f32 {reg}, {self._f32_imm(value.value)};")
            self.values[stmt.name] = reg
            return

        # Integer binding: a precomputed index usable in later expressions.
        self.axes[stmt.name] = self._gexpr(value)

    def _ginit(self, stmt):
        if stmt.dtype.name != "f32":
            raise NotImplementedError("Only f32 Init supported")
        reg = self.gen.register(".f32")
        self._emit(f"mov.f32 {reg}, {self._f32_imm(stmt.identity)};")
        self.values[stmt.name] = reg
        self.carried.add(stmt.name)

    _ACCUM = {
        "add": ("add.f32", 0.0),
        "sum": ("add.f32", 0.0),
        "multiply": ("mul.f32", 1.0),
        "mul": ("mul.f32", 1.0),
        "maximum": ("max.f32", float("-inf")),
        "max": ("max.f32", float("-inf")),
        "minimum": ("min.f32", float("inf")),
        "min": ("min.f32", float("inf")),
    }

    def _accum_spec(self, stmt):
        op = self._operation_name(stmt.op).lower()
        if op not in self._ACCUM:
            raise NotImplementedError(f"Unsupported accumulator op: {op}")
        return self._ACCUM[op]

    def _seed_accum(self, stmt):
        _, identity = self._accum_spec(stmt)
        reg = self.gen.register(".f32")
        self._emit(f"mov.f32 {reg}, {self._f32_imm(identity)};")
        self.values[stmt.name] = reg
        return reg

    def _gaccum(self, stmt):
        if stmt.dtype is None or stmt.dtype.name != "f32":
            raise NotImplementedError("Only f32 accumulators supported")

        mnemonic, _ = self._accum_spec(stmt)
        reg = self.values.get(stmt.name)
        if reg is None:
            reg = self._seed_accum(stmt)

        if stmt.value not in self.values:
            raise ValueError(f"Accum references undefined value: {stmt.value}")

        # base, when present, replaces the previous accumulator value
        # (online-softmax rescaling); otherwise fold into the accumulator.
        if stmt.base is not None:
            if stmt.base not in self.values:
                raise ValueError(f"Accum base is undefined: {stmt.base}")
            previous = self.values[stmt.base]
        else:
            previous = reg

        self._emit(f"{mnemonic} {reg}, {previous}, {self.values[stmt.value]};")

    def _seed_loop_accums(self, axis_name, body):
        """Seed accumulators that reduce over this loop's axis."""
        started = []
        for stmt in self._walk(body):
            if type(stmt).__name__ != "Accum":
                continue
            if (
                axis_name in tuple(stmt.axes)
                and stmt.name not in self.carried
                and stmt.name not in self._live_inits
            ):
                self._seed_accum(stmt)
                self._live_inits.add(stmt.name)
                started.append(stmt.name)
        return started

    def _gloop(self, loop):
        kind = type(loop).__name__
        axis = loop.axis.name
        extent = self._dim_value(loop.axis.extent)

        if kind == "Loop":
            start, step, end = 0, 1, extent
        else:
            start = self._gexpr(loop.start)
            step = self._gexpr(loop.step)
            end = extent if loop.end is None else self._gexpr(loop.end)

            if not isinstance(step, int) or step <= 0:
                raise NotImplementedError("Loop step must be a positive constant")

        # Fully unroll small constant leaf loops flagged for unrolling.
        # Loops that contain other loops stay real loops so the emitted
        # PTX does not blow up multiplicatively.
        body_stmts = list(self._walk(loop.body))
        has_inner_loop = any(
            type(s).__name__ in ("Loop", "StridedLoop") for s in body_stmts
        )

        if (
            loop.unroll
            and not has_inner_loop
            and isinstance(start, int)
            and isinstance(end, int)
            and len(range(start, end, step)) <= self._MAX_UNROLL
            and len(range(start, end, step)) * len(body_stmts)
            <= self._MAX_UNROLL_STATEMENTS
        ):
            for value in range(start, end, step):
                started = self._seed_loop_accums(axis, loop.body)
                self.axes[axis] = value
                self._gbody(loop.body)
                for name in started:
                    self._live_inits.discard(name)
            self.axes.pop(axis, None)
            return

        started = self._seed_loop_accums(axis, loop.body)

        counter = self.gen.register(".u32")
        self._emit(f"mov.u32 {counter}, {self._imm(start) if isinstance(start, int) else start};")
        self.axes[axis] = counter

        top = self._new_general_label("loop")
        done = self._new_general_label("loop_end")
        stop = self.gen.register(".pred")

        self.gen.label(top)
        bound = self._imm(end) if isinstance(end, int) else end
        self._emit(f"setp.ge.u32 {stop}, {counter}, {bound};")
        self._emit(f"@{stop} bra {done};")

        self._gbody(loop.body)

        self._emit(f"add.u32 {counter}, {counter}, {step};")
        self._emit(f"bra {top};")
        self.gen.label(done)

        self.axes.pop(axis, None)
        for name in started:
            self._live_inits.discard(name)

    def _gcond(self, stmt):
        cond = self._gpred(self._gexpr(stmt.cond))

        if isinstance(cond, bool):
            self._gbody(stmt.body if cond else stmt.else_body)
            return

        else_label = self._new_general_label("else")
        end_label = self._new_general_label("endif")

        self._emit(f"@!{cond} bra {else_label};")
        self._gbody(stmt.body)

        if tuple(stmt.else_body):
            self._emit(f"bra {end_label};")
            self.gen.label(else_label)
            self._gbody(stmt.else_body)
            self.gen.label(end_label)
        else:
            self.gen.label(else_label)

    def _gsync(self, stmt):
        if stmt.warp:
            self._emit("bar.warp.sync 0xffffffff;")
        elif stmt.count is None:
            self._emit(f"bar.sync {stmt.barrier_id};")
        else:
            self._emit(f"bar.sync {stmt.barrier_id}, {stmt.count};")

    def _gsmem(self, stmt):
        if stmt.name in self.smem:
            return

        if stmt.dtype not in ("float", "f32"):
            raise NotImplementedError("Only f32 shared arrays supported")

        dims = tuple(self._dim_value(e) for e in stmt.extents)
        count = math.prod(dims)
        align = max(int(stmt.align or 0), 4)

        base = self.gen.register(".u32")

        if self._smem_dynamic:
            offset = -(-self._smem_pool_used // align) * align
            self._smem_pool_used = offset + 4 * count
            self._emit(f"mov.u32 {base}, emmy_smem_pool;")
            if offset:
                shifted = self.gen.register(".u32")
                self._emit(f"add.u32 {shifted}, {base}, {offset};")
                base = shifted
        else:
            self.gen.shared(stmt.name, ".f32", length=count, align=align)
            self._emit(f"mov.u32 {base}, {stmt.name};")

        self.smem[stmt.name] = {"base": base, "dims": dims}

    def _gcpasync(self, stmt):
        if str(stmt.swizzle).upper() != "NONE":
            raise NotImplementedError("Swizzled cp.async is not supported")

        nbytes = stmt.nbytes
        if nbytes not in (4, 8, 16):
            raise NotImplementedError("cp.async copies 4, 8 or 16 bytes")

        if stmt.smem not in self.smem:
            raise ValueError(f"cp.async targets undeclared shared array: {stmt.smem}")

        dst_dims = self.smem[stmt.smem]["dims"]

        # Destination = tile origin + per-lane offset (when provided).
        coords = [self._gexpr(e) for e in stmt.smem_index]
        if stmt.lane_index is not None:
            lane = [self._gexpr(e) for e in stmt.lane_index]
            if len(lane) != len(coords):
                raise NotImplementedError("lane_index rank mismatch")
            coords = [self._garith("+", c, l) for c, l in zip(coords, lane)]

        if len(coords) != len(dst_dims):
            raise NotImplementedError("smem_index rank mismatch")

        flat_dst = coords[0] if coords else 0
        for coord, dim in zip(coords[1:], dst_dims[1:]):
            flat_dst = self._garith("+", self._garith("*", flat_dst, dim), coord)

        src_space, src_dims = self._buffer_space(stmt.src)
        if src_space != "global":
            raise NotImplementedError("cp.async source must be global memory")

        flat_src = self._flatten(stmt.src_index, src_dims)

        dst_base, dst_off = self._gmem_ref("shared", stmt.smem, flat_dst)
        src_base, src_off = self._gmem_ref("global", stmt.src, flat_src)

        cache = "cg" if nbytes == 16 else "ca"
        self._emit(
            f"cp.async.{cache}.shared.global "
            f"{self._operand(dst_base, dst_off)}, "
            f"{self._operand(src_base, src_off)}, {nbytes};"
        )

    def _run_combine(self, combine_states):
        """Execute a WarpShuffle/TreeHalve merge program.

        Each entry is an Assign that reads the current state names and the
        partner (state_b) names and rebinds the state. The assigns carry
        dtype=None; _assign treats that as f32.
        """
        for stmt in combine_states:
            if type(stmt).__name__ != "Assign":
                raise NotImplementedError(
                    f"Unsupported combine statement: {type(stmt).__name__}"
                )
            self._assign(stmt)

    def _gwarpshuffle(self, stmt):
        """Fold per-lane states across a warp with an XOR butterfly."""
        if stmt.length != 32:
            raise NotImplementedError("Only 32-lane warp shuffle supported")
        if len(stmt.state) != len(stmt.state_b):
            raise NotImplementedError("WarpShuffle state/state_b mismatch")

        for mask in (16, 8, 4, 2, 1):
            for name, partner in zip(stmt.state, stmt.state_b):
                if name not in self.values:
                    raise ValueError(
                        f"WarpShuffle references undefined state: {name}"
                    )
                bits_in = self.gen.register(".b32")
                bits_out = self.gen.register(".b32")
                shuffled = self.gen.register(".f32")
                self._emit(f"mov.b32 {bits_in}, {self.values[name]};")
                self._emit(
                    f"shfl.sync.bfly.b32 {bits_out}, "
                    f"{bits_in}, {mask}, 0x1f, 0xffffffff;"
                )
                self._emit(f"mov.b32 {shuffled}, {bits_out};")
                self.values[partner] = shuffled
            self._run_combine(stmt.combine_states)

    def _gtreehalve(self, stmt):
        """Combine per-warp partial states in shared memory and broadcast."""
        length = stmt.length
        if type(length) is not int or length <= 0 or (length & (length - 1)):
            raise NotImplementedError(
                "TreeHalve length must be a positive power of two"
            )
        if (
            len(stmt.bufs) != len(stmt.state)
            or len(stmt.state) != len(stmt.state_b)
        ):
            raise NotImplementedError("TreeHalve buffer/state mismatch")

        bases = []
        for buf in stmt.bufs:
            if buf not in self.smem:
                raise ValueError(
                    f"TreeHalve references undeclared shared array: {buf}"
                )
            bases.append(self.smem[buf]["base"])

        # Thread 0 folds the `length` partials left-to-right, then stores the
        # finals back into slot 0. The merge is an associative monoid, so a
        # serial fold is correct.
        skip = self._new_general_label("treehalve_skip")
        not_zero = self.gen.register(".pred")
        self._emit(f"setp.ne.u32 {not_zero}, {self.tid}, 0;")
        self._emit(f"@{not_zero} bra {skip};")

        for name, base in zip(stmt.state, bases):
            reg = self.gen.register(".f32")
            self._emit(f"ld.shared.f32 {reg}, [{base}];")
            self.values[name] = reg

        for slot in range(1, length):
            for partner, base in zip(stmt.state_b, bases):
                reg = self.gen.register(".f32")
                self._emit(f"ld.shared.f32 {reg}, [{base}+{slot * 4}];")
                self.values[partner] = reg
            self._run_combine(stmt.combine_states)

        for name, base in zip(stmt.state, bases):
            self._emit(f"st.shared.f32 [{base}], {self.values[name]};")

        self.gen.label(skip)
        self._emit(f"bar.sync {int(stmt.barrier_id or 0)};")

        # Every thread reads the broadcast result from slot 0.
        for name, base in zip(stmt.state, bases):
            reg = self.gen.register(".f32")
            self._emit(f"ld.shared.f32 {reg}, [{base}];")
            self.values[name] = reg

    def _gbody(self, statements):
        for stmt in statements:
            kind = type(stmt).__name__

            if kind == "Load":
                self._gload(stmt)
            elif kind == "Assign":
                self._assign(stmt)
            elif kind == "Let":
                self._glet(stmt)
            elif kind == "Init":
                self._ginit(stmt)
            elif kind == "Accum":
                self._gaccum(stmt)
            elif kind in ("Loop", "StridedLoop"):
                self._gloop(stmt)
            elif kind == "Cond":
                self._gcond(stmt)
            elif kind == "Sync":
                self._gsync(stmt)
            elif kind == "Smem":
                self._gsmem(stmt)
            elif kind == "WarpShuffle":
                self._gwarpshuffle(stmt)
            elif kind == "TreeHalve":
                self._gtreehalve(stmt)
            elif kind == "CpAsyncCopy":
                self._gcpasync(stmt)
            elif kind == "CpAsyncCommit":
                self._emit("cp.async.commit_group;")
            elif kind == "CpAsyncWait":
                self._emit(f"cp.async.wait_group {int(stmt.group)};")
            elif kind == "Write":
                self._gwrite(stmt)
            else:
                raise NotImplementedError(
                    f"General lowering does not support {kind}"
                )

    # ------------------------- tile + launch --------------------------

    def _decode_axes(self, axes, source):
        """Decode row-major tile axes from a linear index register."""
        remaining = source
        for position, axis in enumerate(reversed(axes)):
            extent = self._dim_value(axis.extent)
            outermost = position == len(axes) - 1

            if extent == 1:
                self.axes[axis.name] = 0
            elif outermost:
                self.axes[axis.name] = remaining
            else:
                self.axes[axis.name] = self._garith("%", remaining, extent)
                remaining = self._garith("/", remaining, extent)

    def _lower_general(self, tile):
        """Lower a multi-axis Tile whose body is generic statements."""
        threads = tile.block_threads
        if type(threads) is not int or threads <= 0:
            raise NotImplementedError("General lowering requires block_threads")
        if getattr(tile, "aux_threads", 0):
            raise NotImplementedError("Auxiliary threads are not supported")

        launch = self.kernel.launch
        block = launch.get("block")
        grid = launch.get("grid")

        if block is None or grid is None:
            raise ValueError("General lowering requires grid/block metadata")

        # Trailing axes spanning block_threads cells are thread axes;
        # the leading axes are decoded from the block index.
        axes = list(tile.axes)
        product = 1
        split = len(axes)
        while split > 0 and product < threads:
            split -= 1
            product *= self._dim_value(axes[split].extent)

        if product != threads:
            raise NotImplementedError(
                "Tile axes do not factor into block_threads"
            )

        block_axes = axes[:split]
        thread_axes = axes[split:]
        block_cells = math.prod(
            self._dim_value(a.extent) for a in block_axes
        )

        block_total = math.prod(int(v) for v in block)
        grid_total = math.prod(int(v) for v in grid)

        if block_total != threads or grid_total != block_cells:
            raise NotImplementedError(
                "Launch geometry does not match the Tile schedule"
            )

        # Shared memory: dynamic pool when the launch provides it,
        # static declarations otherwise.
        total = 0
        for stmt in self._walk(tile.body):
            if type(stmt).__name__ == "Smem":
                align = max(int(stmt.align or 0), 4)
                size = 4 * math.prod(self._dim_value(e) for e in stmt.extents)
                total = -(-total // align) * align + size

        provided = int(launch.get("smem_bytes", 0) or 0)
        self._smem_dynamic = total > 0 and provided >= total

        if self._smem_dynamic:
            self.module_decls.append(
                ".extern .shared .align 16 .b8 emmy_smem_pool[];"
            )

        self.tid = self._linear_launch_index("%tid", "%ntid", block)
        self.bid = self._linear_launch_index("%ctaid", "%nctaid", grid)

        # Warp-cooperative fused kernels address shared partials by lane and
        # warp; the IR refers to these names directly.
        lane = self.gen.register(".u32")
        warp = self.gen.register(".u32")
        self._emit(f"and.b32 {lane}, {self.tid}, 31;")
        self._emit(f"shr.u32 {warp}, {self.tid}, 5;")
        self.axes["lane"] = lane
        self.axes["warp"] = warp

        self._decode_axes(thread_axes, self.tid)
        self._decode_axes(block_axes, self.bid)

        self._gbody(tile.body)

    def _linear_launch_index(self, coord, size, dims):
        """Row-major linear id over a 3-D launch dimension (x fastest).

        A 1-D launch reads the x builtin directly; multidimensional launches
        fold y and z in with the corresponding size builtins.
        """
        reg = self.gen.register(".u32")
        self._emit(f"mov.u32 {reg}, {coord}.x;")

        if int(dims[1]) == 1 and int(dims[2]) == 1:
            return reg

        # linear = x + ntid.x * (y + ntid.y * z)
        y = self.gen.register(".u32")
        nx = self.gen.register(".u32")
        self._emit(f"mov.u32 {y}, {coord}.y;")
        self._emit(f"mov.u32 {nx}, {size}.x;")

        acc = self.gen.register(".u32")
        self._emit(f"mad.lo.u32 {acc}, {y}, {nx}, {reg};")

        if int(dims[2]) == 1:
            return acc

        z = self.gen.register(".u32")
        ny = self.gen.register(".u32")
        nxy = self.gen.register(".u32")
        self._emit(f"mov.u32 {z}, {coord}.z;")
        self._emit(f"mov.u32 {ny}, {size}.y;")
        self._emit(f"mul.lo.u32 {nxy}, {nx}, {ny};")

        acc2 = self.gen.register(".u32")
        self._emit(f"mad.lo.u32 {acc2}, {z}, {nxy}, {acc};")
        return acc2

    def _render_general(self):
        entry = self.gen.render_entry()
        if self.module_decls:
            return "\n".join(self.module_decls) + "\n\n" + entry
        return entry

    def _snapshot(self):
        """Capture emitter and walker state so a failed general attempt can
        be rolled back before falling through to a specialized lowerer."""
        return {
            "instructions": list(self.gen._instructions),
            "declarations": list(self.gen._declarations),
            "register_counts": dict(self.gen._register_counts),
            "values": dict(self.values),
            "axes": dict(self.axes),
            "smem": dict(self.smem),
            "carried": set(self.carried),
            "live_inits": set(self._live_inits),
            "module_decls": list(self.module_decls),
            "label_counter": self._label_counter,
            "smem_pool_used": self._smem_pool_used,
            "smem_dynamic": self._smem_dynamic,
            "tid": self.tid,
            "bid": self.bid,
        }

    def _restore(self, snap):
        self.gen._instructions[:] = snap["instructions"]
        self.gen._declarations[:] = snap["declarations"]
        self.gen._register_counts.clear()
        self.gen._register_counts.update(snap["register_counts"])
        self.values = snap["values"]
        self.axes = snap["axes"]
        self.smem = snap["smem"]
        self.carried = snap["carried"]
        self._live_inits = snap["live_inits"]
        self.module_decls = snap["module_decls"]
        self._label_counter = snap["label_counter"]
        self._smem_pool_used = snap["smem_pool_used"]
        self._smem_dynamic = snap["smem_dynamic"]
        self.tid = snap["tid"]
        self.bid = snap["bid"]

    def lower(self):
        """Lower supported Emmy Kernel IR into PTX."""

        self._prologue()

        if len(self.kernel.body) == 1:
            root = self.kernel.body[0]

            if type(root).__name__ == "Tile":
                kinds = tuple(
                    type(stmt).__name__
                    for stmt in root.body
                )

                softmax_pattern = (
                    "StridedLoop", "WarpShuffle",
                    "Smem", "Smem", "Cond", "Sync",
                    "TreeHalve", "Assign", "StridedLoop",
                )

                rmsnorm_pattern = (
                    "StridedLoop", "WarpShuffle", "Smem",
                    "Cond", "Sync", "TreeHalve",
                    "Load", "Assign", "Assign",
                    "Load", "Assign", "Assign", "Assign",
                    "StridedLoop",
                )

                l2_pattern = (
                    "StridedLoop", "WarpShuffle",
                    "Smem", "Cond", "Sync",
                    "TreeHalve", "Assign",
                    "Assign", "StridedLoop",
                )

                sum_mean_pattern = (
                    "StridedLoop", "WarpShuffle", "Smem",
                    "Cond", "Sync", "TreeHalve", "Cond",
                )

                # Plain sum/mean keeps the specialized lowerer. Its statement
                # list is also in the general set, but the dedicated path is
                # the one already checked on the runner.
                if kinds == sum_mean_pattern:
                    self._lower_sum_reduction(root)
                    return self.gen.render_entry()

                # Fused, matmul, attention, and any other Tile whose
                # statements the walker understands. Hidden variants that
                # do not match an exact skeleton below take this route.
                if set(kinds) <= self._GENERAL_KINDS:
                    snapshot = self._snapshot()
                    try:
                        self._lower_general(root)
                        return self._render_general()
                    except NotImplementedError:
                        self._restore(snapshot)

                if kinds == softmax_pattern:
                    self._lower_softmax(root)
                    return self.gen.render_entry()

                if kinds == rmsnorm_pattern:
                    self._lower_rmsnorm(root)
                    return self.gen.render_entry()

                if kinds == l2_pattern:
                    self._lower_l2norm(root)
                    return self.gen.render_entry()

                if "StridedLoop" in kinds:
                    self._lower_sum_reduction(root)
                    return self.gen.render_entry()

        # 5. Existing pointwise kernels
        self._body(self.kernel.body)
        return self.gen.render_entry()



class PTXEmitter:
    def __init__(self):
        self.constants = PTXConstants(version=PTX_VERSION)
        self.ptx_code = ""
        self._name_counters: dict[str, int] = {}
        self._used_names: set[str] = set()
        self.abi = PTXABI()
        self.memory = PTXMemoryConsistency(target=TARGET_ARCH)
        self.instructions = PTXInstructionSet(
            target=TARGET_ARCH,
            ptx_version=PTX_VERSION,
        )

    @classmethod
    def _is_predefined(cls, name: str) -> bool:
        """Identify protected PTX built-in symbols."""

        if name in PTX_PREDEFINED:
            return True

        # Components of predefined vector special registers.
        if name in {
            f"{base}.{axis}"
            for base in (
                "%tid", "%ntid", "%ctaid", "%nctaid",
                "%clusterid", "%nclusterid",
                "%cluster_ctaid", "%cluster_nctaid",
            )
            for axis in ("x", "y", "z")
        }:
            return True

        return bool(
            re.fullmatch(r"%pm[0-7]", name)
            or re.fullmatch(r"%envreg(?:[0-9]|[12][0-9]|3[01])", name)
            or re.fullmatch(r"%reserved_smem_offset_[0-9]+", name)
        )


    @classmethod
    def _validate_identifier(cls, name: str) -> str:
        """Validate a user-defined PTX identifier."""

        pattern = r"(?:[A-Za-z][A-Za-z0-9_$]*|[_$%][A-Za-z0-9_$]+)"

        if not re.fullmatch(pattern, name, re.ASCII):
            raise ValueError(f"Invalid PTX identifier: {name!r}")

        if name in PTX_INSTRUCTIONS:
            raise ValueError(f"Reserved PTX instruction keyword: {name}")

        if cls._is_predefined(name):
            raise ValueError(f"Reserved PTX built-in identifier: {name}")

        return name

    @classmethod
    def _validate_opcode(cls, opcode: str) -> str:
        """Check the base instruction keyword."""

        base = opcode.split(".", 1)[0]

        if base not in PTX_INSTRUCTIONS:
            raise ValueError(f"Unknown PTX instruction keyword: {base}")

        return opcode


    @classmethod
    def _validate_directive(cls, directive: str) -> str:
        """Check a PTX directive keyword."""

        if directive not in PTX_DIRECTIVES:
            raise ValueError(f"Unknown PTX directive: {directive}")

        return directive

    def _new_identifier(self, prefix: str = "%r") -> str:
        """Generate a unique PTX identifier."""

        self._validate_identifier(f"{prefix}0")

        number = self._name_counters.get(prefix, 0)

        while True:
            name = f"{prefix}{number}"
            number += 1

            if name not in self._used_names:
                self._used_names.add(name)
                self._name_counters[prefix] = number

                logger.debug("Allocated PTX identifier: %s", name)

                return name
                
    def _generate_ptx_header(self) -> None:
        """Generate the required PTX module header."""

        logger.debug(
            "Generating PTX header: version=%s, target=%s",
            PTX_VERSION,
            TARGET_ARCH,
        )

        self.ptx_code += f".version {PTX_VERSION}\n"
        self.ptx_code += f".target {TARGET_ARCH}\n"
        self.ptx_code += f".address_size {ADDRESS_SIZE}\n"

    def _emit_comment(self, message: str) -> None:
        """Generate a single-line PTX comment."""

        # Normalize newlines so the comment stays on one line.
        message = " ".join(message.splitlines())

        # PTX source must contain ASCII characters.
        logger.debug("Generating PTX comment: %s", message)
        message.encode("ascii")
        self.ptx_code += f"// {message}\n"

    def _emit_block_comment(self, message: str) -> None:
        """Generate a multi-line PTX block comment."""
        if "*/" in message or "/*" in message:
            raise ValueError("Nested or terminating block-comment markers are not allowed.")
        logger.debug("Generating PTX block comment: %s", message)
        message.encode("ascii")
        self.ptx_code += f"/*\n{message}\n*/\n"

    def _emit_instruction(
        self,
        opcode: str,
        *operands: str | object,
        label: str | None = None,
        predicate: str | None = None,
        negate: bool = False,
    ) -> None:
        """Append a PTX instruction with string or object operands."""

        if not isinstance(opcode, str) or not opcode.strip():
            raise ValueError("PTX instruction opcode cannot be empty.")

        if negate and predicate is None:
            raise ValueError(
                "Cannot negate an instruction without a predicate."
            )

        # Convert operand objects into PTX source strings.
        formatted_operands = tuple(str(op) for op in operands)

        # Validate the generated text, not the original objects.
        for value in (
            opcode,
            *formatted_operands,
            label,
            predicate,
        ):
            if value is not None:
                if not isinstance(value, str):
                    raise TypeError("PTX text must be a string.")
                value.encode("ascii")

        # Validate instruction keyword.
        self._validate_opcode(opcode)

        # Optional label.
        prefix = ""
        if label is not None:
            self._validate_identifier(label)
            prefix = f"{label}:\n"

        # Optional predicate guard.
        guard = ""
        if predicate is not None:
            guard = f"@{'!' if negate else ''}{predicate} "

        operand_text = ", ".join(formatted_operands)

        instruction = f"    {guard}{opcode}"

        if operand_text:
            instruction += f" {operand_text}"

        instruction += ";\n"

        logger.debug(
            "Emitting PTX instruction: %s",
            instruction.strip(),
        )

        self.ptx_code += prefix + instruction

    def compile_ptx(self, input_path: Path) -> str:
        """Compile a supported Emmy kernel directly to PTX."""

        self.ptx_code = ""

        kernel = EmmyIRParser(
            input_path,
            require_meta=True,
        ).parse()

        self._generate_ptx_header()
        self.ptx_code += "\n"

        self.ptx_code += EmmyIRLowering(kernel).lower()

        return self.ptx_code


# --------------------------------------------------
# 3. Command-line interface
# --------------------------------------------------


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: python compile.py <kernel.ir.json>",
            file=sys.stderr,
        )
        return 2

    input_path = Path(sys.argv[1])

    if not input_path.is_file():
        print(
            f"Input file not found: {input_path}",
            file=sys.stderr,
        )
        return 1

    try:
        emitter = PTXEmitter()
        ptx = emitter.compile_ptx(input_path)

        # Official submission contract:
        # PTX goes to stdout, diagnostics go to stderr.
        sys.stdout.write(ptx)
        return 0

    except Exception:
        logger.exception("Compilation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
