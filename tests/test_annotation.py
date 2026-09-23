# pyright: standard

import dataclasses
import importlib.util
import json
from typing import Any, cast

import pytest
import triton.language as tl

import intj
from intj.annotation import UNSET


def kernel_from_source(tmp_path, source):
    path = tmp_path / "annotated_kernel.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel


def kernel_with_params(tmp_path, params, decorator="@triton.jit"):
    return kernel_from_source(
        tmp_path,
        f"import intj\nimport triton\nimport triton.language as tl\n\n"
        f"{decorator}\ndef kernel({params}):\n    pass\n",
    )


def test_public_annotation_exports_and_presets():
    assert intj.INT_TYPES == (tl.int32, tl.int64, tl.uint64)
    assert intj.FLOAT_TYPES == (tl.float32,)
    assert "UNSET" not in intj.__all__
    assert intj.Argument().type is UNSET
    assert intj.Argument(type=None).type is None


def test_annotation_sequences_are_immutable():
    arg = intj.Argument(type=[tl.int64, tl.int32, tl.int64])
    assert isinstance(arg.type, tuple)
    assert hash(arg) == hash(intj.Argument(type=(tl.int64, tl.int32, tl.int64)))


@pytest.mark.parametrize(
    "build",
    [
        lambda: intj.Argument(value=[1]),
        lambda: intj.Constexpr(value=[1]),
    ],
)
def test_annotation_values_reject_unhashable_values(build):
    with pytest.raises(TypeError):
        build()


@pytest.mark.parametrize(
    "fact",
    [intj.EqualTo(True), intj.Aligned(cast(Any, 16.0))],
)
def test_assume_fact_values_require_exact_int(fact):
    with pytest.raises(TypeError, match="Fact value must be an int"):
        intj.Assume(fact)


@pytest.mark.parametrize(
    "build,match",
    [
        (lambda: intj.Argument(type=[]), "type sequence must not be empty"),
        (lambda: intj.Argument(bind_value=cast(Any, "tensor")), "bind_value"),
        (
            lambda: intj.Argument(value=1, bind_value=intj.BindValue.POINTER),
            "mutually exclusive",
        ),
        (lambda: intj.Assume(intj.EqualTo(2)), r"EqualTo\(1\)"),
        (lambda: intj.Assume(intj.Aligned(8)), r"Aligned\(16\)"),
        (lambda: intj.Assume(intj.PointerRange(64)), r"PointerRange\(32\)"),
    ],
)
def test_annotation_constructor_errors(build, match):
    with pytest.raises((TypeError, ValueError), match=match):
        build()


def test_annotation_sources_merge_field_by_field(tmp_path):
    from intj.annotation import _resolve_annotations

    kernel = kernel_with_params(
        tmp_path,
        "x: intj.Argument(type=tl.int32), y",
        "@triton.jit(do_not_specialize_on_alignment=['x'])",
    )
    params = _resolve_annotations(kernel, {"x": intj.Argument(specialize=intj.NEVER)})
    x = params[0].annotation
    assert x.types == ("i32",)
    assert (x.equal_to_one, x.aligned_16, x.pointer_range_32) == (
        "never", "never", "never"
    )


def test_annotation_sources_merge_equivalent_type_sets(tmp_path):
    from intj.annotation import _resolve_annotations

    kernel = kernel_with_params(
        tmp_path, "x: intj.Argument(type=[tl.int64, tl.int32, tl.int64])"
    )
    resolved = _resolve_annotations(
        kernel, {"x": intj.Argument(type=[tl.int32, tl.int64])}
    )
    assert resolved[0].annotation.types == ("i32", "i64")


def test_none_and_unset_remain_distinct_after_merge(tmp_path):
    from intj.annotation import _resolve_annotations

    kernel = kernel_with_params(tmp_path, "x, y")
    params = _resolve_annotations(kernel, {"x": None, "y": intj.Argument()})
    assert params[0].annotation.types == (None,)
    assert params[1].annotation.types is None


def test_raw_inline_shorthands_and_decorator_modes(tmp_path):
    from intj.annotation import _resolve_annotations

    kernel = kernel_with_params(
        tmp_path,
        "x: tl.int32, y: tl.constexpr, z: None",
        "@triton.jit(do_not_specialize=['x'], do_not_specialize_on_alignment=['z'])",
    )
    x, y, z = (param.annotation for param in _resolve_annotations(kernel, None))
    assert (x.kind, x.types) == ("argument", ("i32",))
    assert (x.equal_to_one, x.aligned_16, x.pointer_range_32) == ("never", "never", "auto")
    assert (y.kind, y.types) == ("constexpr", None)
    assert (z.types, z.equal_to_one, z.aligned_16, z.pointer_range_32) == (
        (None,), "auto", "never", "auto"
    )


def test_assumptions_merge_independently_and_types_sort(tmp_path):
    from intj.annotation import _resolve_annotations

    kernel = kernel_with_params(
        tmp_path,
        "x: intj.Argument(type=[tl.pointer_type(tl.float32), None, tl.pointer_type(tl.int32)], "
        "specialize=intj.Assume(intj.Aligned(16)))",
    )
    annotation = _resolve_annotations(kernel, {"x": intj.Argument(
        specialize=intj.Assume(intj.PointerRange(32))
    )})[0].annotation
    assert annotation.types == (None, "*fp32", "*i32")
    assert annotation.facts == (("aligned_16", 16), ("pointer_range_32", 32))
    assert (annotation.equal_to_one, annotation.aligned_16, annotation.pointer_range_32) == (
        "auto", "assume", "assume"
    )


def test_equivalent_inputs_have_one_canonical_form(tmp_path):
    from intj.annotation import _resolve_annotations

    kernel = kernel_with_params(tmp_path, "x")
    one = _resolve_annotations(kernel, {"x": intj.Argument(
        type=[tl.pointer_type(tl.float64), tl.pointer_type(tl.int32),
              tl.pointer_type(tl.float64)],
        specialize=intj.Assume(intj.Aligned(16), intj.PointerRange(32)),
    )})
    two = _resolve_annotations(kernel, {"x": intj.Argument(
        type=[tl.pointer_type(tl.int32), tl.pointer_type(tl.float64)],
        specialize=intj.Assume(intj.PointerRange(32), intj.Aligned(16)),
    )})
    assert one == two
    assert hash(one) == hash(two)
    json.dumps(dataclasses.asdict(one[0].annotation), sort_keys=True)


def test_canonical_baked_values_preserve_python_kind_and_bits(tmp_path):
    from intj.annotation import _canonical_value, _resolve_annotations

    assert _canonical_value(None) == ("none",)
    assert _canonical_value(False) == ("bool", False)
    assert _canonical_value(0) == ("int", "0")
    assert _canonical_value(-0.0) == ("float64", "8000000000000000")
    assert _canonical_value(0.0) == ("float64", "0000000000000000")
    kernel = kernel_with_params(tmp_path, "x: tl.constexpr")
    no_value = _resolve_annotations(kernel, None)[0].annotation
    baked_none = _resolve_annotations(kernel, {"x": intj.Constexpr(value=None)})[0]
    assert no_value.baked_value == ()
    assert baked_none.annotation.baked_value == ("none",)
    assert baked_none.baked is None


@pytest.mark.parametrize(
    "params,extra,decorator,field",
    [
        ("x", {"missing": intj.Argument()}, "@triton.jit", "missing"),
        ("x: tl.int32", {"x": intj.Argument(type=intj.AUTO)}, "@triton.jit", "type"),
        ("x", {"x": intj.Argument(specialize=intj.AUTO)},
         "@triton.jit(do_not_specialize=['x'])", "equal_to_one"),
        ("x: intj.Argument()", {"x": intj.Constexpr()}, "@triton.jit", "kind"),
        ("x: intj.Argument(value=False)", {"x": intj.Argument(value=0)}, "@triton.jit", "value"),
    ],
)
def test_annotation_sources_reject_conflicts(tmp_path, params, extra, decorator, field):
    from intj.annotation import _resolve_annotations

    kernel = kernel_with_params(tmp_path, params, decorator)
    with pytest.raises(ValueError, match=field):
        _resolve_annotations(kernel, extra)


@pytest.mark.parametrize(
    "params,extra,match",
    [
        ("x", {"x": intj.Argument(type=tl.float16)}, "unsupported"),
        ("x: tl.constexpr", {"x": intj.Constexpr(type=tl.float32)}, "unsupported"),
        ("x", {"x": intj.Argument(type=[tl.int32, intj.AUTO])}, "AUTO"),
        ("x", {"x": intj.Argument(type=None, specialize=intj.Assume(intj.EqualTo(1)))}, "dead"),
        ("x", {"x": intj.Argument(type=tl.int32, specialize=intj.Assume(
            intj.EqualTo(1), intj.Aligned(16)))}, "impossible"),
        ("x", {"x": intj.Argument(value=object())}, "baked values"),
        ("x", {"x": intj.Argument(type=tl.pointer_type(tl.int32), value=0,
                                    specialize=intj.NEVER)}, "pointer"),
        ("x", {"x": intj.Argument(bind_value=intj.BindValue.POINTER,
                                    specialize=intj.NEVER)}, "pointer type"),
        ("x", {"x": intj.Argument(type=[tl.pointer_type(tl.int32)],
                                    bind_value=intj.BindValue.POINTER,
                                    specialize=intj.NEVER)}, "pointer type"),
        ("x", {"x": intj.Argument(type=None, bind_value=intj.BindValue.POINTER,
                                    specialize=intj.NEVER)}, "pointer type"),
        ("x", {"x": intj.Argument(type=tl.int32,
                                    bind_value=intj.BindValue.TENSOR,
                                    specialize=intj.NEVER)}, "pointer type"),
        ("x: tl.constexpr", {"x": intj.Constexpr(type=[tl.int32, tl.float64],
                                                  power_of_two_or_zero=True)}, "power_of_two_or_zero"),
    ],
)
def test_annotations_reject_invalid_inputs(tmp_path, params, extra, match):
    from intj.annotation import _resolve_annotations

    kernel = kernel_with_params(tmp_path, params)
    with pytest.raises(ValueError, match=match):
        _resolve_annotations(kernel, extra)


def test_external_fact_subclass_is_rejected(tmp_path):
    from intj.annotation import _resolve_annotations

    class ExternalFact(intj.Fact):
        pass

    kernel = kernel_with_params(tmp_path, "x")
    forged = object.__new__(intj.Assume)
    object.__setattr__(forged, "facts", (ExternalFact(),))
    with pytest.raises(TypeError, match="Fact"):
        _resolve_annotations(kernel, {"x": intj.Argument(specialize=forged)})


def test_baked_values_resolve_exact_types_and_reject_bad_facts(tmp_path):
    from intj.annotation import _resolve_annotations

    kernel = kernel_with_params(tmp_path, "x, y: tl.constexpr")
    x, y = _resolve_annotations(kernel, {
        "x": intj.Argument(type=[tl.int64, tl.int32], value=17, specialize=intj.NEVER),
        "y": intj.Constexpr(type=tl.int64, value=17),
    })
    assert (x.annotation.types, x.annotation.baked_value) == (("i32",), ("int", "17"))
    assert (y.annotation.types, y.annotation.baked_value) == (("i64",), ("int", "17"))
    with pytest.raises(ValueError, match="violates aligned_16"):
        _resolve_annotations(kernel, {"x": intj.Argument(
            type=tl.int32, value=17, specialize=intj.Assume(intj.Aligned(16))
        )})


def test_baked_constexpr_chooses_narrowest_matching_type(tmp_path):
    from intj.annotation import _resolve_annotations

    kernel = kernel_with_params(tmp_path, "x: tl.constexpr")
    resolved = _resolve_annotations(kernel, {"x": intj.Constexpr(
        type=[tl.int64, tl.int32], value=17
    )})[0]
    assert resolved.annotation.types == ("i32",)
    assert resolved.annotation.baked_value == ("int", "17")


def test_make_launcher_resolves_before_target_discovery(tmp_path, monkeypatch):
    import intj.launcher as launcher

    kernel = kernel_with_params(tmp_path, "x: intj.Argument(type=tl.float16)")
    monkeypatch.setattr(launcher, "_current_target", lambda: pytest.fail("target discovered"))
    with pytest.raises(ValueError, match="unsupported"):
        intj.make_launcher(kernel)


@pytest.mark.parametrize(
    "widths,fixed_device,expected_offsets,device_offset,nwords",
    [
        ((8, 4, 2, 1), False, (0, 8, 12, 14), 15, 2),
        ((1, 8, 1, 4, 2), False, (14, 0, 15, 8, 12), 16, 3),
        ((2, 2, 1), True, (0, 2, 4), None, 1),
        ((), True, (), None, 0),
    ],
)
def test_layout_key_fields_are_grouped_by_descending_alignment(
    widths, fixed_device, expected_offsets, device_offset, nwords
):
    from intj.annotation import DeviceBinding, KeyField, _layout_fields

    layout = _layout_fields(
        tuple((index, KeyField("payload", width)) for index, width in enumerate(widths)),
        DeviceBinding.FIXED if fixed_device else DeviceBinding.NOT_FIXED,
    )
    assert tuple(field.offset for _, field in layout.fields) == expected_offsets
    assert layout.device_offset == device_offset
    assert layout.nwords == nwords


def test_layout_canonical_key_fields_cover_dynamic_and_fixed_parameters(tmp_path):
    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import _render_params

    kernel = kernel_with_params(
        tmp_path,
        "raw: tl.constexpr, typed: tl.constexpr, power: tl.constexpr, "
        "nothing: tl.constexpr, fixed: tl.int32, changing: tl.int32, bound, baked",
    )
    resolved = _resolve_annotations(kernel, {
        "typed": intj.Constexpr(type=[tl.int16, tl.int32]),
        "power": intj.Constexpr(type=tl.int64, power_of_two_or_zero=True),
        "nothing": intj.Constexpr(type=None),
        "fixed": intj.Argument(specialize=intj.NEVER),
        "bound": intj.Argument(type=tl.pointer_type(tl.int32), specialize=intj.NEVER,
                               bind_value=intj.BindValue.TENSOR),
        "baked": intj.Argument(type=tl.int32, specialize=intj.NEVER, value=4),
    })
    params, device_offset, nwords = _render_params(resolved, DeviceBinding.NOT_FIXED)
    fields = [tuple((f.kind, f.width, f.offset) for f in p.annotation.key_fields) for p in params]
    assert fields == [
        (("payload", 8, 0), ("descriptor", 1, 12)),
        (("payload", 4, 8), ("descriptor", 1, 13)),
        (("payload", 1, 14),),
        (),
        (),
        (("descriptor", 1, 15),),
        (),
        (),
    ]
    assert (device_offset, nwords) == (16, 3)
    assert tuple(p.call_index for p in params) == (0, 1, 2, 3, 4, 5, None, None)
    assert hash(params) and json.dumps(dataclasses.asdict(params[0]), sort_keys=True)
