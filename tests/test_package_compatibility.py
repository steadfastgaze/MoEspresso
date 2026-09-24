"""Container identity and supported runtime operation sets."""

import pytest

from moespresso.package.bundle import BUNDLE_KEY_SUFFIX, BundleFormatError, row_order_for_codecs
from moespresso.package.manifest import PACKAGE_FORMAT, PACKAGE_FORMAT_VERSION
from moespresso.runtime.build import UnsupportedRuntimeAdapter, _runtime_adapter_kind
from moespresso.runtime.verify import expected_keys


def test_package_container_identity():
    assert (PACKAGE_FORMAT, PACKAGE_FORMAT_VERSION) == ("mjtq", 1)
    assert BUNDLE_KEY_SUFFIX == "tq_bundle"
    for codec in ("kquant", "iqk", "mxfp4"):
        assert expected_keys({"format": codec, "key_prefix": "experts"}) == ["experts.tq_bundle"]


@pytest.mark.parametrize(
    ("family", "ops", "adapter"),
    [
        ("qwen3_5_moe", ["kquant_dequant", "f32_passthrough"], "qwen_kquant_moe"),
        (
            "deepseek_v4_flash",
            [
                "affine_dequant",
                "fp16_passthrough",
                "kquant_dequant",
                "mxfp8_dequant",
                "raw_dtype_passthrough",
            ],
            "mjtq_dsv4",
        ),
        (
            "deepseek_v4_flash",
            [
                "affine_dequant",
                "fp16_passthrough",
                "kquant_dequant",
                "iqk_dequant",
                "mxfp8_dequant",
                "raw_dtype_passthrough",
            ],
            "mjtq_dsv4",
        ),
        ("qwen4_exp", ["kquant_dequant", "iqk_dequant", "raw_dtype_passthrough"], "qwen4_iqk_moe"),
    ],
)
def test_supported_operation_sets_select_their_adapters(family, ops, adapter):
    manifest = {"architecture": {"family": family}, "required_ops": ops}
    assert _runtime_adapter_kind(manifest) == adapter
    manifest["required_ops"] = [*ops, "unknown_dequant"]
    with pytest.raises(UnsupportedRuntimeAdapter):
        _runtime_adapter_kind(manifest)


def test_bundle_requires_explicit_supported_projection_codecs():
    projections = ("gate_proj", "up_proj", "down_proj")
    with pytest.raises(BundleFormatError, match="codecs must cover exactly"):
        row_order_for_codecs({})
    with pytest.raises(BundleFormatError, match="unsupported expert codec"):
        row_order_for_codecs(dict.fromkeys(projections, "unknown"))
