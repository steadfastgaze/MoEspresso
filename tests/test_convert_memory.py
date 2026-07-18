"""Conversion is memory-bounded: the streaming invariant, as a test.

Rather than sampling RSS (flaky: GC/pagecache/allocator), it spies on the actual
quantize calls and asserts the bounded-working-set invariant deterministically:

  * affine: no single mx.quantize sees more than chunk_bytes of float32, and a
    tensor large enough to need many chunks produces many calls (proves chunking,
    not one big call).
  * experts: TQ-quantize is called once per expert (one [rows,cols] at a time),
    never once on the whole [E,rows,cols] stack.

A coarse psutil peak-RSS check is added as a backstop, but the spy is the real
guarantee. The psutil check remains guarded for direct test-file execution in
an incomplete development environment.
"""

from __future__ import annotations

import json
import struct
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.turboquant")

from moespresso.inventory.build import build_inventory  # noqa: E402
from moespresso.optimize.decide import decide  # noqa: E402
from moespresso.package.plan import package_plan_from_decision  # noqa: E402
from moespresso.package import write as writemod  # noqa: E402
from moespresso.package.write import write_package  # noqa: E402
from moespresso.probe.build import build_probe_evidence  # noqa: E402
from moespresso.runtime.verify import verify_package  # noqa: E402


def _write_safetensors(path, tensors):
    header, blob, off = {}, bytearray(), 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr, dtype=np.float32)
        b = a.tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape),
                        "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


# A model whose tensors are big enough that whole-tensor would spike but chunked
# stays flat: a wide affine (lm_head-like) + a many-expert stack.
N_EXPERTS = 32
EXP_ROWS, EXP_COLS = 128, 256
AFFINE_ROWS, AFFINE_COLS = 4096, 256
ARCH = {"model_type": "qwen3_moe",
        "text_config": {"num_hidden_layers": 1, "hidden_size": AFFINE_COLS,
                        "num_experts": N_EXPERTS, "num_experts_per_tok": 2,
                        "moe_intermediate_size": EXP_COLS,
                        "layer_types": ["full_attention"], "vocab_size": AFFINE_ROWS}}


def _model(d):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    _write_safetensors(d / "model-00001.safetensors", {
        "model.language_model.layers.0.self_attn.q_proj.weight":
            rng.standard_normal((AFFINE_ROWS, AFFINE_COLS)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.gate_up_proj":
            rng.standard_normal((N_EXPERTS, EXP_ROWS, EXP_COLS)).astype(np.float32),
        # down: [out=hidden? no. Rows = EXP_ROWS//2 so every per-expert TQ call
        # (fused halves AND down) sees the same row count, keeping the
        # one-expert-at-a-time assertion exact across all three projections]
        "model.language_model.layers.0.mlp.experts.down_proj":
            rng.standard_normal(
                (N_EXPERTS, EXP_ROWS // 2, EXP_COLS)).astype(np.float32),
    })
    (d / "config.json").write_text(json.dumps(ARCH))


def _decision_for(tmp_path):
    src = tmp_path / "src"
    _model(src)
    inv = build_inventory(src, layer_types=["full_attention"])
    ev = build_probe_evidence(inv, src, expert_sample=2, sample_rows=64)
    plan, _summary = package_plan_from_decision(decide(ev, target_quality=0.5))
    return src, plan


def test_affine_quantize_is_chunked(tmp_path):
    src, dec = _decision_for(tmp_path)
    chunk_bytes = 256 * 1024  # 256 KB -> AFFINE_ROWS*COLS*4 = 4 MB needs many chunks

    seen_bytes = []
    import mlx.core as mx
    orig = mx.quantize

    def _spy(w, **kw):
        seen_bytes.append(int(w.size) * 4)  # float32 bytes this call sees
        return orig(w, **kw)

    with patch("mlx.core.quantize", side_effect=_spy):
        write_package(dec, src, ARCH, tmp_path / "out", chunk_bytes=chunk_bytes)

    assert seen_bytes, "no affine quantize calls observed"
    for i, nbytes in enumerate(seen_bytes):
        assert nbytes <= chunk_bytes, f"call {i}: {nbytes} > cap {chunk_bytes}"
    # 4 MB f32 affine at 256 KB/chunk -> well over 5 chunks (proves it chunked).
    assert len(seen_bytes) >= 5, f"only {len(seen_bytes)} chunks: not streaming"


def test_experts_quantized_one_at_a_time(tmp_path):
    src, dec = _decision_for(tmp_path)

    expert_call_rows = []
    orig_tq = writemod.quantize_tq

    def _spy_tq(sample, bits, seed):
        expert_call_rows.append(sample.shape[0])
        return orig_tq(sample, bits, seed)

    with patch.object(writemod, "quantize_tq", side_effect=_spy_tq):
        write_package(dec, src, ARCH, tmp_path / "out")

    assert expert_call_rows, "no TQ expert calls observed"
    # Each call must be a single expert's sub-projection (mid = EXP_ROWS//2 for
    # the fused gate/up halves; the fixture's down matches), never the whole
    # stack (which would be N_EXPERTS*rows).
    mid = EXP_ROWS // 2
    for n in expert_call_rows:
        assert n == mid, f"expert TQ call saw {n} rows, expected one expert's {mid}"
    # gate + up + down sub-projections, each over N_EXPERTS -> 3*N_EXPERTS calls.
    assert len(expert_call_rows) == 3 * N_EXPERTS, (
        f"{len(expert_call_rows)} calls != one-per-expert*3 ({3 * N_EXPERTS})")


def test_default_autosizes_the_band_from_free_ram(tmp_path):
    # chunk_bytes=None (default) must route through the RAM-aware auto-sizer, so
    # no affine quantize call exceeds the auto-chosen band, never an unbounded read.
    src, dec = _decision_for(tmp_path)
    from moespresso.package.write import _autosize_chunk_bytes
    auto = _autosize_chunk_bytes()

    seen = []
    import mlx.core as mx
    orig = mx.quantize

    def _spy(w, **kw):
        seen.append(int(w.size) * 4)
        return orig(w, **kw)

    with patch("mlx.core.quantize", side_effect=_spy):
        write_package(dec, src, ARCH, tmp_path / "out")  # no chunk_bytes -> auto

    assert seen, "no affine quantize calls observed"
    for nbytes in seen:
        assert nbytes <= auto, f"auto band {auto} exceeded by a {nbytes}-byte call"


def test_chunked_affine_bytes_identical_to_unchunked(tmp_path):
    """Streaming must not change the output: a tiny chunk and a huge chunk agree.

    Compares the stored packed bytes (no weight reconstruction: the engine never
    reconstructs; a test shouldn't either)."""
    from safetensors.numpy import load_file
    src, dec = _decision_for(tmp_path)
    man_small = write_package(dec, src, ARCH, tmp_path / "small", chunk_bytes=256 * 1024)
    man_big = write_package(dec, src, ARCH, tmp_path / "big", chunk_bytes=10 ** 12)

    a = load_file(str(tmp_path / "small" / man_small["files"][0]["path"]))
    b = load_file(str(tmp_path / "big" / man_big["files"][0]["path"]))
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_large_affine_output_streams_to_shard_files(tmp_path, monkeypatch):
    src, dec = _decision_for(tmp_path)
    monkeypatch.setattr(writemod, "_STREAMED_AFFINE_OUTPUT_THRESHOLD_BYTES", 1)

    def _forbidden(*args, **kwargs):
        raise AssertionError("large affine output used resident concatenate path")

    monkeypatch.setattr(writemod, "_quantize_affine_streamed_chunks", _forbidden)

    out = tmp_path / "out"
    man = write_package(dec, src, ARCH, out, chunk_bytes=256 * 1024)

    assert not verify_package(man, out)
    assert not list(out.glob("*.tmp"))
    assert not list(out.glob(".*.tmp"))

    from safetensors.numpy import load_file

    arrays = {}
    for file in man["files"]:
        arrays.update(load_file(str(out / file["path"])))
    prefix = "model.language_model.layers.0.self_attn.q_proj"
    assert f"{prefix}.weight" in arrays
    assert f"{prefix}.scales" in arrays
    assert f"{prefix}.biases" in arrays


def test_peak_rss_backstop(tmp_path):
    """Coarse backstop: peak RSS growth during a streamed convert stays modest.

    The spy tests are the real invariant; this just catches a gross whole-model
    materialization. Generous ceiling: RSS is noisy (GC, pagecache, allocator).
    """
    psutil = pytest.importorskip("psutil")
    src, dec = _decision_for(tmp_path)

    proc = psutil.Process()
    before = proc.memory_info().rss
    peak = before

    import threading
    import time
    stop = threading.Event()

    def _sample():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, proc.memory_info().rss)
            time.sleep(0.005)

    t = threading.Thread(target=_sample)
    t.start()
    try:
        write_package(dec, src, ARCH, tmp_path / "out", chunk_bytes=256 * 1024)
    finally:
        stop.set()
        t.join()

    grew_mb = (peak - before) / (1024 ** 2)
    # The whole source here is only ~5 MB; a streamed convert should add well
    # under 512 MB. A whole-model materialization regression would blow past this.
    assert grew_mb < 512, f"peak RSS grew {grew_mb:.0f} MB during streamed convert"
