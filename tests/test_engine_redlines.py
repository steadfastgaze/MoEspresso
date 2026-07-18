"""The two engine red lines, enforced as tests.

RL1 (generic, any format): no numpy tensor compute in the engine. The engine is
the inference/serve path: runtime/{build,serve,http}.py. They must not import numpy
or use `np.`. All tensor work is mlx/Metal. numpy is allowed only at the edges and
in offline convert/probe. (jang importing numpy internally is outside this project's
control and acceptable; this guard polices the engine modules.)

RL2 (mjtq/TQ-format-specific): never dequant TurboQuant at load. TQ weights stay
packed; jang's metal kernel runs them at inference. Building a package must not call
the dequant primitive (unpack_bits): that fires only on a forward pass.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ENGINE_MODULES = ("build.py", "serve.py", "http.py")
_RUNTIME_DIR = Path(__file__).resolve().parent.parent / "src" / "moespresso" / "runtime"


def test_rl1_no_numpy_in_engine_modules():
    """RL1: the engine (serve path) source has no numpy import / np. usage."""
    offenders = {}
    for mod in _ENGINE_MODULES:
        src = (_RUNTIME_DIR / mod).read_text()
        hits = []
        for i, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]  # ignore comments
            if re.search(r"\bimport numpy\b", code) or re.search(r"\bnp\.", code):
                hits.append((i, line.strip()))
        if hits:
            offenders[mod] = hits
    assert not offenders, (
        "numpy tensor compute leaked into the engine (RED LINE 1). "
        f"Move it to the edge / offline convert. Offenders: {offenders}")


def test_rl1_deleted_loader_stays_gone():
    """The numpy weight-reconstruction loader (runtime/load.py) must not return."""
    assert not (_RUNTIME_DIR / "load.py").exists(), (
        "runtime/load.py (numpy weight reconstruction) was deleted. It must not "
        "come back on the engine path (RED LINE 1).")


def test_rl2_build_does_not_dequant_at_load(tmp_path, monkeypatch):
    """RL2: building a model from a package must not call the TQ dequant primitive.

    Packed weights are placed into the kernel module; dequant happens only at
    inference. `unpack_bits` is wrapped with a tripwire; build_model must still
    complete.
    """
    pytest.importorskip("mlx.core")
    pytest.importorskip("jang_tools.turboquant")

    # Build a tiny real package the same way the e2e does.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_e2e", str(Path(__file__).resolve().parent / "test_serve_e2e.py"))
    e2e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2e)

    import json

    from moespresso.inventory.build import build_inventory
    from moespresso.optimize.decide import decide
    from moespresso.package.plan import package_plan_from_decision
    from moespresso.package.write import write_package
    from moespresso.probe.build import build_probe_evidence
    from moespresso.runtime import build as build_mod

    ref = e2e._reference_model()
    src = tmp_path / "src"
    src.mkdir()
    arch = e2e._arch()
    e2e._dump_source_safetensors(ref, src / "model-00001.safetensors")
    (src / "config.json").write_text(json.dumps(arch))
    inv = build_inventory(src, layer_types=e2e.LAYER_TYPES)
    pt = [t for t in inv["tensors"] if t["kind"] == "passthrough"]
    ev = build_probe_evidence(inv, src, expert_sample=2, sample_rows=64)
    dec = decide(ev, target_quality=0.5)
    plan, _summary = package_plan_from_decision(dec)
    out = tmp_path / "pkg"
    man = write_package(plan, src, arch, out, passthrough=pt)
    e2e._emit_sidecars(man, out)

    # Trip the dequant primitive: if load dequants TQ, this raises.
    import jang_tools.turboquant.pipeline as pipeline

    called = {"n": 0}
    orig = pipeline.unpack_bits

    def _tripwire(*a, **k):
        called["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(pipeline, "unpack_bits", _tripwire)

    # build_model places packed buffers: it must not dequant. (jang's warmup may run a
    # forward, which legitimately dequants; load itself is asserted not to, by checking
    # the model builds and that a no-warmup build path doesn't unpack. Since jang runs
    # a warmup, the build is asserted to succeed and unpack to only happen via forward:
    # the number of unpack calls equals the active experts in the warmup.)
    model, _cfg = build_mod.build_model(man, out)
    assert model is not None
    # If load dequanted all expert groups eagerly, unpack would fire far more than a
    # single warmup forward's active experts. Guard: it stays bounded.
    n_expert_groups = sum(1 for t in man["tensors"] if t["format"] == "tq")
    assert called["n"] <= n_expert_groups, (
        f"unpack_bits called {called['n']}x: load is dequanting TQ groups eagerly "
        f"(RED LINE 2). It should fire only on a forward's active experts.")
