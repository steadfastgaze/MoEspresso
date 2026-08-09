"""The WikiText perplexity gate: corpus handling, protocol, and the declared bar.

No package and no oracle material. The corpus is synthetic bytes written into
tmp_path, and the model is a stub whose logits are chosen so the expected loss
is computable by hand.
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from moespresso.core.artifact import compute_artifact_id, validate_base
from moespresso.correctness.deepseek_v4.wikitext_ppl import (
    RECORDED_CORPUS_NAME,
    RECORDED_CORPUS_SHA256,
    WIKITEXT_CORPUS_ENV,
    WIKITEXT_PPL_EVIDENCE_SCHEMA,
    aggregate_wikitext_score,
    load_wikitext_corpus,
    main,
    make_wikitext_ppl_evidence,
    resolve_corpus_path,
    score_wikitext_windows,
    validate_wikitext_ppl_evidence,
    wikitext_windows,
)


def test_the_recorded_corpus_is_the_canonical_wikitext_test_split():
    """The gate's held-out bar must be pinned to held-out text.

    An earlier pin named a file a third-party converter ships as calibration
    data, whose own notice describes it as a contiguous subset of the train
    split; a held-out perplexity bar scored on training text measures the
    wrong thing. The digest below is the public dataset file's, so this pin
    is checkable by anyone who downloads the split rather than by whoever
    happens to hold a local copy.
    """
    from moespresso.correctness.deepseek_v4.wikitext_ppl import (
        CANONICAL_WIKITEXT_TEST_SHA256,
        DEFAULT_WINDOW_COUNT,
        DEFAULT_WINDOW_SIZE,
        RECORDED_CORPUS_BYTES,
        RECORDED_CORPUS_TOKEN_COUNT,
    )

    assert RECORDED_CORPUS_NAME == "wiki.test.raw"
    assert RECORDED_CORPUS_SHA256 == CANONICAL_WIKITEXT_TEST_SHA256
    assert CANONICAL_WIKITEXT_TEST_SHA256 == (
        "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08")
    assert RECORDED_CORPUS_BYTES == 1_290_590
    assert RECORDED_CORPUS_TOKEN_COUNT == 287_730
    # The protocol of record scores 32 windows, and the corpus has to carry
    # them: a default the corpus cannot satisfy fails only at runtime.
    assert DEFAULT_WINDOW_COUNT == 32
    assert DEFAULT_WINDOW_SIZE * DEFAULT_WINDOW_COUNT <= RECORDED_CORPUS_TOKEN_COUNT


def _corpus(tmp_path, text: str = "wiki text corpus\n"):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    return path, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _window_rows(nlls, target_tokens=4):
    return [
        {
            "index": i,
            "input_tokens": target_tokens + 1,
            "target_tokens": target_tokens,
            "nll": nll,
            "avg_nll": nll / target_tokens,
            "perplexity": math.exp(nll / target_tokens),
        }
        for i, nll in enumerate(nlls)
    ]


def _external(nlls=(4.0, 4.0), limit=4.0):
    rows = _window_rows(nlls)
    return {
        "family": "deepseek_v4_flash",
        "schema": WIKITEXT_PPL_EVIDENCE_SCHEMA,
        "run": {
            "decode": "teacher_forced_corpus_ppl",
            "thinking": False,
            "cache": "none",
            "logits_dtype_for_loss": "float32",
            "loss": "logsumexp(logits) - target_logit",
        },
        "protocol": {
            "window_size": 5,
            "window_count": len(rows),
            "window_layout": "first complete, contiguous, non-overlapping windows",
            "reduction": "MLX sum per window, Python float sum across windows",
            "corpus_tokens": 20,
            "scored_input_tokens": 5 * len(rows),
        },
        "corpus": {
            "path": "/tmp/corpus.txt",
            "sha256": "0" * 64,
            "bytes": 16,
            "matches_recorded_corpus": False,
        },
        "candidate": {
            "kind": "moespresso_mlx_package",
            "package_dir": "/tmp/package",
            "package_manifest_id": "pkg:abc",
            "family": "deepseek_v4_flash",
        },
        "inputs": [],
        "windows": rows,
        "score": aggregate_wikitext_score(rows, limit=limit),
    }


def _codes(findings):
    return {v.code for v in findings}


# --- corpus handling ---------------------------------------------------------


def test_corpus_absence_names_the_expected_file_and_digest(tmp_path, monkeypatch):
    monkeypatch.delenv(WIKITEXT_CORPUS_ENV, raising=False)

    with pytest.raises(SystemExit) as unset:
        resolve_corpus_path(None)
    with pytest.raises(SystemExit) as missing:
        resolve_corpus_path(tmp_path / "absent.txt")

    for message in (str(unset.value), str(missing.value)):
        assert RECORDED_CORPUS_NAME in message
        assert RECORDED_CORPUS_SHA256 in message
        assert WIKITEXT_CORPUS_ENV in message


def test_corpus_path_comes_from_the_environment_when_no_flag_is_given(tmp_path, monkeypatch):
    path, _ = _corpus(tmp_path)
    monkeypatch.setenv(WIKITEXT_CORPUS_ENV, str(path))

    assert resolve_corpus_path(None) == path
    assert resolve_corpus_path(path) == path


def test_corpus_digest_mismatch_is_refused_and_names_both_digests(tmp_path):
    path, digest = _corpus(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        load_wikitext_corpus(path, expected_sha256="1" * 64)

    message = str(excinfo.value)
    assert "1" * 64 in message
    assert digest in message


def test_corpus_loads_with_its_declared_digest(tmp_path):
    path, digest = _corpus(tmp_path)

    text, provenance = load_wikitext_corpus(path, expected_sha256=digest)

    assert text == "wiki text corpus\n"
    assert provenance["sha256"] == digest
    assert provenance["bytes"] == len(text.encode("utf-8"))
    # A synthetic corpus is not the corpus of record, and the evidence says so.
    assert provenance["matches_recorded_corpus"] is False


def test_corpus_default_expectation_is_the_recorded_digest(tmp_path):
    path, _ = _corpus(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        load_wikitext_corpus(path)

    assert RECORDED_CORPUS_SHA256 in str(excinfo.value)


# --- protocol ----------------------------------------------------------------


def test_windows_are_contiguous_non_overlapping_and_drop_the_remainder():
    windows = wikitext_windows(list(range(11)), window_size=4, window_count=2)

    assert windows == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_windows_fail_closed_when_the_corpus_is_too_short():
    with pytest.raises(SystemExit) as excinfo:
        wikitext_windows(list(range(7)), window_size=4, window_count=2)

    assert "8" in str(excinfo.value)


def test_scoring_a_window_matches_the_hand_computed_loss():
    """A uniform logit row gives loss log(V) per target, exactly."""
    import mlx.core as mx

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        vocab = 8

        class _UniformModel:
            def __call__(self, tokens):
                steps = tokens.shape[1]
                return mx.zeros((1, steps, vocab))

        rows = score_wikitext_windows(_UniformModel(), [[0, 1, 2, 3]], mx=mx)

        assert rows[0]["target_tokens"] == 3
        assert rows[0]["nll"] == pytest.approx(3 * math.log(vocab), rel=1e-6)
        assert rows[0]["perplexity"] == pytest.approx(vocab, rel=1e-6)
    finally:
        mx.set_default_device(previous_device)


def test_aggregate_is_token_weighted_across_windows():
    score = aggregate_wikitext_score(_window_rows([4.0, 8.0]), limit=100.0)

    assert score["target_tokens"] == 8
    assert score["avg_nll"] == pytest.approx(1.5)
    assert score["perplexity"] == pytest.approx(math.exp(1.5))
    assert score["passed"] is True


# --- the declared bar --------------------------------------------------------


def test_evidence_is_valid_under_its_declared_limit():
    external = _external(limit=100.0)

    findings = validate_wikitext_ppl_evidence(external)
    artifact = make_wikitext_ppl_evidence({"package_id": "pkg:test"}, external)

    assert findings == []
    assert validate_base(artifact) == []
    assert artifact["rung"] == "PPL"
    assert artifact["status"] == "valid"
    assert artifact["summary"]["limit"] == 100.0
    assert artifact["artifact_id"] == compute_artifact_id(artifact)


def test_evidence_fails_above_the_declared_limit():
    external = _external(limit=1.5)

    artifact = make_wikitext_ppl_evidence({"package_id": "pkg:test"}, external)
    codes = {v["code"] for v in artifact["validation"]}

    assert "deepseek_v4.ppl.above_limit" in codes
    assert artifact["status"] == "invalid"


def test_the_limit_is_not_a_constant_of_the_instrument():
    """The same windows pass or fail purely on the limit the caller declares."""
    rows = _window_rows([4.0, 4.0])
    perplexity = aggregate_wikitext_score(rows, limit=1.0)["perplexity"]

    assert aggregate_wikitext_score(rows, limit=perplexity + 1)["passed"] is True
    assert aggregate_wikitext_score(rows, limit=perplexity - 0.5)["passed"] is False


def test_a_non_finite_window_is_a_blocking_finding():
    """The failure this arm exists for: overflow on ordinary prose that the
    numbered gates and a code corpus both score as finite."""
    external = _external()
    external["windows"][1]["nll"] = float("nan")
    external["score"]["perplexity"] = float("nan")

    findings = validate_wikitext_ppl_evidence(external)

    assert "deepseek_v4.ppl.non_finite_window" in _codes(findings)
    assert "deepseek_v4.ppl.non_finite_perplexity" in _codes(findings)


def test_evidence_requires_the_uncached_teacher_forced_protocol():
    external = _external()
    external["run"]["decode"] = "greedy"
    external["run"]["cache"] = "reuse"

    findings = validate_wikitext_ppl_evidence(external)

    assert "deepseek_v4.ppl.decode_not_teacher_forced" in _codes(findings)
    assert "deepseek_v4.ppl.cache_not_disabled" in _codes(findings)


def test_evidence_requires_corpus_provenance_and_a_package_candidate():
    external = _external()
    external["corpus"] = {"path": "/tmp/corpus.txt"}
    external["candidate"] = {"kind": "reference_implementation"}

    findings = validate_wikitext_ppl_evidence(external)

    assert "deepseek_v4.ppl.missing_corpus_provenance" in _codes(findings)
    assert "deepseek_v4.ppl.candidate_kind" in _codes(findings)


def test_evidence_rejects_a_scored_window_count_the_protocol_does_not_declare():
    external = _external()
    external["protocol"]["window_count"] = 8

    findings = validate_wikitext_ppl_evidence(external)

    assert "deepseek_v4.ppl.window_count_mismatch" in _codes(findings)


def test_evidence_without_a_limit_is_not_a_gate():
    external = _external()
    external["score"]["limit"] = None

    findings = validate_wikitext_ppl_evidence(external)

    assert "deepseek_v4.ppl.missing_limit" in _codes(findings)


# --- CLI ---------------------------------------------------------------------


def test_cli_requires_a_limit(tmp_path):
    with pytest.raises(SystemExit):
        main(["--package", str(tmp_path)])


def test_cli_refuses_to_discover_a_package(monkeypatch):
    monkeypatch.delenv("MOESPRESSO_DS4_QUALITY_PACKAGE", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        main(["--limit", "6.75"])

    assert "never" in str(excinfo.value)


def test_cli_names_the_corpus_before_loading_a_package(tmp_path, monkeypatch):
    """Corpus resolution happens before any model work, so a missing corpus
    costs a message rather than a package load."""
    monkeypatch.delenv(WIKITEXT_CORPUS_ENV, raising=False)
    package = tmp_path / "pkg"
    package.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        main(["--package", str(package), "--limit", "6.75"])

    assert RECORDED_CORPUS_SHA256 in str(excinfo.value)


def test_evidence_round_trips_as_json(tmp_path):
    artifact = make_wikitext_ppl_evidence({"package_id": "pkg:test"}, _external(limit=100.0))
    path = tmp_path / "ppl.json"
    path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")

    assert json.loads(path.read_text(encoding="utf-8")) == artifact
