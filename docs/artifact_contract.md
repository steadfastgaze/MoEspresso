# The Artifact Contract

*Single source of truth: `src/moespresso/core/artifact.py`.*

Every durable thing MoEspresso produces (a source inventory, probe evidence, an
optimizer decision, a package plan, a package manifest) is an **Artifact**: a JSON dict
payload plus a small set of required base keys and a content-addressed id.
`core/artifact.py` owns the contract and **nothing else**: canonical serialization, the
content hash, base-key validation (fail-closed), and the read/write helpers that compute
and verify the id.

Phase-specific schemas live next to their phase and may only **add** keys; they never
re-define canonicalization or base validation. Keeping the contract in one tiny module is
exactly what stops every tool from inventing its own serialization, its own id scheme, or
its own notion of "valid." If you are tempted to hash, version, or validate an artifact
anywhere else, you are wrong. Call into `core/artifact.py`.

---

## 1. Base keys every artifact carries

Every artifact, regardless of kind, carries the following base keys. The required set
(checked by `validate_base`) is `artifact_kind`, `schema_version`, `producer`, `subject`,
`status`.

| Key | Type | Meaning |
| --- | --- | --- |
| `artifact_kind` | string | One of the registered kinds (Section 2). Unknown -> fail closed. |
| `schema_version` | `{major:int, minor:int}` | Contract version. See versioning (Section 4). |
| `artifact_id` | string | Content hash, `"<tag>:<sha256hex>"`. Computed, never authored. Excluded from the hash. |
| `producer` | dict | Who/what produced this artifact (tool/phase identity). |
| `created_at` | string (UTC) | Wall-clock write stamp. Stamped at persist time, **excluded from the hash**. |
| `inputs` | list | Upstream artifacts/identities this one was derived from. Defaults to `[]`. |
| `subject` | dict | What the artifact is *about* (e.g. the model/tensor/package under analysis). |
| `status` | string | Lifecycle: `draft`, `valid`, `invalid`, `superseded`, `retired`. |
| `validation` | list | Structured validation entries (see below). Defaults to `[]`. |
| `required_features` | list[string] | Feature strings a reader **must** understand to load. Defaults to `[]`. |

`make_artifact(...)` fills these base keys, runs `validate_base`, and computes
`artifact_id`. It is **deterministic**: it reads no wall-clock, so
identical content always yields the same id. `created_at` is deliberately *not* set here:
it is stamped later by `write_artifact` and excluded from the hash, so persisting an
artifact never changes its id.

### Validation entries

`validation` is a list of structured entries (the `Validation` dataclass). Each entry has:

- `severity`: `"error" | "warning" | "info"`
- `code`: dotted, e.g. `"tensor.shape_mismatch"`, `"artifact.missing_key"`
- `message`: human-readable
- `path`: JSON pointer into the payload (e.g. `/status`), default `""`
- `phase`: producing phase (e.g. `"contract"`), default `""`
- `blocking`: bool; default `false`
- `expected` / `actual`: optional, omitted from the serialized form when `None`

### `required_features` (fail-closed gate)

`required_features` declares capabilities a reader must understand before it may trust the
artifact. Every entry must be in `KNOWN_FEATURES` for this build, or `validate_base`
emits a blocking `artifact.unknown_required_feature` error. Currently
`KNOWN_FEATURES = {"calibration"}` (probe evidence carrying calibration-dataset identity).
The set grows as real features land; a format declares its requirements here.

---

## 2. The artifact kinds

`ARTIFACT_KINDS` registers seven kinds. Five belong to the build pipeline; two are
standalone correctness-ladder evidence. Each has a short id tag used as the
`artifact_id` prefix:

| Kind | Tag | Role |
| --- | --- | --- |
| `source_inventory` | `inv` | What the source model *is*: tensors, shapes, dtypes, file identities. The ground truth other phases read. |
| `probe_evidence` | `probe` | What probing/calibration measured about the model (e.g. activation statistics, calibration identity). |
| `optimizer_decision` | `dec` | What the optimizer chose and why: the decision record for a quantization/optimization pass. |
| `package_plan` | `plan` | The writer-facing allocation IR. Probe/optimizer output and GGUF recipe import both converge here; the writer consumes only the plan, plus its provenance (`producer_kind`, `producer_reference`, `source_decision_id`, `source_probe_id`, force overrides). |
| `package_manifest` | `pkg` | The shippable package's contents: every file's identity, so a tampered/partial package fails verification. |
| `architecture_profile` | `arch` | The model-family correctness contract the ladder checks a package against. |
| `correctness_evidence` | `correct` | What a correctness-ladder rung actually found. |

The two ladder kinds are standalone evidence under the same base rules; they are not
wired into the convert/serve/verify path. The pipeline kinds chain provenance forward:
the manifest records `source_plan_id` (the plan's `artifact_id`) plus the
`source_decision_id` and `source_probe_id` the plan copied through, so a package traces
back to the evidence that produced it.

An artifact whose `artifact_kind` is anything not in `ARTIFACT_KINDS` is rejected
(fail-closed, Section 4). An unrecognized kind also has no id tag of its own; the generic
`art` tag is only a defensive fallback inside `compute_artifact_id` and is never the result
of a successful `make_artifact`, since unknown kinds are rejected first.

---

## 3. Canonicalization and the content hash

The `artifact_id` is a **content hash**: identical content -> identical id, no matter who
wrote it or when. This is what makes artifacts content-addressed and reproducible across
machines and test runs. The id is `"<kind-tag>:<sha256hex>"`, where the hex is SHA-256 over
the canonical JSON of the payload.

`canonical_json(payload)` produces the bytes that get hashed. The rules:

1. **Sorted-key, compact UTF-8 JSON.** `json.dumps(..., sort_keys=True,
   separators=(",", ":"), ensure_ascii=False, allow_nan=False)`. Deterministic byte output;
   non-ASCII is emitted as real UTF-8; `\u` escaping is disabled.
2. **`artifact_id` and `created_at` are excluded from the hash** (`_HASH_EXCLUDED`). The id
   can never depend on itself, and a wall-clock stamp must not perturb content identity:
   two artifacts with identical content share an id regardless of when each was written.
3. **No `NaN`/`Inf`.** `_assert_finite` walks the entire payload (dicts, lists, tuples,
   floats) and raises `ArtifactError` on any non-finite float. `allow_nan=False` is a
   second backstop. Non-finite floats are forbidden in persisted artifacts.
4. **Integers for structural quantities.** Shapes, byte counts, and counts use integer
   values, which serialize exactly and hash stably (no `2.0` vs `2`).
5. **File identity = relative POSIX path + size + SHA-256.** A referenced file is identified
   by the triple `{path (relative POSIX), size_bytes, sha256}`, never by absolute path or
   mtime. This is how `source_inventory` and `package_manifest` pin file content: the
   manifest records `{"path": ..., "size_bytes": ..., "sha256": ...}` per file, and
   `moespresso-verify` re-checks presence + size + sha256 against it.

Because the hash is over the canonical (compact) form, the on-disk representation is free
to differ for human convenience: `write_artifact` stores the file **pretty-printed**
(`indent=2`, sorted keys) for readable diffs, while the *id* remains the hash of the compact
canonical bytes.

---

## 4. Fail-closed versioning

`schema_version` is `{major, minor}`. This build is
`SCHEMA_MAJOR=1`, `SCHEMA_MINOR=0`. Loading is **fail-closed**: when in doubt, reject.

- **Unknown kind -> reject.** `artifact_kind` not in `ARTIFACT_KINDS` raises `ArtifactError`.
- **Unknown major -> reject.** Any `schema_version.major != SCHEMA_MAJOR` raises
  `ArtifactError`. A different major is a different, incompatible contract.
- **Unknown minor -> conditional.** A higher (unknown) minor within the same major is
  *allowed to load only if every `required_features` entry is understood* (i.e. all are in
  `KNOWN_FEATURES`). A minor bump that relies on a feature this build does not understand
  declares that feature in `required_features`, and the unknown-required-feature check
  fails closed. If the bump added only things you can safely ignore, it declares nothing
  there and loads fine.
- **Unknown optional annotation -> ignore.** Extra keys a newer minor adds that are *not*
  gated by `required_features` are simply additive and ignored by older readers. Phases only
  ever ADD keys; they never remove or repurpose base keys within a major.

`validate_base` enforces this split: the two hard cases (unknown kind, unknown major) raise
immediately, while softer problems (missing base key, bad `status`, unknown required
feature) come back as blocking `Validation` entries so the caller can decide. `make_artifact`
treats any returned base issue as fatal and raises.

---

## 5. Read / write semantics

- **`make_artifact(kind, subject, producer, *, inputs, required_features, status,
  validation, **fields)`**: builds the payload, fills base keys, runs `validate_base`,
  computes `artifact_id`. Deterministic; reads no clock. `status` defaults to `"draft"`,
  `required_features` and `inputs` default to empty.
- **`write_artifact(path, payload, created_at=None)`**: recomputes the id and refuses to
  write if a pre-existing `artifact_id` disagrees with the content
  (`"artifact_id does not match payload content"`). Stamps `created_at` (a caller-supplied
  UTC string, no wall-clock is read, so writes stay reproducible in tests) only if absent.
  Writes pretty JSON. Returns the id.
- **`read_artifact(path)`**: loads, **recomputes the id and rejects on mismatch**
  (`artifact_id mismatch: stored ... != computed ...`, catches tampering/corruption), then
  runs `validate_base` so all fail-closed conditions apply on read.

`ArtifactError` is the single exception type for every base-contract violation: unknown
kind, bad version, unknown required feature, a failed hash check, or a content/id mismatch.

---

## 6. Why this lives in one module

`core/artifact.py` is the **single implementation** of canonicalization and base
validation. There is exactly one definition of how an artifact is serialized, how its id is
computed, and what "base-valid" means. Phase code imports these helpers; it does not
reimplement them. That single point of truth is what guarantees that an `inv:...` written by
the inventory phase, a `probe:...` from probing, a `dec:...` from the optimizer, a
`plan:...` from planning, and a `pkg:...` from packaging all obey the same hashing, the
same versioning, and the same fail-closed rules, and that any two artifacts with identical
content are provably the same artifact, everywhere.
