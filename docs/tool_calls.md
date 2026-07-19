# Served tool calls

The HTTP layer implements the OpenAI tools contract end to end. The render
layer turns a request's `tools` schemas into dialect instructions inside
the prompt, and the serve layer parses the model's marked-up emission back
into structured `tool_calls` entries with `finish_reason: "tool_calls"`.
The dialect text exists only between the server and the model; a client
works with ordinary OpenAI JSON on both legs.

## Dialects

A dialect is the text grammar the model emits tool invocations in. The
grammars, strict parsers, serializers, and the repair layer live in
`moespresso.toolcalls`; the serve layer and agentlib share that code.

- `qwenxml`: the Qwen-family native format the vendored template teaches.
  One call per `<tool_call>` block, one `<function=name>` element,
  `<parameter=name>` values as raw text with one newline of padding on each
  side. Parameter values are typed against the request's tool schemas: a
  parameter whose declared type is not string must decode as JSON of that
  type.
- `dsml`: the DeepSeek format. One `<DSML tool_calls>` block (where `DSML`
  stands for the marker token) holding one or more `invoke` elements, each
  parameter carrying a `string="true|false"` attribute that decides raw
  text versus JSON decoding.

DeepSeek-V4 always serves DSML; the renderer owns that as part of the model
contract. Template families default to their template's native dialect and
can serve DSML instead through the dialect swap below.

## Parse path

`runtime.tool_stream.ToolCallStreamer` sits on the answer channel of the
streaming classifier (`runtime.chat_stream.ReasoningSplitter`), so text
inside a thinking block never triggers it. It scans decoded text for
dialect open markers at line starts (the block wrapper, and the bare
function or invoke element for the wrapper-dropped malformation), holds
back only the longest tail that could begin a marker, and buffers each
block until its close marker. A completed block goes to the strict parser;
the parsed calls stream out as `tool_calls` deltas (one complete call per
delta) and land on the response message with `finish_reason:
"tool_calls"`. Text around blocks flows as ordinary content, and a marker
quoted mid-sentence stays prose.

Parsing adds no model work: logits, sampling, and decode arithmetic are
untouched, and a request without `tools` skips the whole path and serves
byte-identically to a server without it. A streaming request classifies
each decoded chunk between decode steps, exactly like the reasoning
splitter it chains onto, at the cost of a string scan over a few bytes per
step. A non-streaming request parses the completed text once after
generation.

Failure handling is strict-parse-first:

- A block the strict parser rejects goes to the repair layer
  (`toolcalls.repair`): a bounded sequence of text transformations for the
  malformed shapes small models actually emit (dropped closers, missing
  wrappers, wrong quoting, mistyped scalar values). The first transformed
  candidate the strict parser accepts wins. Because attempts buffer
  through their own markers, a repaired call never leaks raw markup into
  streamed content.
- Text that still fails flushes back into `content`, so bytes are never
  silently dropped.
- A block left open when generation hit the token limit is never repaired:
  closing a half-emitted value would fabricate a plausible but wrong
  argument. The turn keeps `finish_reason: "length"` and the raw text.
- Parsed argument values are typed against the request's tool schemas on
  every path: the Qwen XML parser types at parse time, and a DSML value
  whose `string` flag disagrees with the declared schema is coerced after
  parse, so a client never receives `"5"` where the schema declares an
  integer.
- Repair activity is reported per request under
  `usage.moespresso.tool_call_repair` (`fires`/`salvaged`/`failed`
  counters). A rising `failed` count is the alarm that the served dialect
  is not viable for the model.

## Turn 2: history rendering

The follow-up request carries the assistant `tool_calls` message and
`role: "tool"` results. Validation accepts an assistant message without a
`content` key when it carries `tool_calls`. Before rendering, argument
JSON strings are decoded to objects so the template renders parameter
elements instead of dumping raw JSON, and a null content becomes empty
text.

The strict parser trims exactly the newline padding the template
re-inserts, so for the native Qwen dialect the re-rendered history turn is
byte-identical to the original emission. Turn N's rendered prompt plus the
served completion is a byte prefix of turn N+1's rendered prompt, and the
KV prefix extends across tool turns instead of re-prefilling them
(`tests/test_render_contract.py` pins this). Under the DSML swap the
re-serialized block can differ from the emission by leading whitespace;
prefix reuse then falls back to re-prefilling from the assistant turn,
which is a cost, never a correctness risk: both the in-memory prefix cache
and the disk checkpoint tier match exact token prefixes and fail closed to
cold serving on any mismatch.

The rendered-prompt identity accounts for the dialect: a request rendered
under a non-native dialect carries `tool_dialect` inside its rendering
identity, and every identity minted before the field existed stays
byte-identical.

## Dialect selection

Per server process, resolved at startup and printed as a `[serve]` line:

1. `MOESPRESSO_TOOL_CALLS=0` disables served tool-call handling entirely:
   tools render exactly as sent, tool markup returns verbatim as content,
   and the pre-parsing request contract applies (every message needs a
   content key, `tool_choice` is not interpreted).
   `MOESPRESSO_TOOL_REPAIR=0` keeps parsing strict-only. Both default on.
2. `--tool-dialect native|dsml` selects explicitly.
3. Otherwise the package's `agentic_profile.json` dialect of record
   applies (`dsml` for the Ornith family, from the recorded dialect
   study). A missing or unreadable profile, or a schema version above the
   supported one, falls back to native.

The `dsml` selection for a template family renders the DSML instruction
block into the system region instead of passing `tools` to the template,
serializes past assistant `tool_calls` into DSML text, and parses DSML
from the emission. The native XML parser stays active second: the family's
trained format can bleed through, and catching it costs nothing. Tool
results still travel as `role: "tool"` messages either way.

`tool_choice` accepts `auto` (default behavior) and `none` (tools are
withheld from the render; note that flipping between `none` and `auto`
inside one session changes the system region and costs a prefix re-render
on that turn). A forced function selection is refused with a 400 rather
than accepted and not enforced.

## Per-request verbatim mode

`metadata.moespresso_tool_calls: "verbatim"` opts a single request out of
every served tool-call behavior: tools render exactly as sent, the
completion text returns unparsed, and the pre-parsing request contract
applies. The agent library's client exposes it as
`complete(..., verbatim_tool_calls=True)`. The road-test and the dialect
study send it on every request, because those instruments measure the raw
text dialect client-side; without the flag the serve layer would parse the
emission first and the instrument would read empty content.
