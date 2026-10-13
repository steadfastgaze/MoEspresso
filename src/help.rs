//! Command-line help, kept separate from argument parsing.

pub(crate) const OVERVIEW: &str = "\
MoEspresso: native Qwen inference on Apple Silicon

Usage: moespresso <COMMAND> [OPTIONS]

Commands:
  serve       Start a local Chat Completions server
  generate    Generate text from one prompt
  speed       Measure an existing server's prefill and decode speed

Options:
  -h, --help  Print help

Run 'moespresso <COMMAND> --help' for command options.
";

pub(crate) const SERVE: &str = "\
Start a serial Chat Completions server on 127.0.0.1.

Usage: moespresso serve PACKAGE [OPTIONS]

Options:
  --port N                  Listen port [default: 8080]
  --thinking on|off         Default thinking mode [default: on]
  --max-context N           Context token limit [default: min(package, 131072)]
  --max-memory-gb GIB       Cap the automatic memory budget in GiB
  --prompt-cache-gb GIB     RAM prompt-cache budget; 0 disables [default: 1]
  --no-disk-cache           Disable disk KV reads and writes; keep RAM caching
  --disk-cache-mode MODE    Writes: prefix|full [default: prefix, first 16K tokens]
  --disk-cache-dir PATH     Store disk KV in this private directory
  --disk-cache-gb GIB       Disk KV retention budget in GiB [default: 8]
  --no-adaptive-memory      Keep automatically admitted expert capacity fixed
  --cache-routing MODE      Decode: off|prefer-resident [default: prefer-resident]
  --expert-capacity N       Fixed slots per layer, 10..512; disables both caches
  -h, --help                Print help

--expert-capacity conflicts with --max-memory-gb and enabled cache settings.
--no-disk-cache conflicts with disk-cache options; existing files remain.
Full mode also saves completed prompts and answers for deeper disk resume.

Disable both cache tiers:
  moespresso serve PACKAGE --no-disk-cache --prompt-cache-gb 0
";

pub(crate) const GENERATE: &str = "\
Generate text from one prompt.

Usage: moespresso generate PACKAGE --prompt TEXT [OPTIONS]

Options:
  --prompt TEXT             Input prompt (required)
  --no-thinking             Disable thinking [default: on]
  --max-tokens N            Output token limit [default: 128]
  --max-context N           Context token limit [default: 4096]
  --max-memory-gb GIB       Cap the automatic memory budget in GiB
  --expert-capacity N       Fixed slots per layer, 10..512; bypass automatic sizing
  --cache-routing MODE      Decode: off|prefer-resident [default: prefer-resident]
  --temperature F           Sampling temperature; 0 selects greedy decoding
  --top-p F                 Nucleus sampling threshold
  --top-k N                 Keep the N most likely tokens; 0 disables
  --min-p F                 Minimum probability relative to the most likely token
  --presence-penalty F      Penalty for tokens already present in recent history
  --presence-context-size N Penalty window in tokens; 0 means all [default: 20]
  --seed U64                Random seed [default: random]
  -h, --help                Print help

Sampling defaults come from the package unless overridden.
--expert-capacity and --max-memory-gb are mutually exclusive.
";

pub(crate) const SPEED: &str = "\
Measure prefill, decode and first-token timing against an existing server.
Topic results print as each request finishes.
Then runs ~4K prefill and asks whether to run ~32K (y/N).

Usage: moespresso speed [OPTIONS]

Options:
  --url HTTP_ORIGIN         Server origin [default: http://127.0.0.1:8080]
  --max-tokens N            Topic output cap, 2..2048 [default: 256]
  --timeout SECONDS         Request timeout [default: 300; 32K: 1800]
  --stream                 Use SSE streaming instead of buffered JSON
  --compare-id U64          Repeatable prompts for native/fork comparisons
  -h, --help                Print help

Uses greedy sampling and does not bypass server caches.
Comparison pins thinking on and excludes cached/early-stop results from matched rates.
Non-interactive runs skip 32K. Prefill cases generate at most 8 tokens.
For uncached prefill, start the server with --no-disk-cache --prompt-cache-gb 0.
Native/fork setup and timing definitions: docs/speed_comparison.md
";
