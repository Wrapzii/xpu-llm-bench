# xpu-llm-bench

Consistent, comparable benchmark runs for LLM serving on an Intel XPU bench host. It was written against 2× Intel Arc Pro B60 with a vLLM XPU fork (MTP / QSA / K8V4 / DFlash2), but works with any OpenAI-compatible vLLM server: start from [`specs/generic-vllm.yaml`](specs/generic-vllm.yaml).

Same methodology, same metrics, same report columns — so a run from Tuesday can be put next to a run from Friday without guessing what was measured.

## Prerequisites

The harness does **not** boot, configure or stop vLLM. You bring a running
test container; the harness measures it and checks it is what the spec says.

1. **Clone and install** (Python 3.11+):

   ```bash
   git clone https://github.com/Wrapzii/xpu-llm-bench && cd xpu-llm-bench
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```

2. **Write the site config** (see [Site config](#site-config) below). A live
   `bench run` refuses to start without one unless you pass `--no-site-config`.

3. **Boot a test container yourself**, on a port that is not a production
   port. An **example** for an Intel XPU host (every image, path, name and
   value is a placeholder; the `K8V4_QSA` env var and the speculative config
   only apply to a vLLM fork that has them, so drop them for stock vLLM):

   ```bash
   #!/usr/bin/env bash
   # Example only: adjust image, paths, model, TP size and port to your host.
   set -euo pipefail
   IMAGE="<vllm-xpu-image>:<tag>"            # your vLLM XPU image
   MODEL_DIR="/path/to/models/<model>"        # target model on the host
   DRAFT_DIR="/path/to/models/<drafter>"      # MTP / draft model, if any
   PORT=8102                                  # NOT a production port

   docker run -d --name vllm-bench-qsa \
     --device /dev/dri --group-add render \
     --ipc=host --shm-size 16g \
     -v "$MODEL_DIR":/models/target:ro \
     -v "$DRAFT_DIR":/models/draft:ro \
     -p "$PORT":"$PORT" \
     -e K8V4_QSA=1 \
     "$IMAGE" \
     vllm serve /models/target \
       --served-model-name my-model \
       --tensor-parallel-size 2 \
       --kv-cache-dtype <kv-format> \
       --speculative-config '{"method": "mtp", "model": "/models/draft", "num_speculative_tokens": 6}' \
       --max-num-seqs 8 \
       --host 0.0.0.0 --port "$PORT"
   ```

   Then put the same values in the spec (`server.env`, `server.launch_args`,
   `server.image`) so preflight can check the container matches.

   Never point the harness at a production container or port; it refuses
   protected names and ports, but it is your container to launch.

4. **Point a spec at it.** Copy a spec from `specs/` and set:
   - `endpoint.base_url` to the container's port (`metrics_url` / `health_url`
     default to the same host and port), and `endpoint.model` to the served
     model name;
   - `server.container_name` to the container's name (it must exist and be
     running: preflight runs `docker inspect` on it);
   - `server.image`, `server.env`, `server.launch_args` to what you launched,
     or `null` / empty to skip a check.

   **`server.env`, `server.launch_args` and `server.image` are expectations,
   not settings.** The harness never applies them. At preflight it compares
   them with `docker inspect` (env, command line, image) and, on any
   mismatch, flags every row `NOT_AS_LABELED` and records the evidence in
   `manifest.json` → `launch_check`.

   Every `TODO` or `<...>` value must be replaced (or set to `null`) before a
   live run; preflight lists the remaining ones by field path and stops.

5. **Check, dry-run, then run:**

   ```bash
   bench plan my-spec.yaml               # order, time, GPU/RAM estimates (information only; never blocks)
   bench plan my-spec.yaml --max-model-len 262144 --kv-capacity-tokens 700000   # full auto ladder + per-user-count ceilings
   bench run my-spec.yaml --dry-run      # plan + the exact safety commands; writes only <out>/<spec>/dry-run/
   bench run my-spec.yaml --output-dir results
   bench run my-spec.yaml --repeats 3    # measure each rung's decode window 3x on the same context (median, min to max)
   bench report results/<spec-name>      # markdown on stdout
   bench report results/<spec-name> --html   # self-contained report.html with charts
   ```

### A/B specs (variants)

Each variant must run against **its own booted container**. Variants share
one endpoint by default, so `bench run` refuses to run them back to back
unless you choose one of:

- `--variant NAME`: run one variant; boot its container first, then run the
  next variant with its own container;
- `--pause-between-variants`: the runner stops before each variant and tells
  you which container (name, env, args) to boot, then waits for Enter;
- a per-variant `endpoint:` (`base_url`; metrics and health follow it) so
  both containers can be up at once on different ports.

Without one of these, `bench run` refuses with exit 3.

Each variant's container is verified against that variant's `env` /
`launch_args` / `image`; a container that does not match is `NOT_AS_LABELED`.

## Site config

The production details of your bench host live in an untracked YAML file,
searched in this order (the current directory is **not** searched):

1. `$XPU_BENCH_SITE` (a path; an empty value disables site config),
2. `<repo root>/bench-site.local.yaml` (gitignored),
3. `~/.config/xpu-llm-bench/site.yaml`.

`bench plan` and `bench run` print which file was loaded. Start from
[`bench-site.example.yaml`](bench-site.example.yaml) and fill in:

| Field | Required | What to put there |
|---|---|---|
| `host` | yes | Display name of the bench host (written to results as `hardware.host`). |
| `ssh_host` | if `access.mode: ssh` | An `~/.ssh/config` alias for the bench host. Never an IP in a spec. |
| `protected_containers` | yes | Exact names of every production container the runner must never touch. |
| `protected_patterns` | no | Case-insensitive regexes for other protected services (gateways, git hosting, CI). |
| `forbidden_ports` | yes | Host ports of every production endpoint; endpoints and containers on them are refused. |
| `hold_command` | no | Path of a `bench-hold` style lock CLI on the bench host. Unset: runs take no hold (with a warning). |
| `quiet_hold_file` | no | Operator quiet-hold file (read only); an active quiet hold blocks a run. |
| `timezone` | no | IANA zone for the `*_local` timestamps (default: `$XPU_BENCH_TZ`, else system local). |
| `hardware` | no | Host facts specs do not set: `gpu_model`, `gpu_count`, `gpu_mem_gib`, `tensor_parallel`, `mem_available_floor_gib` (default 2). |
| `planner` | no | Planner estimates, e.g. `host_bytes_per_token` (host RAM growth per live token). |

Any value still containing `<...>`, `TODO`, `example` or `changeme` is
rejected, as is an empty `host`, `protected_containers` or `forbidden_ports`.
Site values only ever add protection; the built-in neutral defaults
(`vllm-production`, port 8001, CI-runner names) always apply.

## Quick start

```bash
# See the progressive rung order, a time estimate, and GPU/RAM feasibility
# (no GPUs, no containers):
bench plan specs/qsa-vs-dense.yaml --profile quick

# Dry run: the plan plus the exact safety commands it would run, printed to
# stdout. It writes only under <out>/<spec>/dry-run/ (manifest status
# dry_run, ignored by --resume, no sessions.json) and still refuses
# protected production ports / container names:
bench run specs/qsa-vs-dense.yaml --dry-run --output-dir /tmp/xpu-llm-bench-demo

# Against your running test container (see Prerequisites):
#   export OPENAI_API_KEY=...    # optional; read from the env, never from the spec
bench run my-qsa-full-fix.yaml --profile quick --output-dir results

# After an abort, continue from the last completed rung:
bench run my-qsa-full-fix.yaml --profile quick --output-dir results --resume

# A/B: one variant at a time, each against its own container:
bench run my-qsa-vs-dense.yaml --variant qsa --output-dir results
bench run my-qsa-vs-dense.yaml --variant dense --output-dir results

# Report (markdown on stdout; --output FILE / --csv FILE to write files) and A/B deltas:
bench report results/my-qsa-full-fix
bench compare results/qsa-vs-dense__dense/results.jsonl results/qsa-vs-dense__qsa/results.jsonl

# HTML report with charts (also written automatically as report.html at the end
# of every real run; --no-html skips it). --html-out picks the path:
bench report results/my-qsa-full-fix --html --html-out /tmp/report.html
# A/B overlay: first run solid, the others dashed:
bench compare results/qsa-vs-dense__dense results/qsa-vs-dense__qsa --html /tmp/ab.html

# Older rung JSONL (scaffold rows or speed-fix ladder-rungs.jsonl):
# --attention-path records what the old run used as DECLARED, not proven:
bench import-legacy old.jsonl results/imported.jsonl --prompt-type code_edits --attention-path qsa
```

Tests and lint (no GPUs; a mock OpenAI server is used):

```bash
ruff check src tests
pytest
```

### Fast concurrency subset

Use the same resolved spec and profile for planning and running:

```bash
bench plan specs/generic-vllm.yaml --concurrency 1,4
bench run specs/generic-vllm.yaml --concurrency 1,4
```

Only the requested concurrency steps are measured; they must be unique, positive,
strictly increasing members of the resolved spec's `ladder.concurrency_steps`.
The progressive ladder retains the full schedule's session IDs for comparable
prompts while skipping unselected decode workloads. The manifest config records
the selected steps and the original session-ID ladder. Contexts, output tokens,
warmup, and metric definitions are not changed. Use the same subset when resuming.

## What a run records

Every run directory describes itself:
- `launch.json` and the header of `run.log`: how the server was started, captured
  read-only at preflight with UTC and local timestamps. From `docker inspect` of the
  test container (entrypoint, cmd, args, env, image + digest, devices, ports, IPC /
  shm), or the server's command line from `ps` when it is not in a container, plus
  vLLM's own boot-log "non-default args" / engine-config lines. Secret-looking values
  (`*_TOKEN`, `*_KEY`, `*SECRET*`, `*PASSWORD*`, `--api-key X`, token-shaped strings)
  are redacted. The same record is in `manifest.json` → `launch`. Nothing is started,
  stopped or restarted to get it.
- `run.log`: that header, then one line per rung and the final status.
- `report.html`: one self-contained file (inline SVG drawn by the harness: no
  scripts, no CDN, no network assets; opens offline; follows the system light/dark
  setting). For decode tok/s, prefill tok/s, TTFT, verify steps/s, accepted drafts
  per step and true hit rate it draws one chart per prompt type and load mode:
  X = the context rungs that ran (log scale when they span a wide range, e.g. 8K to
  262K), Y = the metric, one line per concurrency (users), so you see the family of
  curves as both grow. A line ends at its user count's context ceiling, with a marker
  whose tooltip gives the reason. Hover a point for the exact value. Group headings are
  plain words ("Code edits, closed loop") with the group's quality pass counts above the
  charts ("Code edit: 5 of 5 pass · Needle: 9 of 10 pass · Format: 6 of 6 pass · Greedy
  match: 4 of 5 · Logprob: ok 3 of 3"; checks that never ran are left out), an overall
  count at the top, and the same per-group counts above the "All numbers" table. An
  "Averages across prompt types" table and chart group average decode tok/s, steps/s,
  accepted drafts/step, true hit rate, TTFT and prefill over the prompt types for every
  users × context cell where each prompt type has a clean row (flagged and invalid rows
  are excluded). Flagged rows (`NOT_AS_LABELED`, `QSA_FALLBACK`, `GRAPH_MISS`,
  `HIT_RATE_UNKNOWN`, ...) and invalid rows are drawn hollow and list their flags.
  The spec-decode charts are replaced by a note when the run has no speculative
  decoding. A header shows the run fingerprint (image, attention path from logs, KV
  format, graphs and capture sizes, spec method and k, hardware, timestamps,
  max_model_len and the context rungs, KV capacity, the pre-run load, how far each user
  count got, the captured start command) and caveats; a full numbers table follows the
  charts. `python scripts/make_example_report.py out.html` renders an example from
  synthetic data. `bench report` on a refused or empty run says so, with the reason
  from the manifest.

**Headline speed is verify steps/s per user, shown next to decode tok/s.**
The true hit rate is `(delivered − verify_steps) / (k × verify_steps)`, with k
taken from the spec (MTP-6 = 6, DFlash2 = 7, MTP-3 = 3), never hard-coded.

Each rung records:
- **Speed:** prefill tok/s, TTFT, TPOT, ITL p50/p95/p99, end-to-end latency, queue time
- **Server state:** running/waiting (mean and max), prefix-cache hit rate, KV usage %, preemptions
- **Host memory:** MemAvailable minimum, RSS of the API server, engine and each TP worker, and any other resident vLLM stack
- **What actually ran:** the attention path, graphs, k and KV dtype, read from server logs rather than flags, plus a build fingerprint (image id + hash of every mount; model weights are listed by name + size, never read)
- **Seed:** the fixed prompt seed
- **Quality:** logprob vs a dense baseline, MTP on/off greedy divergence, needle, code-edit unit test, format/repetition
- **Outcome:** an exit reason (`RAM_FLOOR` / `OOM` / `SPEC_ASSERT` / `ENGINE_ERROR` / `CTX_CEILING`) and flags
- **Context:** the tokens actually filled per session (`ctx_tokens`, `ctx_fill_pct`), the server's `max_model_len` and KV capacity, and each user count's context ceiling

The ladder is progressive, with one boot per config, for every profile
(including `custom`), and it goes to the server's maximum context:
1. Session 1 grows 8K → 32K → 64K → 131K → 196K → 262K → … in one conversation, up to the
   server's `max_model_len` (read from `/v1/models`, else the boot log) minus room for the
   output. It is never refilled and never shrunk.
2. Then 2, 4 and 8 users join at 8K and grow together, rung by rung, all decoding in every
   window, until the top rung or until that user count stops fitting.
3. A user count stops at its **context ceiling**: the rung needs more KV than the server has
   (checked before sending, from the boot log `KV cache size` or `vllm:cache_config_info`), the
   server rejects the context length, or the RAM floor / OOM hits. The run records `CTX_CEILING`
   with the reason and moves on to the next user count instead of failing.

`context_rungs_tokens: auto` is the default (`ladder.max_context` caps it, and an explicit
list still works). Each rung is filled to its target within about 1% using the server's own
tokenizer (`/tokenize`), before anything is sent.

The old grid order (every user at once at the first rung, then step down) is
refused unless a spec sets both `ladder.shape: grid` and `ladder.allow_grid: true`,
and then `bench plan` / `bench run` print a loud warning.

There are three prompt types (code edit, writing, chat) and two load levels (all decoding together, or staggered arrivals). `--resume` continues from the last completed rung. The `quick` profile (session 1 to 32K, then 2/4/8 users at 8K, × 3 types: 15 rungs) is about 11 minutes by the planner's conservative rates; `full` and the shipped specs climb to the server's max context. Details are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md); columns and flags are in [docs/RESULTS-SCHEMA.md](docs/RESULTS-SCHEMA.md).

## Example specs

| Spec | Role |
|---|---|
| [`specs/generic-vllm.yaml`](specs/generic-vllm.yaml) | Any vLLM server: quick progressive ladder, nothing fork-specific declared |
| [`specs/prod-dense-baseline.yaml`](specs/prod-dense-baseline.yaml) | Dense baseline, MTP-6, full matrix |
| [`specs/qsa-full-fix-mtp6.yaml`](specs/qsa-full-fix-mtp6.yaml) | QSA + current full-fix set, MTP-6 |
| [`specs/qsa-vs-dense.yaml`](specs/qsa-vs-dense.yaml) | Prioritized A/B: `K8V4_QSA=1` vs `0` |
| [`specs/qsa-width-clamp.yaml`](specs/qsa-width-clamp.yaml) | Prioritized A/B: width clamp before/after |
| [`specs/dflash2-k7.yaml`](specs/dflash2-k7.yaml) | DFlash2 k=7, capture sizes 8..64 step 8 |
| [`specs/mtp3.yaml`](specs/mtp3.yaml) | Planned MTP-3 (k=3), capture sizes 4..32 |
| [`specs/mtp-off-reference.yaml`](specs/mtp-off-reference.yaml) | Spec decode off: greedy reference for the MTP on/off check |
| [`specs/one-fix-at-a-time.yaml`](specs/one-fix-at-a-time.yaml) | Template for a single-fix A/B |

The other specs are reference configs for one vLLM XPU fork (QSA / K8V4 / MTP / DFlash env flags and graph settings); copy and adapt them. Unknown real values (image digests, served model name, launch args, some env flags, how the runner reaches the bench host) are marked `TODO` in those files. A live run refuses to start until each is set (or set to `null`); `--dry-run` lists them as warnings.

## Safety

These rules are enforced in code, and the tests check them:
- Refuses any endpoint on a protected production port and any protected production container or service, plus CI-runner containers. The real names and ports come from an untracked site config (`bench-site.local.yaml`, `~/.config/xpu-llm-bench/site.yaml`, or `$XPU_BENCH_SITE`; see [`bench-site.example.yaml`](bench-site.example.yaml)). Site values only add protection; the built-in neutral defaults (`vllm-production`, port 8001) always apply. `$BENCH_HOST` can name the host for SSH access.
- Never starts, stops or restarts a container.
- `bench-hold` (optional, when `hold_command` is set): checks `status` first and **refuses if anyone else holds it**, because `acquire` silently overwrites. Also refuses an active operator quiet hold (optional file, from site config). It then acquires under a per-run name (the run id, unless `safety.hold_name` is set), confirms, renews while it still owns the hold, and releases only its own hold. Without a `hold_command` the quiet hold is still honored.
- Preflight stops (exit 3, manifest `status: refused`) on: no site config, `TODO`/`<...>` spec values, a missing or stopped test container, a variant A/B that would share one endpoint, or a bench-hold held by someone else. GPU and host-RAM **estimates never block** a run, a user count or a bigger ladder; they are printed and noted in the manifest.
- Pre-run load check: if the server is already busy (`/metrics` running or waiting requests) or the GPU already loaded (tokens generated over ~2 s, KV in use), it warns and asks once (`--yes` skips the question; a non-interactive run proceeds). The starting load goes into the manifest (`pre_run_load`) and every row is flagged `PRE_EXISTING_LOAD`. No process watching.
- A live `MemAvailable` floor, 2 GiB by default (`hardware.mem_available_floor_gib`), stops the user count that hits it (`CTX_CEILING`, reason `ram_floor`) and the run moves on to the next user count; it never aborts the run. MemAvailable already under the floor before the first rung is a warning (printed and noted in the manifest), not a refusal: the run goes ahead with the floor guard off (MemAvailable still sampled), and the same happens if RAM does not recover after a RAM-floor ceiling. OOM, KV-capacity and server-rejection ceilings always apply. The only pre-run questions are whether the GPU is already loaded or the server already busy.
- Never runs `xpu-smi`, not even discovery.
- API keys come from the environment only.

## License

No license has been chosen yet; until one is added, all rights are reserved.
