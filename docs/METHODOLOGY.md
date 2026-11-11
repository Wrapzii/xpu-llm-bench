# Methodology

This is the contract for every benchmark run on the bench host. The runner,
planner and reporter implement it, and the tests check it. It follows the
progressive ladder and the true hit-rate definition below.

## 0. Safety rules (enforced in code, also in `--dry-run`)

| Rule | Where |
|---|---|
| Refuse any endpoint (base, metrics, health, tokenize) on a **protected production port**. The ports are the union of the built-in neutral default (8001), `safety.forbidden_ports`, and the untracked site config (`bench-site.local.yaml`; see `bench-site.example.yaml`). | `safety.assert_endpoint_allowed`, `runner.safety_preflight` |
| Refuse protected containers: the built-in `vllm-production`, CI-runner name patterns, `safety.protected_containers`, and the site config's `protected_containers` / `protected_patterns`. Also refuse any container whose `docker inspect` shows a protected host port (port binding or `--port N` under host networking). Site config only adds protection. | `safety`, `site`, `runner._preflight` |
| The runner never starts, stops, or restarts a container. A container it may *manage* must also carry a test prefix (`vllm-bench-`, `vllm-test-`, `vllm-qsa-`, `vllm-dflash`). | `runner`, `safety.assert_test_container_name` |
| bench-hold: run `bench-hold status` first. **Refuse if anyone else holds it**, because `acquire` silently overwrites. Also refuse while the optional operator quiet hold (`safety.quiet_hold_file` / site config, read only) is active. Then `acquire <min> <reason> --by <hold_name>`, re-check `status` to confirm ownership, renew (same `--by`) while running, and `release` only if `status` still shows our name. `release` also lets the host's queued GPU work resume. | `hold.BenchHold.guard_and_acquire`, `release_if_ours` |
| `MemAvailable` (`/proc/meminfo`, locally or over SSH) is sampled every 0.5 s. At or below **2 GiB** the rung stops with `RAM_FLOOR` and becomes that user count's `CTX_CEILING`; the run continues with the next user count and never aborts on RAM (with `ladder.stop_at_ceiling: false` the old stop-and-`--resume` behaviour applies). `--ram-wait-min N` waits up to N min for floor + 1 GiB before moving on. Under the floor before the first rung: a warning only (guard off for that run). | `ram.RamWatcher`, `runner` |
| Never run `xpu-smi`, not even discovery. Every command passes `assert_no_xpu_smi`. GPU facts come from sysfs (`/sys/class/drm/card*/device/{vendor,device,driver}`), `uname`, `/proc`, and `dpkg-query`. | `hostexec`, `hostinfo` |
| API keys come from the environment only (`endpoint.api_key_env`). Secret-looking env vars are redacted in fingerprints. | `client`, `fingerprint.redact_env` |

## 1. Ladder (progressive ladder, `ladder.shape: session1_then_join`)

One boot per config. `max-num-seqs` must cover the largest concurrency step.
Never restart to change concurrency.

**Context rungs.** `ladder.context_rungs_tokens: auto` (the default) builds
8K, 32K, 64K, 131K, then +64K steps (196K, 262K) and doubling beyond 262K
(524K, 1M, ...; `auto_step_tokens` / `auto_step_mode` change this), up to the
server's `max_model_len` minus `ctx_headroom_tokens` (default: output window +
1024), with that maximum as the final rung. `max_model_len` is read at
preflight from `/v1/models` (vLLM reports it per model), else from the boot
log. `ladder.max_context` caps the ladder; an explicit list still works and
rungs above the limit are dropped. `bench plan` cannot ask the server, so it
shows up to 131K unless given `--max-model-len`. The resolved rungs are stored
in the manifest (`context_rungs`) and a `--resume` keeps them.

**Fill.** Each rung is filled to its target: the new user turn is sized from a
chars-per-token estimate, then counted with the server's own tokenizer
(`/tokenize` with the chat messages, template included) and topped up or
trimmed before anything is sent, to within 1%. Rows record the tokens the
server reported (`ctx_tokens`) and `ctx_fill_pct`; a row more than 2% off
(no tokenizer available) is flagged `CTX_FILL_OFF`.

**Phase S (session growth).** Session 1 grows through every context rung, up
to the top rung, in **one conversation**. Each growth appends a new user turn
of fresh corpus text. After each measured window the model's own reply, cut at
EOS, is appended as the assistant turn. Earlier turns are never edited, so the
rendered prefix is stable and the prefix cache is reused. The conversation is
never refilled from scratch and never shrinks. One measured rung per context.

**Phase C (users join and climb).** For each concurrency step n > 1 (2, 4, 8),
n users join at `join_ctx_tokens` (default: the first rung, 8K), each
prefilling on its own, and then grow **together** rung by rung, with every one
of the n users decoding in each window, until the top rung or that user
count's ceiling. Each user count has its own conversations, so all n users sit
at the same context and every concurrency gets one clean curve to the
maximum. Conversations still only grow (append-only, prefix reuse); the
previous user count's conversations are released, never shrunk or refilled.
(Before this change session 1 stayed live at the top context while joiners sat
at 8K, which capped multi-user rows at the first rung.) The quick profile
keeps 2/4/8 users at 8K (`join_max_context_tokens: 8192`).

**Context ceilings.** A user count stops climbing when the next rung does not
fit, and the run records it instead of failing:
- `kv_capacity`: the rung's live KV tokens (n × context + n × output) exceed
  the server's KV capacity (boot log `KV cache size: N tokens`, else
  `vllm:cache_config_info` blocks × block size). Checked before sending, so
  nothing is sent; the row has exit reason `CTX_CEILING`.
- `server_rejected_context`: the server refuses the request length.
- `ram_floor` / `oom`: MemAvailable hit the floor or the engine was OOM-killed
  during the rung. After a RAM-floor ceiling the run waits up to
  `--ram-wait-min` for MemAvailable to recover and moves on to the next user
  count; if RAM did not recover it notes that and turns the floor guard off
  for the rest of the run. A RAM floor never aborts the run. MemAvailable
  already under the floor before the first rung is a warning (printed and in
  `manifest.notes`), not a refusal, and the run goes ahead with the floor
  guard off (MemAvailable is still sampled per rung).
The failed rung is flagged `CTX_CEILING`, the manifest records
`ctx_ceilings["<prompt type>|<load mode>|<users>"] = {max_ok_ctx, stopped_at,
reason, detail}`, the remaining rungs of that user count are skipped, and the
next user count starts with its own ceiling logic. `ladder.stop_at_ceiling:
false` restores the old behaviour (abort on the first failure).

**Pre-run load.** Before the first rung the runner scrapes `/metrics` twice,
about 2 s apart. If requests are running or waiting (server busy) or tokens
were generated / KV is in use in between (GPU loaded), it warns and asks once
(`--yes` answers yes; a non-interactive run proceeds), records
`manifest.pre_run_load` and flags every row `PRE_EXISTING_LOAD` (a `warning`
row). Nothing else is watched.

**Prompt types.** `code_edits`, `long_form` (writing), and `multi_turn_chat`
are separate families with separate sessions. Within a family, context
never shrinks.

**Load levels.** `closed_all_decoding` means all sessions start their
window together. `staggered` offsets arrivals by `ladder.staggered.delay_s`.
`open_loop` arrives at a fixed rate. Rows only compare within the same load
mode.

**Profiles.**

| | quick (≈ 11 min by the planner's conservative rates) | full |
|---|---|---|
| contexts | 8K, 32K | auto: 8K … the server's max context |
| users | 1 (S, to 32K), 2/4/8 (C, at 8K) | 1 (S), 2/4/8 (C), each to its ceiling |
| prompt types | all three | all three |
| load modes | closed | closed, staggered |
| window | 96 tokens | 256 tokens |

The quick preset on a typical 2-GPU box: 15 rungs, at most 8 × 8K = 64K live
tokens (well inside any 2 × 24 GiB KV budget), session 1 alone at 32K. It is a
smoke test of the whole pipeline, not a context sweep; use `full` (or a
shipped spec) for the climb to max context.

`bench plan` prints the order, per-session contexts, a wall-time estimate
from `planner.assumed_*` (not measured), the context rungs, a **per-user-count
context ceiling estimate** (from the KV capacity: `--kv-capacity-tokens`, or
`hardware.gpu_*` with `planner.assumed_*`), and GPU / host-RAM estimates.
**None of the estimates block a run** (exit 0); they are information:
- GPU: weights + KV(peak live tokens of the rungs estimated to fit) + device
  overhead against `gpu_count × gpu_mem_gib` (WARN when over).
- Host RAM: growth over the run, peak live tokens ×
  `planner.host_bytes_per_token` (ESTIMATE, default 64 KiB) plus any
  host-resident KV share, against live MemAvailable minus the floor. Shown on
  one line in the plan and as an `INFO` note in the manifest at preflight.
  The live RAM floor during the run is what stops a user count.

**Resume.** After each rung the runner saves `manifest.json` (with
`completed_rung_ids`) and `sessions.json` (a compact record of every
conversation: each user turn as its corpus segment index, length and a
SHA-256, regenerated from the seeded corpus on load; the model's replies
verbatim). `bench run --resume` reloads both (refusing if a regenerated turn
does not match its hash) and re-prefills each live session to its
high-water context **outside** any window. Sessions are never rebuilt
smaller. If the container's `StartedAt` changed, the first resumed rung is
flagged `RESUMED_REFILL`.

The progressive ladder is the default for every profile, including
`custom`. The old scaffold order (`shape: grid`: all concurrencies at 8K,
then grow everyone and quiesce down) is refused unless the spec also sets
`allow_grid: true`; `bench plan` and `bench run` then print a loud warning.
Grid rows carry `phase=meas` and are never compared with S/C rows.

**Dry run.** `--dry-run` writes only under `<out>/<spec>/dry-run/`: a
manifest with `status: dry_run` and `planned_rung_ids` (never
`completed_rung_ids`), `DRY_RUN` placeholder rows, and no `sessions.json`.
`--resume` ignores a `dry_run` manifest. The safety commands a live run
would execute are printed to stdout.

## 2. Workload

- **Fixed seed.** `prompts.seed` (default 20261111) is recorded on every row
  as `prompt_seed`. The per-session seed is
  `seed*1000 + family_index*100 + sid` (`session_seeds`). Nothing is
  timestamp-based. The same seed and spec produce byte-identical
  conversations (tested).
- **Realistic text** (`prompts.style: realistic`, the default): English prose,
  real-looking Python, and support-chat turns built from fixed vocabularies by
  the seeded RNG. It is not repeated padding. For headline numbers, prefer
  `prompts.corpus_dir` (real files per prompt type, read in sorted order,
  sha256 of each file in the manifest). The original syllable filler is
  `style: synthetic`.
- **Needles.** Every growth segment plants one fact ("the access code for
  storage locker N is colour-NNNN") halfway through the segment.
- **Sampling.** `ladder.temperature` defaults to 0 (greedy), which keeps A/B
  workloads identical and makes the MTP on/off check meaningful. Measured
  windows send `ignore_eos: true` so every session decodes exactly
  `output_tokens_per_rung`. They also send `skip_special_tokens: false`, so
  an EOS inside the window is visible (`EOS_IN_WINDOW`), and
  `return_token_ids: true`.

## 3. Decode window and verify steps (method id `fixed-tokens-chunks-v2`)

**Window.** Per session, the window runs from its first streamed token to its
last token of a fixed `output_tokens_per_rung`. Fixed tokens rather than fixed
seconds means every config decodes the same greedy tokens.

**Verify steps from the stream.** vLLM streams one SSE chunk per engine step
per request (`stream_interval` 1). With MTP a step carries 1..k+1 tokens. The
first chunk is the prefill step. Every later token-bearing chunk is one
verify step. Tokens per chunk come from `token_ids` if returned, else from
the `continuous_usage_stats` `completion_tokens` delta. If neither is
present, steps come from the server (`DERIVED_STEPS`).

**Cross-check from the server.** `S_server` = Δ`vllm:spec_decode_num_drafts_total`
over the window. Every verify step of every request counts one draft.
`|S_stream − S_server| / S_server > 5%` → `STEP_COUNT_MISMATCH`. Counter
snapshots are taken after growth and before the window, so prefill steps are
excluded.

**Steady state with several users.** All sessions of a `closed_all_decoding`
window are sent together, but each request first prefills its new tail, so
the first part of one session's decode window overlaps the other sessions'
prefills. With short windows (the quick profile's 96 tokens) most of a
per-request window can be prefill-contended, which makes the multi-user
`decode_tok_s_per_user` (E_i / window_i) read low. The runner therefore also
reports a steady-state rate over the interval in which every session is
decoding: `[max_i first token_i, min_i last token_i]` (`steady_window_s`).
Each session's token-bearing chunks that arrive inside that interval are
counted (a chunk exactly at its start belongs to the step before, as in the
per-request window); `decode_tok_s_per_user_steady` is the mean of
tokens_i / `steady_window_s` and `decode_tok_s_total_steady` the sum. If any
session has fewer than `ladder.min_window_steps` steps inside the overlap the
two rates are left empty and the row is flagged `SHORT_OVERLAP`; raise
`output_tokens_per_rung` to get a usable overlap. With one user the overlap is
the request's own window, so the steady values equal the per-request ones.
These fields are additional; `decode_tok_s_per_user` is unchanged, so older
results stay comparable.

**Quiescence before a window.** vLLM's `num_requests_running` /
`num_requests_waiting` gauges can still count the runner's own requests from
growth or the previous rung for a moment after they finish. Before the
window's `before` snapshot the runner polls `/metrics` (every 0.25 s, up to
`ladder.quiesce_timeout_s`, default 10 s) until both are 0. `CONCURRENT_CLIENT`
is set only when the server is still busy after that wait, or when running +
waiting during the window exceeds the runner's own sessions.

**Repeats (`ladder.repeats`, `--repeats N`, default 1).** With N > 1 the
decode window is measured N times back to back on the same context: the
context is grown once, the same messages are re-sent for each repeat, and
no reply is appended until all N windows are done (repeat 1's reply is the
one kept, so later rungs see the same conversation as with N = 1).
For decode tok/s (per-request and steady), steps/s, accepted drafts/step, true hit rate and TTFT
the row stores the median across repeats, and `repeat_stats` keeps every
per-repeat value with min, max and (N ≥ 3) the sample stdev. Totals are the
median per user × users. Prefill is measured once per rung (from the growth
requests; nothing is regrown), so its `repeat_stats` entry holds that one
value (n = 1) and the report shows no range for it. Flags, ITL, KV/queue samples and the attention path come from
repeat 1. Quality checks run once per rung, not per repeat. A repeat that
errors is left out of the stats and flags the row `REPEAT_FAILED`. Re-sent
prompts are normally served from the prefix cache, so repeat TTFTs are
lower than a cold first request; compare TTFT only across runs with the
same `repeats`. `bench plan` multiplies the decode window by N.

## 4. Metrics (per rung)

Notation: k = `server.speculative.num_speculative_tokens` (**a spec parameter,
never hard-coded**: MTP-6 k=6, DFlash2 k=7, MTP-3 k=3). If a spec omits k,
k = draft_tokens/drafts from the counters and every row is flagged
`K_UNDECLARED`. E_i = delivered decode tokens of session i (tokens after its
first). S_i = its verify steps. E = ΣE_i, S = ΣS_i.

| Field | Definition |
|---|---|
| `steps_s_per_user` (**headline**) | mean_i S_i / window_i |
| `decode_tok_s_per_user` (next to it) | mean_i E_i / window_i |
| `steps_s_total`, `decode_tok_s_total` | Σ over sessions |
| `decode_tok_s_per_user_steady`, `decode_tok_s_total_steady`, `steady_window_s` | mean / Σ of tokens_i inside the all-decoding overlap / its length (§3) |
| `tokens_per_step` | E / S. Identity: tok/s = steps/s × tokens_per_step |
| `accepted_drafts_per_step` | E/S − 1 (0..k) |
| **`hit_rate` (true hit rate)** | **(E − S) / (k × S)**: the fraction of proposed draft slots that became delivered tokens. It never reads the accepted-tokens counter, which some builds over-report (accepted > drafted, i.e. above 100%). |
| `acceptance_rate_vllm` | Δaccepted / Δdraft_tokens, kept for continuity only |
| `verify_steps`, `verify_steps_stream`, `verify_steps_server`, `tokens_delivered` | the raw counts behind the above |
| `prefill_tok_s` | Σ(new prompt tokens) / Σ(TTFT) over this rung's growth requests (sequential). New = `prompt_tokens − cached_tokens` (needs `--enable-prompt-tokens-details`), else Δprefix queries − Δprefix hits |
| `ttft_ms` | client clock, from **request send** to the first streamed token, mean over measured sessions |
| `tpot_ms` | (e2e − TTFT) / (tokens − 1) |
| `itl_p50/p95/p99_ms` | gaps between consecutive token-bearing chunks, pooled over sessions. With MTP one chunk is one step, so ITL is the per-step latency the user sees. |
| `e2e_ms` | request send → stream end, mean |
| `queue_time_ms` | Δsum/Δcount of `vllm:request_queue_time_seconds` |
| `running_reqs`/`waiting_reqs` (mean), `running_max`/`waiting_max` | `/metrics` polled every `ladder.metrics_poll_s` during the window |
| `prefix_cache_hit_rate` | Δhits / Δqueries over the window |
| `kv_cache_usage_pct`, `kv_cache_usage_max_pct` | `vllm:kv_cache_usage_perc` ×100 (V1 reports a fraction), or legacy `gpu_cache_usage_perc` |
| `preemptions` | Δ`vllm:num_preemptions_total` |
| `mem_available_min_gib` | minimum `MemAvailable` over the rung |
| `rss_kib` | peak VmRSS of `api_server`, `engine` (`VLLM::EngineCore`), and each `worker_tpN`, from `/proc`, grouped by the container's cgroup id |
| `other_vllm_stacks` | vLLM processes in a different container (→ `FOREIGN_STACK`) |
| `errors`, `timeouts`, `exit_reason` | see §6 |

**Counter identities** (any failure → `SPEC_COUNTER_BROKEN`; the true hit
rate is still reported):
- T = k·D
- A ≤ T
- Σ_p A_p = A
- A ≈ E − D, allowing up to k discarded drafts per request on the final step (max_tokens truncation) plus 2%

## 5. What actually ran (from logs and docker inspect, never from flags)

Read at preflight with `docker logs --timestamps --tail 200000 <test
container>`. Runtime lines are filtered to the window by timestamp.

| Field | Evidence |
|---|---|
| `build_fingerprint` | sha256 over the image ID plus, for every bind mount, `(destination, sha256)`. Model weights are never read, whatever the mount path: weight-type files (`.safetensors .bin .gguf .pt .pth .ckpt .onnx .msgpack .h5 .npz`) and any file over 16 MiB contribute name + size only, from one metadata-only `find` pass (symlinks contribute their target, so an HF snapshot points at its blob name). Only small non-weight files (patch `.py`, config / tokenizer `.json`) are content-hashed; a directory hashes their sorted `sha256sum` manifest merged with those name + size lines (no `__pycache__`, `.git` or HF-cache `blobs/`). Mounts at `/model`, `/draft`, `/root/.cache` and `/data` are listed by name + size only. Validation (and the per-rung reboot check) uses `docker inspect` alone; mounts are hashed once, after every pre-run check has passed, and a hashing failure is a fingerprint problem, never a reason not to run. Test containers often share one locally built image (no RepoDigest), so the image ID alone does not identify a build. |
| `launch_fingerprint` | sha256 of Cmd/Args + Env (secrets redacted) |
| `attention_boot` | `qsa` only if both `QSA_WRAP` **and** `QSA_MEMORY_PLAN` lines appear. `dense` if the banner `Using Xe2 K8/V4 attention backend.` appears with no QSA lines. One of the two QSA lines alone → `unknown`. The banner is printed by both builds and is never sufficient. |
| `attention_path` (per rung) | runtime `QSA_GRAPH width=` / `skip` lines inside the window, if any (`qsa`/`fallback`/`mixed`); otherwise `attention_boot` |
| graphs | `graphs_on` = `Graph capturing finished` present and no `enforce_eager`. Also records `cudagraph_mode`, logged `cudagraph_capture_sizes`, and `Capturing CUDA graphs (MODE) n/N` |
| `decode_batch_tokens` | users × (k+1); `GRAPH_MISS` if graphs are off or the batch is above the largest capture size |
| KV | logged `kv_cache_dtype` and `XPU KV cache size: N tokens` |
| speculative | `SpeculativeConfig(method=…, num_spec_tokens=…)` must match the spec's method and k |

`NOT_AS_LABELED` is set when evidence contradicts the label:
- QSA requested but the boot is dense, or the reverse
- logged k ≠ spec k
- logged KV dtype ≠ spec
- graphs requested but off

## 6. Failures and flags

`exit_reason`, most specific first:
1. `RAM_FLOOR`: the runner guard tripped
2. `SPEC_ASSERT`: the known assert `Expected spec_token == num_spec_decodes * (num_speculative_tokens + 1)` in logs or the error body; also flagged on a successful rung if it appears in the window
3. `OOM`: `State.OOMKilled`, or out-of-memory / `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` / `Killed process` lines
4. `ENGINE_ERROR`: `EngineDeadError`, `EngineCore encountered`, `Traceback`, `RuntimeError:`, `AssertionError`
5. `TIMEOUT`
6. `CLIENT_ERROR`

`RAM_FLOOR`, `OOM`, `SPEC_ASSERT` and `ENGINE_ERROR` stop the run (`--resume` continues).

The full flag list with meanings is `results.FLAGS` and appears in each report.
- `valid=warning`: `NOT_AS_LABELED`, `SPEC_COUNTER_BROKEN`, `SPEC_ASSERT`, `REBOOTED`, `CONCURRENT_CLIENT`, `QSA_FALLBACK`, `MIXED_PATH`
- `valid=invalid`: failed rungs
- `bench compare` marks any flagged pair NON-COMPARABLE, except `K_UNDECLARED`

## 7. Quality, in the same pass (outside every window)

1. **Logprob scoring.** Four fixed (context, reference answer) probes, built
   from the seed, are tokenized via `/tokenize` and scored teacher-forced with
   vLLM `prompt_logprobs` (fallback: `echo` + `logprobs`). The score is the
   mean logprob over the answer tokens.
   - `logprob_mode: short` (default) uses a ~`probe_ctx_tokens` context each rung.
   - `ladder_ctx` prepends session 1's whole conversation. vLLM skips prefix-cache reads for `prompt_logprobs`, so that re-prefills the full context; use it deliberately.
   - Scores go to `logprob_scores.jsonl`. A dense K8/V4 run's file is the baseline (`quality.logprob_baseline`). Fail if current − baseline < −`logprob_tolerance_nats` (default 0.05) on any probe.
2. **MTP on vs off.** Every measured greedy output is written to
   `greedy_outputs.jsonl` (`probe_id = <rung_id>:s<sid>`, token ids + text).
   A run of `specs/mtp-off-reference.yaml` provides the reference. The row
   records `match`, `diverged` with the first differing token index, or
   `no_reference`. Because replies are appended to the conversation, later
   rungs of a diverged session are marked `context_diverged`. The manifest
   keeps the first divergence.
3. **Pass/fail tasks.**
   - **Needle:** at every rung, session 1 is asked for the first planted fact and the newest one (greedy, 24 tokens, on its live context).
   - **Code edit:** on code-edit S rungs, a small buggy function plus a request to fix it is appended to session 1's context. The reply's code runs against fixed asserts in an isolated child Python (`-I`, temp dir, CPU 5 s, 512 MB address-space rlimit, empty env).
   - **Format / repetition:** on writing and chat outputs, the text is cut at EOS. Checks: leaked special tokens, non-printables, chat role bleed (`User:`), 6-gram repeat ratio > 0.35, and the same line 3+ times.

Any failure → `QUALITY_FAIL`. With `ladder.repeats` > 1 the checks still
run once per rung, on repeat 1's output.

## 8. Reporting and comparison

`bench report <results.jsonl | run dir>` prints markdown to stdout
(`--output FILE` writes it to a file instead): header (fingerprint,
boot evidence, hold), quoted evidence lines, **one table** with steps/s next
to tok/s, flags legend. `--csv` writes every column. The column order is
fixed (`report.COLUMNS`).

HTML (`--html`): when a run used repeats, chart points are medians with
min–max error bars, and the averages cards and All numbers show
`median (min to max, n=N)` (stdev on hover in All numbers when N ≥ 3).
With the default `repeats: 1` the report is unchanged.

`bench compare A B` pairs rows on (users, ctx, prompt type); repeated keys
within one run (e.g. two load modes) stay separate pairs in order. A pair
is NON-COMPARABLE if load mode or ladder phase differ, or if any of these differ:
- k (compare steps/s, not hit rate)
- seed
- temperature
- per-session contexts, by more than 5%

It is also NON-COMPARABLE if either row has a flag, path mismatch, fallback, error, or the RAM floor. Deltas are B − A.

Graph A/B (≥ 25% bar): same image, fingerprint (apart from the graph
change), prompt seed and k. Report the steps/s change at S-8K, S-32K,
S-131K, C-2u and C-4u.

## 9. Legacy rows

`bench import-legacy` maps old JSONL into this schema with the `CONVERTED`
flag (never headline). It reads scaffold-era rows (users, ctx, tok/s, acc)
and speed-fix ladder rows (`prompt_tokens`, `rates`, `total_toks`,
`per_user_steps_s`, `first_token_s`, `mem.min_kb`, `tokens_per_step`,
`drafts_delta`, `draft_tokens_delta`, `accepted_tokens_delta`). Users come
from `prompt_tokens`, the rung context from the label (`...-8192`), phase
from the label (`join` → C, `start` → S).

`hit_rate` is derived as `(tokens_per_step − 1) / k` with k =
draft_tokens/drafts (or from delivered tokens and drafts). It is **never**
copied from the acceptance counter: counter values go to
`acceptance_rate_vllm`, and a row with nothing to derive from gets an
empty `hit_rate` and `HIT_RATE_UNKNOWN`. The old summed ratio
2A/((k+1)D) converts to A/(kD) in `acceptance_rate_vllm` with
`--correct-summed-ratio K` (factor (k+1)/(2k), 7/12 for k=6;
`--correct-mtp6-hit-rate` is the k=6 shortcut). `--port N` (or a legacy
URL) on a protected production port adds `PROTECTED_PORT`.

## 10. Not settled (need the operator or a live check)

Open items:
- whether vLLM 0.30 XPU emits exactly one SSE chunk per step (the `STEP_COUNT_MISMATCH` check guards this)
- whether the build returns `token_ids` on chat streams
- the meaning of `QSA_GRAPH skip starts=… n=…` at boot
- the QSA stats file format, needed for a per-window sparse/dense call count
- real prefill/decode rates for the planner
