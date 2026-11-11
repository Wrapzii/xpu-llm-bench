# Results schema

A run writes these files under `--output-dir/<spec-name>/`:

- `manifest.json`: identity, fingerprint, boot evidence, host facts, hold state, and `completed_rung_ids` (the resume cursor).
- `results.jsonl`: one JSON object per measured rung, in schedule order.
- `launch.json`: the server's start command and settings, captured read-only at preflight (also `manifest.launch`). Fields: `captured_utc`, `captured_local`, `display_timezone`, `endpoint`, `source` (`docker_inspect` / `process_cmdline` / `none`), `container`, `image`, `image_id`, `repo_digests`, `entrypoint`, `cmd`, `args`, `command_line`, `env` (name → value, secrets `<redacted>`), `devices`, `group_add`, `ipc_mode`, `shm_size`, `network_mode`, `port_bindings`, `restart_policy`, `container_started_at`, `vllm_boot_config` (vLLM's "non-default args" / engine-config log lines, redacted), `problems`, `read_only`.
- `run.log`: human-readable launch header (UTC and local time), then one line per rung and the final status.
- `report.html`: self-contained HTML report with inline SVG charts (`bench report --html`; written at the end of every real run unless `--no-html`): log-scaled context axis for wide ladders, a ceiling marker where each user count stopped, quality pass counts per group, and averages across prompt types over clean rows.
- `sessions.json`: a compact record of every live conversation, used by `--resume` to re-prefill: user turns as (segment, n_chars, opener, sha256), regenerated from the seeded corpus on load; model replies verbatim. Never edit it. Not written by `--dry-run`.
- `dry-run/`: a `--dry-run` writes its own `manifest.json` (`status: dry_run`, `planned_rung_ids`) and `DRY_RUN` rows here, never next to a real run.
- `greedy_outputs.jsonl`: greedy window outputs (`probe_id`, token ids, text). This is the MTP on/off reference.
- `logprob_scores.jsonl`: teacher-forced probe scores. A dense run's file is the logprob baseline.

`schemas/result.schema.json` is generated from the Pydantic models in
`xpu_llm_bench.results` (`python scripts/gen_schema.py`). `tests/test_schema.py`
fails if it drifts. Every timestamp is recorded twice: UTC (`*_utc`) and the display timezone
(`*_local`; `$XPU_BENCH_TZ`, else site `timezone`, else system local; named in
`manifest.display_timezone`), both with explicit offsets. Older files with
`started_et` / `finished_et` still load. Definitions of every metric are in
[METHODOLOGY.md](METHODOLOGY.md) §4.

## Manifest (`RunManifest`)

| Field | Meaning |
|---|---|
| `run_id`, `spec_name`, `spec_path`, `profile`, `tool_version` | What produced the run |
| `started_*`, `finished_*`, `status`, `abort_reason` | `in_progress` / `completed` / `aborted` (RAM floor, OOM, SPEC_ASSERT, engine error) / `refused` (preflight or bench-hold refused; nothing ran) / `dry_run` (ignored by `--resume`) |
| `planned_rung_ids` | `--dry-run` only: the rungs a live run would execute |
| `site_config` | which site config file was loaded (or that none was) |
| `launch_check` | spec env / launch_args / image vs `docker inspect` (`mismatches` → `NOT_AS_LABELED`), plus the preflight feasibility result |
| `prompt_seed`, `k` | Fixed workload seed; declared k (null if undeclared) |
| `fingerprint` | `build_fingerprint`, `launch_fingerprint`, image id, per-mount hashes, StartedAt, redacted env |
| `boot` | Parsed boot evidence (attention_boot, QSA_WRAP / QSA_MEMORY_PLAN / banner, graphs, capture sizes, KV dtype and tokens, speculative config) |
| `host` | sysfs GPU ids/driver, kernel, oneAPI / level-zero package versions (no xpu-smi) |
| `hold` | bench-hold status before, our acquire, status after, release (`{"skipped": ...}` when no `hold_command` is configured). The default hold name is the run id (`<spec name>-<utc stamp>-…`), so copies of a spec never share a hold. |
| `launch` | Server start command and settings captured read-only at preflight (same as `launch.json`; secrets redacted). |
| `max_model_len`, `max_model_len_source` | The server's max context and where it was read (`/v1/models` or `boot_log`). |
| `context_rungs`, `context_note` | The context ladder this run used (`auto` resolved) and how it was derived. `--resume` keeps it. |
| `kv_capacity_tokens`, `kv_capacity_source` | Server KV capacity in tokens (`boot_log` "KV cache size", or `metrics cache_config_info`). |
| `ctx_ceilings` | Per `"<prompt type>\|<load mode>\|<users>"`: `{family, load_mode, users, max_ok_ctx, stopped_at, reason, detail, rung_id}`; reason is `kv_capacity`, `server_rejected_context`, `ram_floor` or `oom`. |
| `ctx_reached` | Highest context each `"<prompt type>\|<load mode>\|<users>"` line measured successfully. |
| `pre_run_load` | Two `/metrics` scrapes ~2 s apart before the first rung: `requests_running`, `requests_waiting`, `generation_tokens_in_probe`, `kv_cache_usage_pct`, `server_busy`, `gpu_loaded`, `loaded`, `confirmed`. |
| `notes` | Free-text notes, including the informational host-RAM estimate (`INFO host RAM estimate (never blocks)`) and `PRE_EXISTING_LOAD`. |
| `graph` | `graphs_declared` (from the spec's graph mode, or null), `graphs_on` (true/false only from boot-log evidence, else null) and `graphs_basis` (`boot_log` / `unknown`). Dry runs never claim graphs on. |
| `corpus_files` | sha256 of each corpus file when `prompts.corpus_dir` is used |
| `first_greedy_divergence` | First probe and token where MTP-on output left the reference |
| `image`, `hardware`, `config`, `attention`, `kv_format`, `graph`, `speculative`, `decode_method` | Spec-declared config, plus the method id `fixed-tokens-chunks-v2` and the formula text |
| `completed_rung_ids`, `output_jsonl`, `notes` | Resume cursor, path, notes |

## Rung row (`RungResult`)

Fixed CSV / wide-report column order (`report.COLUMNS`). Steps/s comes
immediately before tok/s:

```
run_id, rung_id, phase, users, ctx_per_user, ctx_tokens, prompt_type, load_mode, prompt_seed, k, build_fingerprint, attention_path, attention_boot, graphs_on, capture_sizes, decode_batch_tokens, kv_format, steps_s_per_user, decode_tok_s_per_user, steps_s_total, decode_tok_s_total, tokens_per_step, accepted_drafts_per_step, hit_rate, acceptance_rate_vllm, verify_steps, verify_steps_server, tokens_delivered, decode_window_s, prefill_tok_s, ttft_ms, tpot_ms, itl_p50_ms, itl_p95_ms, itl_p99_ms, e2e_ms, queue_time_ms, running_reqs, waiting_reqs, running_max, waiting_max, prefix_cache_hit_rate, kv_cache_usage_pct, kv_cache_usage_max_pct, preemptions, mem_available_min_gib, rss_kib, other_vllm_stacks, gpu_mem, eager_fallback, quality, errors, timeouts, exit_reason, flags, valid, reason, decode_tok_s_per_user_steady, decode_tok_s_total_steady, steady_window_s
```

The three steady-state columns were added after the others and are appended
at the end so existing column positions do not move.

The default markdown table (`report.SUMMARY_COLUMNS`) uses:
`rung_id, users, ctx_tokens, attention_path, steps_s_per_user, decode_tok_s_per_user, decode_tok_s_total, decode_tok_s_per_user_steady, decode_tok_s_total_steady, tokens_per_step, hit_rate, prefill_tok_s, ttft_ms, itl_p95_ms, kv_cache_usage_max_pct, mem_available_min_gib, quality, flags`.

Fields on the row that are not in the table: `session_seeds`, `temperature`,
`launch_fingerprint`, `image_id`, `boot_started_at`, `verify_steps_stream`,
`draft_tokens`, `accepted_drafts`, `drafts`, `prefill_tokens`,
`attention_evidence`, `repeats`, `repeat_stats`, and timestamps.

- `rung_id` is `{prompt_type}:{load_short}:{users}u:{ctx}:{phase}`, where phase is `S` (session-1 growth), `C` (a user count climbing), or `meas` (grid shape).
- `ctx_tokens` lists the context of every live decoding session as reported by the server (`prompt_tokens`). `ctx_per_user` is the rung's target; `ctx_fill_pct` is mean(`ctx_tokens`) / target × 100 (filled to within ~1% when the server has `/tokenize`).
- A `CTX_CEILING` row with `exit_reason: CTX_CEILING` was never sent (KV capacity check); other ceiling rows keep the failure's exit reason and carry the `CTX_CEILING` flag.
- `repeats` is the number of decode windows measured on this rung's context (`ladder.repeats`, default 1). When it is > 1, `decode_tok_s_per_user`, `decode_tok_s_per_user_steady`, `steps_s_per_user`, `accepted_drafts_per_step`, `hit_rate`, `ttft_ms` and `prefill_tok_s` are medians across repeats (totals = median per user × users), and `repeat_stats` maps each of those fields to `{values, n, median, min, max, stdev}`: `values` is per repeat in order (null for a failed repeat), `stdev` is the sample stdev and only set when n ≥ 3. Prefill is measured once per rung, so its entry holds that single value (n = 1). With `repeats: 1`, `repeat_stats` is empty. Quality is checked once per rung. These two fields are in the JSONL, not in the fixed CSV columns.
- `decode_tok_s_per_user_steady` / `decode_tok_s_total_steady` are decode tok/s counted only inside the interval where every session of the window is decoding (`[max first token, min last token]`, length `steady_window_s`); see METHODOLOGY §3. They are additive: `decode_tok_s_per_user` keeps its per-request definition. One user: equal to `decode_tok_s_per_user` / `decode_window_s`. Several users with fewer than `ladder.min_window_steps` steps in the overlap for any session: both rates are null, `steady_window_s` still holds the overlap length, and the row is flagged `SHORT_OVERLAP`. Rows written before these fields existed read them as null.
- `quality` holds needle, code-edit, format/repetition, greedy reference/divergence, and logprob scores. Unchecked items are null.
- `valid` is `valid`, `warning` (see METHODOLOGY §6), or `invalid`.
- `exit_reason` is set only on failed rungs.

## Flags

| Flag | Meaning |
|---|---|
| `NOT_AS_LABELED` | Log evidence contradicts the spec label (attention path, graphs, k, KV dtype). |
| `QSA_FALLBACK` | QSA requested/wrapped, but runtime lines in the window show the dense/skip path. |
| `MIXED_PATH` | Both QSA and skip/dense runtime lines in the window. |
| `GRAPH_MISS` | Decode batch tokens users*(k+1) not in the logged/declared capture sizes, or graphs off. |
| `SPEC_COUNTER_BROKEN` | T != k*D, A > T, sum(A_p) != A, or A disagrees with delivered - D. |
| `STEP_COUNT_MISMATCH` | Stream-counted verify steps and server drafts differ by > 5%. |
| `DERIVED_STEPS` | Stream gave no per-step token counts; verify steps taken from server drafts. |
| `K_UNDECLARED` | Spec did not declare k; k was taken from draft_tokens/drafts. |
| `SHORT_WINDOW` | Fewer verify steps per session than ladder.min_window_steps. |
| `SHORT_OVERLAP` | Multi-user window: some session had fewer than ladder.min_window_steps verify steps while all sessions were decoding; the steady-state decode fields are left empty. |
| `EOS_IN_WINDOW` | Model emitted EOS inside the window (ignore_eos kept decoding). |
| `PREEMPTED` | Preemptions > 0 during the rung. |
| `CONCURRENT_CLIENT` | Server still had running/waiting requests after the pre-window wait (ladder.quiesce_timeout_s), or more during the window than this runner's sessions (another client). |
| `FOREIGN_STACK` | Another vLLM stack (different container) is resident on the host. |
| `PROTECTED_PORT` | Server bound to a protected production port (only possible on imported rows). |
| `RAM_FLOOR` | MemAvailable hit the floor; runner guard aborted. |
| `CTX_CEILING` | This user count's context ceiling: the rung did not fit (KV capacity, the server rejected the context, RAM floor or OOM). The user count stops here; the run continues with the next one. |
| `CTX_FILL_OFF` | Filled context more than 2% from the rung target (no server tokenizer to top up). |
| `PRE_EXISTING_LOAD` | The server was busy or the GPU loaded when the run started (manifest `pre_run_load`); the row is a `warning`. |
| `OOM` | Out-of-memory (container OOMKilled or OOM lines in logs). |
| `SPEC_ASSERT` | Known assert: Expected spec_token == num_spec_decodes * (num_speculative_tokens + 1). |
| `ENGINE_ERROR` | Engine error / traceback in logs or HTTP 5xx. |
| `TIMEOUT` | Client timeout. |
| `CLIENT_ERROR` | Request failed without a server-side cause in the logs. |
| `REBOOTED` | Container StartedAt changed mid-run (boot-once violated). |
| `RESUMED_REFILL` | First rung after --resume on a new boot: sessions were re-prefilled. |
| `REPEAT_FAILED` | One of the repeated decode windows (ladder.repeats) failed; it is left out of the median / min / max. |
| `QUALITY_FAIL` | A quality check failed on this rung (see quality fields). |
| `CONVERTED` | Value converted from a legacy definition (import-legacy). |
| `DRY_RUN` | Placeholder row from --dry-run. |
| `ATTENTION_DECLARED` | Imported row: attention_path was declared with --attention-path, not proven by log evidence. |
| `HIT_RATE_UNKNOWN` | Imported row: no tokens_per_step / drafts to derive the true hit rate; hit_rate left empty (the acceptance counter is never used as hit_rate). |

## Comparison

Rows pair on `(users, ctx_per_user, prompt_type, load_mode, phase)`. A pair is
NON-COMPARABLE if any of these hold:
- k, seed or temperature differ
- per-session contexts drift by more than 5%
- attention paths differ
- either row has errors or a flag (other than `K_UNDECLARED`)

## Legacy import

`bench import-legacy src.jsonl dest.jsonl [--correct-summed-ratio K | --correct-mtp6-hit-rate] [--port N]`
maps old fields (`users`, `ctx`, `tok/s` / `total_tps` / `per_user_tps`,
`acc` / `acceptance`, `mem_min_kb`) into this schema. Every imported row is
flagged `CONVERTED`. A protected production port adds `PROTECTED_PORT`. The summed-ratio
correction multiplies by (k+1)/(2k).
