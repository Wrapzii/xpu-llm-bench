"""Result schema: one manifest per run, one JSONL row per measured rung."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import AliasChoices, BaseModel, Field

from xpu_llm_bench import TZ_ENV, __version__

AttentionPath = Literal["qsa", "dense", "fallback", "mixed", "unknown"]
Validity = Literal["valid", "invalid", "warning"]

# Row flags. A flagged row is excluded from headline numbers and from clean A/B.
FLAGS: dict[str, str] = {
    "NOT_AS_LABELED": "Log evidence contradicts the spec label (attention path, graphs, k, KV dtype).",
    "QSA_FALLBACK": "QSA requested/wrapped, but runtime lines in the window show the dense/skip path.",
    "MIXED_PATH": "Both QSA and skip/dense runtime lines in the window.",
    "GRAPH_MISS": "Decode batch tokens users*(k+1) not in the logged/declared capture sizes, or graphs off.",
    "SPEC_COUNTER_BROKEN": "T != k*D, A > T, sum(A_p) != A, or A disagrees with delivered - D.",
    "STEP_COUNT_MISMATCH": "Stream-counted verify steps and server drafts differ by > 5%.",
    "DERIVED_STEPS": "Stream gave no per-step token counts; verify steps taken from server drafts.",
    "K_UNDECLARED": "Spec did not declare k; k was taken from draft_tokens/drafts.",
    "SHORT_WINDOW": "Fewer verify steps per session than ladder.min_window_steps.",
    "SHORT_OVERLAP": (
        "Multi-user window: some session had fewer than ladder.min_window_steps verify steps while all "
        "sessions were decoding; the steady-state decode fields are left empty."
    ),
    "EOS_IN_WINDOW": "Model emitted EOS inside the window (ignore_eos kept decoding).",
    "PREEMPTED": "Preemptions > 0 during the rung.",
    "CONCURRENT_CLIENT": (
        "Server still had running/waiting requests after the pre-window wait (ladder.quiesce_timeout_s), "
        "or more during the window than this runner's sessions (another client)."
    ),
    "FOREIGN_STACK": "Another vLLM stack (different container) is resident on the host.",
    "PROTECTED_PORT": "Server bound to a protected production port (only possible on imported rows).",
    "RAM_FLOOR": "MemAvailable hit the floor; runner guard aborted.",
    "PRE_EXISTING_LOAD": (
        "The server was already busy or the GPU already loaded when the run started (see manifest "
        "pre_run_load); numbers include that load."
    ),
    "CTX_FILL_OFF": (
        "Filled context is more than 2% away from the rung target (see ctx_tokens / ctx_fill_pct); "
        "the server tokenizer (/tokenize) was not available to top up the prompt."
    ),
    "CTX_CEILING": (
        "This user count's context ceiling: the rung did not fit (KV capacity, the server rejected "
        "the context, RAM floor or OOM). The user count stops here; the run continues with the next one."
    ),
    "OOM": "Out-of-memory (container OOMKilled or OOM lines in logs).",
    "SPEC_ASSERT": "Known assert: Expected spec_token == num_spec_decodes * (num_speculative_tokens + 1).",
    "ENGINE_ERROR": "Engine error / traceback in logs or HTTP 5xx.",
    "TIMEOUT": "Client timeout.",
    "CLIENT_ERROR": "Request failed without a server-side cause in the logs.",
    "REBOOTED": "Container StartedAt changed mid-run (boot-once violated).",
    "RESUMED_REFILL": "First rung after --resume on a new boot: sessions were re-prefilled.",
    "REPEAT_FAILED": "One of the repeated decode windows (ladder.repeats) failed; it is left out of the median / min / max.",
    "QUALITY_FAIL": "A quality check failed on this rung (see quality fields).",
    "CONVERTED": "Value converted from a legacy definition (import-legacy).",
    "HIT_RATE_UNKNOWN": "Imported row: no tokens_per_step / drafts to derive the true hit rate; hit_rate left empty (the acceptance counter is never used as hit_rate).",
    "ATTENTION_DECLARED": "Imported row: attention_path was declared with --attention-path, not proven by log evidence.",
    "DRY_RUN": "Placeholder row from --dry-run.",
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def local_tz_name() -> str:
    """IANA name of the display timezone, or "system-local"."""
    import os

    name = os.environ.get(TZ_ENV, "").strip()
    if name:
        return name
    try:
        from xpu_llm_bench.site import load_site

        site_tz = load_site().timezone
    except Exception:  # noqa: BLE001
        site_tz = None
    return site_tz or "system-local"


def local_now(now: datetime | None = None) -> datetime:
    src = now or utc_now()
    name = local_tz_name()
    if name == "system-local":
        return src.astimezone()
    return src.astimezone(ZoneInfo(name))


def stamp_pair(now: datetime | None = None) -> tuple[str, str]:
    """Return (utc_iso, local_iso), both with explicit offsets."""
    utc = (now or utc_now()).astimezone(UTC)
    return utc.isoformat(), local_now(utc).isoformat()


class HardwareSnapshot(BaseModel):
    host: str = "unspecified"
    gpu_model: str = "unspecified"
    gpu_count: int | None = None
    tensor_parallel: int | None = None
    driver: str | None = None  # TODO: collect if available without xpu-smi
    oneapi: str | None = None  # TODO
    host_ram_gib: float | None = None
    mem_available_floor_gib: float = 2.0


class GraphSnapshot(BaseModel):
    mode: str | None = None
    per_gpu: bool = False
    capture_sizes: list[int] = Field(default_factory=list)
    # From evidence only (the boot log); None = unknown (e.g. a dry run).
    graphs_on: bool | None = None
    graphs_basis: Literal["boot_log", "unknown"] = "unknown"
    # What the spec declared (graph.mode != EAGER); not evidence.
    graphs_declared: bool | None = None
    eager_fallback: bool = False


class SpecDecodeSnapshot(BaseModel):
    method: str | None = None
    num_speculative_tokens: int | None = None
    dflash_k: int | None = None


class AttentionSnapshot(BaseModel):
    requested: str | None = None
    actual: AttentionPath = "unknown"
    evidence_lines: list[str] = Field(default_factory=list)


class DecodeMethod(BaseModel):
    method_id: str = "fixed-tokens-chunks-v2"
    window: str = "fixed_output_tokens"
    output_tokens: int
    counters: list[str] = Field(default_factory=list)
    formula: str = (
        "Per session: window = first token -> last token of a fixed max_tokens output "
        "(ignore_eos). delivered E_i = tokens after the first (the first is the prefill step); "
        "verify steps S_i = token-bearing SSE chunks after the first (one chunk per engine step). "
        "decode tok/s = E_i / window; steps/s = S_i / window. "
        "true hit rate = (E - S) / (k * S) with E, S summed over sessions; "
        "S_server = delta(spec_decode_num_drafts_total) cross-checks S (STEP_COUNT_MISMATCH > 5%). "
        "accepted_drafts_per_step = E/S - 1 (from the stream); acceptance_rate_vllm = A/T is kept "
        "for continuity only."
    )


class QualityResult(BaseModel):
    needle_pass: bool | None = None
    needle_detail: str | None = None
    code_edit_pass: bool | None = None
    code_edit_detail: str | None = None
    format_ok: bool | None = None
    format_problems: list[str] = Field(default_factory=list)
    repetition_max: float | None = None
    greedy_ref: str | None = None  # match | diverged | context_diverged | no_reference
    greedy_first_divergence: int | None = None  # token index in the session output
    logprob_ok: bool | None = None
    logprob_worst_delta: float | None = None  # current - baseline mean logprob (nats/token)
    logprob_scores: dict[str, float] = Field(default_factory=dict)


# Metrics re-measured per repeat (ladder.repeats). The row's own field holds the
# median across repeats; repeat_stats keeps every value plus min/max/stdev.
REPEAT_METRICS: tuple[str, ...] = (
    "decode_tok_s_per_user",
    "decode_tok_s_per_user_steady",
    "steps_s_per_user",
    "accepted_drafts_per_step",
    "hit_rate",
    "ttft_ms",
    "prefill_tok_s",
)


class RepeatStat(BaseModel):
    """One metric across the repeated decode windows of a rung."""

    values: list[float | None] = Field(default_factory=list)  # per repeat, in order
    n: int = 0  # repeats with a value
    median: float | None = None
    min: float | None = None
    max: float | None = None
    stdev: float | None = None  # sample stdev; only when n >= 3


def repeat_stat(values: list[float | None]) -> RepeatStat:
    import statistics

    xs = [float(v) for v in values if v is not None]
    if not xs:
        return RepeatStat(values=list(values), n=0)
    return RepeatStat(
        values=list(values),
        n=len(xs),
        median=statistics.median(xs),
        min=min(xs),
        max=max(xs),
        stdev=statistics.stdev(xs) if len(xs) >= 3 else None,
    )


class RungResult(BaseModel):
    run_id: str
    rung_id: str
    users: int
    ctx_tokens: list[int]  # per live session, in session order
    ctx_per_user: int  # representative / target context for this rung
    # Mean of ctx_tokens (server-reported prompt tokens) as % of ctx_per_user.
    ctx_fill_pct: float | None = None
    prompt_type: str
    load_mode: str
    phase: str = "meas"  # S (session 1 grows) | C (sessions join) | meas (grid)
    # Identity of what ran
    prompt_seed: int | None = None
    session_seeds: list[int] = Field(default_factory=list)
    temperature: float | None = None
    k: int | None = None
    build_fingerprint: str | None = None
    launch_fingerprint: str | None = None
    image_id: str | None = None
    boot_started_at: str | None = None
    attention_boot: str | None = None
    capture_sizes: list[int] = Field(default_factory=list)
    decode_batch_tokens: int | None = None
    # Prefill / TTFT
    prefill_tok_s: float | None = None
    prefill_tokens: int | None = None
    ttft_ms: float | None = None
    # Decode
    decode_tok_s_per_user: float | None = None
    decode_tok_s_total: float | None = None
    steps_s_per_user: float | None = None
    steps_s_total: float | None = None
    tpot_ms: float | None = None
    decode_window_s: float | None = None
    # Steady state: only the interval where every session is decoding
    # ([max first token, min last token]), so other sessions' prefill is not
    # mixed in. One user: same as decode_tok_s_per_user / decode_window_s.
    # Empty (None) with SHORT_OVERLAP when the overlap is too short.
    decode_tok_s_per_user_steady: float | None = None
    decode_tok_s_total_steady: float | None = None
    steady_window_s: float | None = None  # overlap length (kept even with SHORT_OVERLAP)
    # MTP / speculation. hit_rate is the TRUE hit rate:
    # (delivered - verify_steps) / (k * verify_steps).
    accepted_drafts_per_step: float | None = None
    hit_rate: float | None = None
    acceptance_rate_vllm: float | None = None  # A/T from the counters, for continuity
    tokens_per_step: float | None = None
    tokens_delivered: int | None = None
    verify_steps: int | None = None
    verify_steps_stream: int | None = None
    verify_steps_server: int | None = None
    draft_tokens: int | None = None
    accepted_drafts: int | None = None
    drafts: int | None = None
    # Latency
    itl_p50_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None
    e2e_ms: float | None = None
    # Queue / cache / KV
    queue_time_ms: float | None = None
    running_reqs: float | None = None
    waiting_reqs: float | None = None
    running_max: float | None = None
    waiting_max: float | None = None
    kv_cache_usage_max_pct: float | None = None
    prefix_cache_hit_rate: float | None = None
    kv_cache_usage_pct: float | None = None
    preemptions: int | None = None
    # Memory
    mem_available_min_gib: float | None = None
    gpu_mem: str | None = None  # free-form; never collected via xpu-smi
    rss_kib: dict[str, int] = Field(default_factory=dict)  # api_server, engine, worker_tp0, ...
    other_vllm_stacks: list[str] = Field(default_factory=list)
    # Path / graphs actually observed
    attention_path: AttentionPath = "unknown"
    attention_evidence: list[str] = Field(default_factory=list)
    graphs_on: bool | None = None
    eager_fallback: bool | None = None
    kv_format: str | None = None
    # Validity
    valid: Validity = "valid"
    reason: str | None = None
    errors: list[str] = Field(default_factory=list)
    timeouts: int = 0
    exit_reason: str | None = None  # RAM_FLOOR | OOM | SPEC_ASSERT | ENGINE_ERROR | TIMEOUT | CLIENT_ERROR
    flags: list[str] = Field(default_factory=list)
    quality: QualityResult = Field(default_factory=QualityResult)
    # Repeats (ladder.repeats): how many decode windows were measured on this
    # rung's context. When > 1 the REPEAT_METRICS fields above are medians and
    # repeat_stats has the per-repeat values. Quality checks run once per rung.
    repeats: int = 1
    repeat_stats: dict[str, RepeatStat] = Field(default_factory=dict)
    started_utc: str | None = None
    started_local: str | None = Field(
        default=None, validation_alias=AliasChoices("started_local", "started_et")
    )
    finished_utc: str | None = None
    finished_local: str | None = Field(
        default=None, validation_alias=AliasChoices("finished_local", "finished_et")
    )

    @property
    def comparable_key(self) -> tuple[Any, ...]:
        return (
            self.users,
            self.ctx_per_user,
            self.prompt_type,
            self.load_mode,
            self.phase,
        )


class RunManifest(BaseModel):
    run_id: str
    spec_name: str
    spec_path: str | None = None
    profile: str
    tool_version: str = __version__
    started_utc: str
    # Same instant in the display timezone (old files used "started_et").
    started_local: str = Field(validation_alias=AliasChoices("started_local", "started_et"))
    finished_utc: str | None = None
    finished_local: str | None = Field(
        default=None, validation_alias=AliasChoices("finished_local", "finished_et")
    )
    # IANA zone of the *_local stamps ("system-local" = the runner's own zone).
    display_timezone: str | None = None
    # dry_run: a --dry-run manifest (written under <out>/dry-run/); --resume ignores it.
    # refused: preflight or bench-hold refused the run before any rung ran.
    status: Literal["in_progress", "completed", "aborted", "refused", "dry_run"] = "in_progress"
    abort_reason: str | None = None
    # Rung ids a --dry-run would execute (never counted as completed).
    planned_rung_ids: list[str] = Field(default_factory=list)
    # Which site config file was in effect (or that none was).
    site_config: str | None = None
    # Spec expectations (env / launch_args / image) checked against docker inspect.
    launch_check: dict[str, Any] = Field(default_factory=dict)
    image: str | None = None
    image_digest: str | None = None
    server_source_commit: str | None = None
    hardware: HardwareSnapshot
    config: dict[str, Any] = Field(default_factory=dict)
    attention: AttentionSnapshot = Field(default_factory=AttentionSnapshot)
    kv_format: str | None = None
    graph: GraphSnapshot | None = None
    speculative: SpecDecodeSnapshot | None = None
    decode_method: DecodeMethod | None = None
    completed_rung_ids: list[str] = Field(default_factory=list)
    output_jsonl: str | None = None
    notes: list[str] = Field(default_factory=list)
    prompt_seed: int | None = None
    k: int | None = None
    fingerprint: dict[str, Any] = Field(default_factory=dict)
    boot: dict[str, Any] = Field(default_factory=dict)
    host: dict[str, Any] = Field(default_factory=dict)
    corpus_files: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    hold: dict[str, Any] = Field(default_factory=dict)
    first_greedy_divergence: dict[str, Any] | None = None
    # Server start command and settings captured read-only at preflight
    # (docker inspect / ps / vLLM boot log), secrets redacted. Also in launch.json.
    launch: dict[str, Any] = Field(default_factory=dict)
    # Context ladder actually used (``auto`` resolved against the server).
    max_model_len: int | None = None
    max_model_len_source: str | None = None  # "/v1/models", "boot_log" or None
    context_rungs: list[int] = Field(default_factory=list)
    context_note: str | None = None
    # Server KV cache capacity in tokens and where it came from.
    kv_capacity_tokens: int | None = None
    kv_capacity_source: str | None = None
    # Per "family|load_mode|users": {"users", "max_ok_ctx", "stopped_at", "reason", "detail", "rung_id"}.
    ctx_ceilings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Highest context each "family|load_mode|users" line measured successfully.
    ctx_reached: dict[str, int] = Field(default_factory=dict)
    # Server / GPU load seen just before the first rung (two /metrics scrapes).
    pre_run_load: dict[str, Any] = Field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.model_dump(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> RunManifest:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def write_jsonl_row(path: str | Path, row: RungResult | dict[str, Any]) -> None:
    payload = row.model_dump() if isinstance(row, RungResult) else row
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=False) + "\n")


def read_jsonl(path: str | Path) -> list[RungResult]:
    rows: list[RungResult] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(RungResult.model_validate(json.loads(line)))
    return rows
