"""YAML run-spec format."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PromptType = Literal["code_edits", "long_form", "multi_turn_chat"]
LoadMode = Literal["closed_all_decoding", "staggered", "open_loop"]
ProfileName = Literal["quick", "full", "custom"]
SpecMethod = Literal["mtp", "dflash2", "none"]
GraphMode = Literal["FULL_DECODE_ONLY", "FULL", "EAGER", "custom"]
AttentionRequested = Literal["qsa", "dense", "auto"]


class GraphSettings(BaseModel):
    # None: not declared, so the graph mode is recorded from the boot log but not checked.
    mode: GraphMode | None = None
    per_gpu: bool = False
    capture_sizes: list[int] = Field(default_factory=list)
    eager_fallback_when_unmatched: bool = True


class SpeculativeSettings(BaseModel):
    """Speculative decoding as booted. ``num_speculative_tokens`` is k.

    k is never assumed: MTP-6 is k=6, DFlash2 is k=7, MTP-3 is k=3. If a spec
    leaves it unset, the runner derives k from the counters (draft_tokens /
    drafts) and flags every row ``K_UNDECLARED``.
    """

    # None: not declared. k is then derived from the counters (K_UNDECLARED) if
    # the server reports speculative decoding at all.
    method: SpecMethod | None = None
    num_speculative_tokens: int | None = None
    dflash_k: int | None = None

    @model_validator(mode="after")
    def _k_consistent(self) -> SpeculativeSettings:
        k = self.num_speculative_tokens
        if k is not None and k < 1:
            raise ValueError("num_speculative_tokens (k) must be >= 1")
        if self.method == "none" and k not in (None, 0):
            raise ValueError("speculative.method is none but num_speculative_tokens is set")
        if self.dflash_k is not None and k is not None and self.dflash_k != k:
            raise ValueError(
                f"dflash_k={self.dflash_k} disagrees with num_speculative_tokens={k}"
            )
        return self

    @property
    def k(self) -> int | None:
        if self.method == "none":
            return None
        return self.num_speculative_tokens or self.dflash_k


class HardwareSpec(BaseModel):
    """Facts about the bench host. Put them in the site config's ``hardware:``
    block (a spec may override). Unset values are reported as unknown; the
    planner skips a check it has no numbers for and says so."""

    host: str = "unspecified"  # display name; site config `host` overrides
    gpu_model: str = "unspecified"
    gpu_count: int | None = None
    # Device memory per GPU, used by the planner's GPU feasibility check.
    gpu_mem_gib: float | None = None
    tensor_parallel: int | None = None
    # None: read MemTotal from the bench host (live preflight, or `bench plan`
    # when access.mode is local).
    host_ram_gib: float | None = None
    # Abort when MemAvailable on the bench host reaches this. A generic safety
    # default; raise or lower it for your host.
    mem_available_floor_gib: float = 2.0


class EndpointSpec(BaseModel):
    base_url: str = "http://127.0.0.1:8000/v1"
    # None: derived from base_url's origin (<scheme://host:port>/metrics, /health).
    metrics_url: str | None = None
    health_url: str | None = None
    # None: derived from base_url's origin (<scheme://host:port>/tokenize).
    tokenize_url: str | None = None
    # API keys are read from this environment variable only. Never put a key in the spec.
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "TODO-model-served-name"
    timeout_s: float = 600.0

    @model_validator(mode="after")
    def _derive_urls(self) -> EndpointSpec:
        from urllib.parse import urlsplit

        parts = urlsplit(self.base_url)
        origin = f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else self.base_url
        if not self.metrics_url:
            self.metrics_url = origin + "/metrics"
        if not self.health_url:
            self.health_url = origin + "/health"
        return self


class ServerSpec(BaseModel):
    """What the test container is EXPECTED to be running.

    The harness never applies these settings. You boot the test container
    yourself; at preflight the runner reads ``docker inspect`` and checks
    that ``image``, ``env`` and ``launch_args`` match. A mismatch flags every
    row NOT_AS_LABELED. Leave a field ``null`` to skip its check; a
    ``TODO``/``<...>`` placeholder stops a live run at preflight.
    """

    image: str | None = None  # e.g. the test container's image tag; null: not checked
    image_digest: str | None = None
    container_name: str = "vllm-bench-test"
    source_commit: str | None = None
    launch_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    kv_format: str | None = None  # None: not declared, not checked against the boot log
    graph: GraphSettings = Field(default_factory=GraphSettings)
    speculative: SpeculativeSettings = Field(default_factory=SpeculativeSettings)
    attention_requested: AttentionRequested = "auto"
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float | None = None
    # If false, the runner attaches to an already-running endpoint and will not start/stop a container.
    manage_container: bool = False


class AccessSpec(BaseModel):
    """How the runner reaches the bench host.

    mode=local runs on the bench host itself; mode=ssh drives it over SSH.
    Put the ssh target in the untracked site config (``ssh_host``) or
    ``$BENCH_HOST``, never in a committed spec. Never put host keys,
    passwords, or private IPs in a spec.
    """

    mode: Literal["local", "ssh"] = "local"
    ssh_host: str | None = None  # an ~/.ssh/config alias; default: site config, then $BENCH_HOST
    ssh_extra_args: list[str] = Field(default_factory=list)


class SafetySpec(BaseModel):
    # bench-hold CLI used to take a host-wide benchmark hold. None (default):
    # no hold is taken. Set it (usually in the site config) on a shared host.
    hold_command: str | None = None
    # None (default): "<spec name>-<run id>", unique per run, so two copies of
    # a spec never treat each other's hold as their own. A resumed run keeps
    # its run id and so its hold name.
    hold_name: str | None = None
    hold_minutes: int = 120
    # None (default): "xpu-llm-bench <spec name> run <run id>".
    hold_reason: str | None = None
    # Renew (re-acquire with the same --by) when less than this many minutes remain.
    hold_renew_below_min: float = 20.0
    # Optional operator quiet-hold file, not shown by `bench-hold status`. It is
    # read (never written) and an active quiet hold also blocks a run. ``~`` is
    # the runner's home on the bench host. Usually set in site config.
    quiet_hold_file: str | None = None
    never_use_xpu_smi: bool = True
    # Extra protected container names. Always unioned with the built-in neutral
    # names (safety.PROTECTED_CONTAINER_EXACT) and the site config.
    protected_containers: list[str] = Field(default_factory=list)
    allowed_containers: list[str] = Field(default_factory=list)
    # Extra production ports. Unioned with safety.DEFAULT_PRODUCTION_PORT and the
    # site config; endpoints and containers bound to them are refused outright.
    forbidden_ports: list[int] = Field(default_factory=list)
    # Containers the runner may start/stop must also carry one of these prefixes.
    test_container_prefixes: list[str] = Field(
        default_factory=lambda: ["vllm-bench-", "vllm-test-", "vllm-qsa-", "vllm-dflash"]
    )


class WarmupPolicy(BaseModel):
    tokens: int = 8
    requests: int = 1


class OpenLoopSettings(BaseModel):
    requests_per_second: float = 1.0
    duration_s: float | None = None  # if set, overrides output-token window as the stop condition


class StaggeredSettings(BaseModel):
    delay_s: float = 0.5


class PromptSource(BaseModel):
    types: list[PromptType] = Field(
        default_factory=lambda: ["code_edits", "long_form", "multi_turn_chat"]
    )
    # Fixed seed. Recorded on every row together with the per-session seed.
    # Never derived from a timestamp. Change it only when you intend a new corpus.
    seed: int = 20261111
    # realistic: English prose / real-looking Python / support-chat turns built
    # from fixed vocabularies. synthetic: the original syllable filler (kept for
    # back-compat; it inflates MTP acceptance and is not the headline workload).
    style: Literal["realistic", "synthetic", "mixed"] = "realistic"
    # Optional directory of real text per prompt type (<dir>/<prompt_type>/*.txt|*.py|*.md).
    # Files are read in sorted order; their sha256 goes in the manifest.
    corpus_dir: str | None = None
    # Approximate characters per token used before a server /tokenize calibration.
    chars_per_token_guess: float = 4.0
    # Plant one retrievable fact (needle) in every growth segment.
    plant_needles: bool = True


LadderShape = Literal["session1_then_join", "grid"]


class LadderSpec(BaseModel):
    # session1_then_join (the progressive ladder): boot once; session 1 grows
    # through every context rung in one conversation; then, for each
    # concurrency step, that many users join at join_ctx_tokens and grow
    # together rung by rung until the top rung or the user count's ceiling.
    # grid: the original scaffold order (all concurrencies at the first rung,
    # then grow everyone and quiesce down). It starts at the widest point
    # (every user at once), so it is refused unless allow_grid is also true.
    shape: LadderShape = "session1_then_join"
    allow_grid: bool = False
    join_ctx_tokens: int | None = None  # where joining users start; default: first context rung
    # Cap for multi-user rungs (None: they climb to the top rung like session 1).
    # The quick profile sets 8192 so 2/4/8 users stay short.
    join_max_context_tokens: int | None = None
    # Sampling for measured windows. Greedy keeps A/B workloads identical and
    # makes MTP on/off divergence checks meaningful.
    temperature: float = 0.0
    # Keep decoding to exactly output_tokens_per_rung (vLLM ignore_eos). EOS
    # positions are still detected (skip_special_tokens=false) and flagged.
    ignore_eos: bool = True
    # A measured window with fewer verify steps than this is SHORT_WINDOW.
    min_window_steps: int = 20
    # Poll /metrics this often during a window (running / waiting / KV).
    metrics_poll_s: float = 1.0
    # "auto" (default): 8K, 32K, 64K, 131K, then +auto_step_tokens (196K,
    # 262K, ...) up to the server's max_model_len (read from /v1/models or the
    # boot log at preflight) minus ctx_headroom_tokens, with that maximum as
    # the final rung. Or pin an explicit increasing list.
    context_rungs_tokens: list[int] | Literal["auto"] = "auto"
    # Optional cap for the ladder (auto or explicit), e.g. to stop below the
    # server's max_model_len. None: the server's max_model_len decides.
    max_context: int | None = None
    auto_base_tokens: list[int] = Field(default_factory=lambda: [8192, 32768, 65536, 131072])
    # After the base rungs: "add" auto_step_tokens per rung, "double", or
    # "auto" (add up to 262144 = 196K, 262K, then double: 524K, 1M, ...).
    auto_step_tokens: int = 65536
    auto_step_mode: Literal["add", "double", "auto"] = "auto"
    # Room left under max_model_len for the question and the output window.
    # None: output_tokens_per_rung + 1024.
    ctx_headroom_tokens: int | None = None
    # Keep climbing each user count until it stops fitting (KV capacity, the
    # server rejects the context, RAM floor, OOM); record CTX_CEILING and move
    # on to the next user count instead of aborting the run.
    stop_at_ceiling: bool = True
    concurrency_steps: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    # Original ladder steps retain per-concurrency session IDs under CLI selection.
    concurrency_session_steps: list[int] | None = Field(default=None, exclude=True)
    # Measure each rung's decode window this many times on the same context
    # (the context is not regrown between repeats). Rows report the median with
    # min/max (stdev from 3). Quality checks still run once per rung.
    repeats: int = Field(default=1, ge=1, le=50)
    load_modes: list[LoadMode] = Field(default_factory=lambda: ["closed_all_decoding"])
    open_loop: OpenLoopSettings = Field(default_factory=OpenLoopSettings)
    staggered: StaggeredSettings = Field(default_factory=StaggeredSettings)
    output_tokens_per_rung: int = 128
    warmup: WarmupPolicy = Field(default_factory=WarmupPolicy)
    grow_monotonically: bool = True
    boot_once: bool = True
    resume: bool = True
    # Target wall time used by `bench plan` and the quick profile, in minutes.
    target_minutes: float | None = None

    @field_validator("context_rungs_tokens")
    @classmethod
    def _rungs_increasing(cls, v: list[int] | str) -> list[int] | str:
        if v == "auto":
            return v
        if not v or any(n <= 0 for n in v):
            raise ValueError("context_rungs_tokens must be a non-empty list of positive ints")
        if v != sorted(v):
            raise ValueError("context_rungs_tokens must be strictly increasing")
        if len(set(v)) != len(v):
            raise ValueError("context_rungs_tokens must be unique")
        return v

    @model_validator(mode="after")
    def _grid_needs_opt_in(self) -> LadderSpec:
        if self.shape == "grid" and not self.allow_grid:
            raise ValueError(
                "ladder.shape=grid starts every user at once at the first rung and steps down; "
                "it is refused by default. Use the progressive session1_then_join ladder, or set "
                "ladder.allow_grid: true if you really want grid."
            )
        return self

    @field_validator("concurrency_steps")
    @classmethod
    def _conc_positive(cls, v: list[int]) -> list[int]:
        if not v or any(n <= 0 for n in v):
            raise ValueError("concurrency_steps must be a non-empty list of positive ints")
        if v != sorted(v):
            raise ValueError("concurrency_steps must be strictly increasing")
        return v


class PlannerAssumptions(BaseModel):
    """Planning-only rate and footprint assumptions. These are NOT measurements."""

    assumed_prefill_tok_s: float = 1000.0
    assumed_decode_tok_s_per_user: float = 20.0
    # ESTIMATE placeholder for int8-K / int4-V full-attn KV traffic. Confirm on the bench host.
    assumed_kv_bytes_per_token: int = 24576
    assumed_weight_gib: float = 18.0
    # Device-side runtime overhead (graphs, activations, allocator slack). ESTIMATE.
    assumed_fixed_overhead_gib: float = 4.0
    # Host-RAM growth per live context token once the server is booted
    # (pinned buffers, prefix-cache bookkeeping, CPU-side metadata). ESTIMATE:
    # measure it on your stack (MemAvailable drop / live tokens) and set it in
    # the spec or the site config's `planner:` block.
    host_bytes_per_token: int = 65536
    # Share of the KV footprint that also lives in host RAM (CPU offload /
    # swap space). 0 for device-only KV.
    host_kv_fraction: float = 0.0
    # Leave this much of GPU memory free (fail above, warn above warn_frac).
    gpu_fail_frac: float = 0.95
    gpu_warn_frac: float = 0.85
    # Host RAM: warn when headroom above the MemAvailable floor is below this.
    host_tight_gib: float = 2.0
    overhead_s_per_rung: float = 3.0
    quality_s_per_rung: float = 6.0


class MetricNameMap(BaseModel):
    """Prometheus names as observed on the vLLM 0.30 XPU fork. Confirm any TODO."""

    drafts: str = "vllm:spec_decode_num_drafts_total"
    draft_tokens: str = "vllm:spec_decode_num_draft_tokens_total"
    accepted_tokens: str = "vllm:spec_decode_num_accepted_tokens_total"
    accepted_per_pos: str = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
    preemptions: str = "vllm:num_preemptions_total"
    generation_tokens: str = "vllm:generation_tokens_total"
    prompt_tokens: str = "vllm:prompt_tokens_total"
    # TODO: confirm these names on the target build (they are the common vLLM names).
    prefix_hits: str = "vllm:prefix_cache_hits_total"
    prefix_queries: str = "vllm:prefix_cache_queries_total"
    # vLLM V1 renamed gpu_cache_usage_perc to kv_cache_usage_perc (a 0..1 fraction).
    # Both are tried; the first present wins.
    gpu_cache_usage: str = "vllm:kv_cache_usage_perc"
    gpu_cache_usage_aliases: list[str] = Field(
        default_factory=lambda: ["vllm:gpu_cache_usage_perc"]
    )
    running: str = "vllm:num_requests_running"
    waiting: str = "vllm:num_requests_waiting"
    queue_time_sum: str = "vllm:request_queue_time_seconds_sum"
    queue_time_count: str = "vllm:request_queue_time_seconds_count"


class LogPatterns(BaseModel):
    qsa_active: str = r"QSA_GRAPH width=(\d+)"
    qsa_skip: str = r"QSA_GRAPH skip starts=\(([^)]*)\) n=(\S+)"
    # Tokens above this have no QSA graph path and run dense.
    qsa_graph_max_tokens: int = 28
    qsa_wrap: str = r"QSA_WRAP\s+(\S+)"
    qsa_memory_plan: str = r"QSA_MEMORY_PLAN\s"
    dense_banner: str = r"Using Xe2 K8/V4 attention backend"
    spec_assert: str = (
        r"Expected spec_token == num_spec_decodes \* \(num_speculative_tokens \+ 1\)"
    )


class QualitySpec(BaseModel):
    """Lightweight quality checks run in the same pass, outside measured windows."""

    enabled: bool = True
    # 1) Logprob scoring of fixed reference answers. ``short`` scores against a
    # fixed ~probe_ctx_tokens context (cheap). ``ladder_ctx`` appends the
    # reference to the live session context; vLLM skips prefix-cache reads for
    # prompt_logprobs requests, so that costs a full re-prefill per rung.
    logprob_mode: Literal["off", "short", "ladder_ctx"] = "short"
    probe_ctx_tokens: int = 2048
    # Stored dense K8/V4 baseline: JSONL of {probe_id, mean_logprob, ...}.
    logprob_baseline: str | None = None
    logprob_tolerance_nats: float = 0.05
    # 2) Greedy MTP on vs off divergence. Reference = greedy outputs captured
    # on an MTP-off (or dense) boot with the same seed; JSONL keyed by rung probe.
    greedy_reference: str | None = None
    greedy_probe_tokens: int = 64
    # 3) Pass/fail tasks.
    needle: bool = True
    code_edit: bool = True
    code_edit_timeout_s: float = 10.0
    format_checks: bool = True
    repetition_ngram: int = 6
    repetition_max_ratio: float = 0.35


class VariantEndpoint(BaseModel):
    """Per-variant endpoint override: point each variant at its own booted container."""

    base_url: str | None = None
    metrics_url: str | None = None
    health_url: str | None = None
    tokenize_url: str | None = None


class VariantSpec(BaseModel):
    """One side of an A/B (or one-fix) comparison, overlaid on the parent spec.

    env / launch_args / image are expectations about the container you
    booted for this variant; they are verified at preflight, never applied.
    """

    name: str
    env: dict[str, str] = Field(default_factory=dict)
    launch_args: list[str] | None = None
    attention_requested: AttentionRequested | None = None
    graph: GraphSettings | None = None
    speculative: SpeculativeSettings | None = None
    kv_format: str | None = None
    container_name: str | None = None
    image: str | None = None
    endpoint: VariantEndpoint | None = None
    notes: str | None = None


class RunSpec(BaseModel):
    name: str
    description: str = ""
    profile: ProfileName = "full"
    kind: Literal["run", "comparison", "template"] = "run"
    endpoint: EndpointSpec = Field(default_factory=EndpointSpec)
    server: ServerSpec = Field(default_factory=ServerSpec)
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    access: AccessSpec = Field(default_factory=AccessSpec)
    safety: SafetySpec = Field(default_factory=SafetySpec)
    prompts: PromptSource = Field(default_factory=PromptSource)
    ladder: LadderSpec = Field(default_factory=LadderSpec)
    planner: PlannerAssumptions = Field(default_factory=PlannerAssumptions)
    metrics: MetricNameMap = Field(default_factory=MetricNameMap)
    log_patterns: LogPatterns = Field(default_factory=LogPatterns)
    quality: QualitySpec = Field(default_factory=QualitySpec)
    variants: list[VariantSpec] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _allowed_containers(self) -> RunSpec:
        allowed = list(self.safety.allowed_containers)
        if self.server.container_name and self.server.container_name not in allowed:
            allowed.append(self.server.container_name)
        for var in self.variants:
            if var.container_name and var.container_name not in allowed:
                allowed.append(var.container_name)
        self.safety.allowed_containers = allowed
        return self


# Built-in profile overlays. Spec files may still override any field.
QUICK_PROFILE: dict[str, Any] = {
    "profile": "quick",
    "ladder": {
        "shape": "session1_then_join",
        "context_rungs_tokens": [8192, 32768],
        "concurrency_steps": [1, 2, 4, 8],
        "load_modes": ["closed_all_decoding"],
        # MTP-6 delivers ~3-5 tokens per verify step, so 96 tokens is ~20-30
        # steps: the shortest window that still clears min_window_steps.
        "output_tokens_per_rung": 96,
        "warmup": {"tokens": 8, "requests": 1},
        # Quick: 2/4/8 users measured at 8K only (session 1 still grows to 32K);
        # 8 x 32K would triple the prefill time for a smoke run.
        "join_max_context_tokens": 8192,
        "target_minutes": 12.0,
    },
}

FULL_PROFILE: dict[str, Any] = {
    "profile": "full",
    "ladder": {
        "shape": "session1_then_join",
        "context_rungs_tokens": "auto",
        "concurrency_steps": [1, 2, 4, 8],
        "load_modes": ["closed_all_decoding", "staggered"],
        "output_tokens_per_rung": 256,
        "warmup": {"tokens": 16, "requests": 1},
        "target_minutes": None,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def apply_profile(data: dict[str, Any], profile: ProfileName | None = None) -> dict[str, Any]:
    forced = profile is not None
    chosen: ProfileName = profile or data.get("profile") or "full"
    if chosen == "custom":
        out = dict(data)
        out["profile"] = "custom"
        return out
    defaults = QUICK_PROFILE if chosen == "quick" else FULL_PROFILE
    if forced:
        # CLI profile imposes the standard ladder shape (rungs, concurrency, load modes).
        merged = _deep_merge(data, defaults)
        merged["profile"] = chosen
        return merged
    # Profile taken from the file: defaults fill gaps, explicit file values win.
    return _deep_merge(defaults, {**data, "profile": chosen})


def load_spec(path: str | Path, profile: ProfileName | None = None) -> RunSpec:
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if data is None:
        raise ValueError(f"Empty spec: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Spec must be a mapping: {path}")
    return RunSpec.model_validate(apply_site(apply_profile(data, profile)))


def apply_site(data: dict) -> dict:
    """Fill host-specific values the spec leaves unset from the untracked site
    config (then ``$BENCH_HOST``). Explicit spec values win."""
    import os

    from .site import load_site

    site = load_site()
    hw = data.setdefault("hardware", {}) or {}
    data["hardware"] = hw
    if "host" not in hw and (site.host or os.environ.get("BENCH_HOST")):
        hw["host"] = site.host or os.environ["BENCH_HOST"]
    acc = data.setdefault("access", {}) or {}
    data["access"] = acc
    if not acc.get("ssh_host") and (site.ssh_host or os.environ.get("BENCH_HOST")):
        acc["ssh_host"] = site.ssh_host or os.environ["BENCH_HOST"]
    saf = data.setdefault("safety", {}) or {}
    data["safety"] = saf
    for key, val in (site.hardware or {}).items():
        hw.setdefault(key, val)
    plan = data.setdefault("planner", {}) or {}
    data["planner"] = plan
    for key, val in (site.planner or {}).items():
        plan.setdefault(key, val)
    if "hold_command" not in saf and site.hold_command:
        saf["hold_command"] = site.hold_command
    if "quiet_hold_file" not in saf and site.quiet_hold_file:
        saf["quiet_hold_file"] = site.quiet_hold_file
    return data


def apply_variant(spec: RunSpec, variant: VariantSpec) -> RunSpec:
    data = spec.model_dump()
    data["name"] = f"{spec.name}__{variant.name}"
    env = dict(data["server"].get("env") or {})
    env.update(variant.env)
    data["server"]["env"] = env
    if variant.launch_args is not None:
        data["server"]["launch_args"] = list(variant.launch_args)
    if variant.attention_requested is not None:
        data["server"]["attention_requested"] = variant.attention_requested
    if variant.graph is not None:
        data["server"]["graph"] = variant.graph.model_dump()
    if variant.speculative is not None:
        data["server"]["speculative"] = variant.speculative.model_dump()
    if variant.kv_format is not None:
        data["server"]["kv_format"] = variant.kv_format
    if variant.container_name is not None:
        data["server"]["container_name"] = variant.container_name
    if variant.image is not None:
        data["server"]["image"] = variant.image
    if variant.endpoint is not None:
        given = {k: v for k, v in variant.endpoint.model_dump().items() if v is not None}
        if "base_url" in given:
            # Other URLs follow the variant's own base_url unless it names them.
            for key in ("metrics_url", "health_url", "tokenize_url"):
                data["endpoint"][key] = None
        data["endpoint"].update(given)
    data["variants"] = []
    return RunSpec.model_validate(data)


_PLACEHOLDER_RE = re.compile(r"<[^<>\s][^<>]*>|\bTODO\b|TODO-", re.IGNORECASE)
# Free-text fields that may legitimately mention TODO or <...>.
_PLACEHOLDER_SKIP = {"notes", "description", "log_patterns", "metrics", "planner"}


def is_placeholder(value: object) -> bool:
    return isinstance(value, str) and bool(_PLACEHOLDER_RE.search(value))


def spec_placeholders(spec: RunSpec) -> list[str]:
    """Dotted paths of spec values that are still TODO / <...> placeholders."""
    found: list[str] = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if not path and key in _PLACEHOLDER_SKIP:
                    continue
                if key in {"notes", "description"}:
                    continue
                walk(val, f"{path}.{key}" if path else str(key))
        elif isinstance(obj, list):
            for i, val in enumerate(obj):
                walk(val, f"{path}[{i}]")
        elif is_placeholder(obj):
            found.append(f"{path} = {obj!r}")

    walk(spec.model_dump(), "")
    return found


# ---------------------------------------------------------------------------
# Context ladder resolution
# ---------------------------------------------------------------------------
AUTO_UNKNOWN_TOP = 131072  # what `auto` shows before the server's max_model_len is known


def ctx_headroom(ladder: LadderSpec) -> int:
    if ladder.ctx_headroom_tokens is not None:
        return ladder.ctx_headroom_tokens
    return ladder.output_tokens_per_rung + 1024


def resolve_contexts(ladder: LadderSpec, max_model_len: int | None = None) -> tuple[list[int], str]:
    """Context rungs for this run and a one-line note on where they came from.

    The limit is ``min(max_model_len - headroom, ladder.max_context)`` over the
    values that are known. Auto rungs: ``auto_base_tokens``, then
    ``auto_step_tokens`` steps, every rung below the limit, plus the limit
    itself (rounded down to 1K) as the final rung. Explicit lists are kept
    but rungs above the limit are dropped.
    """
    limits = []
    if max_model_len:
        limits.append(max_model_len - ctx_headroom(ladder))
    if ladder.max_context:
        limits.append(ladder.max_context)
    limit = min(limits) if limits else None
    if limit is not None:
        limit = max(limit, 1)
        if limit >= 16384:
            limit = limit // 1024 * 1024  # a round final rung (260096, not 260992)
    src = (f"max_model_len={max_model_len}" if max_model_len else "") + (
        f"{', ' if max_model_len else ''}max_context={ladder.max_context}" if ladder.max_context else "")
    if ladder.context_rungs_tokens != "auto":
        ctxs = list(ladder.context_rungs_tokens)
        if limit is None:
            return ctxs, "explicit context rungs"
        kept = [c for c in ctxs if c <= limit] or [min(ctxs[0], limit)]
        dropped = [c for c in ctxs if c > limit]
        note = "explicit context rungs" + (f"; dropped {dropped} above the limit {limit} ({src})" if dropped else "")
        return kept, note
    base = sorted(set(ladder.auto_base_tokens))
    if limit is None:
        return [c for c in base if c <= AUTO_UNKNOWN_TOP] or base[:1], (
            f"auto: server max_model_len not known yet, showing rungs up to {AUTO_UNKNOWN_TOP // 1000}K; "
            "a live run extends them to max_model_len at preflight (or set ladder.max_context / --max-context)"
        )
    seq = list(base)
    nxt = seq[-1]
    while nxt < limit:
        double = ladder.auto_step_mode == "double" or (ladder.auto_step_mode == "auto" and nxt >= 262144)
        nxt = nxt * 2 if double else nxt + ladder.auto_step_tokens
        seq.append(nxt)
    gap = min(4096, limit // 32)  # drop a step rung that sits right under the final one
    ctxs = [c for c in seq if c <= limit - gap]
    ctxs.append(limit)
    return ctxs, f"auto: up to {limit} ({src} minus {ctx_headroom(ladder)} headroom)" if max_model_len else (
        f"auto: up to {limit} ({src})")


def with_contexts(spec: RunSpec, ctxs: list[int]) -> RunSpec:
    """Copy of ``spec`` with an explicit, resolved context list."""
    data = spec.model_dump()
    data["ladder"]["context_rungs_tokens"] = list(ctxs)
    return RunSpec.model_validate(data)


def ladder_contexts(spec: RunSpec) -> list[int]:
    """Resolved context rungs using only what the spec knows (no server)."""
    return resolve_contexts(spec.ladder, None)[0]
