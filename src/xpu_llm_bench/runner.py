"""Ladder runner: boot once, grow sessions, measure fixed windows, resume after abort.

Safety, in order, before any request:
1. Endpoint ports: refuse the protected production port(s) (spec
   ``safety.forbidden_ports`` plus the untracked site config).
2. Container: refuse protected production containers and services (built-in
   neutral names, spec, and site config) and CI runners; refuse a container
   that publishes a protected port.
3. bench-hold: ``status`` first; refuse if anyone else holds it (acquire
   overwrites silently); refuse an active operator quiet hold; acquire; confirm.
4. MemAvailable above the floor (default 2 GiB); sampled every 0.5 s.

The runner never starts, stops, or restarts any container.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xpu_llm_bench import bootlog, hostinfo, launch, procmem
from xpu_llm_bench.client import OpenAICompatClient, StreamResult
from xpu_llm_bench.container import ContainerAdapter
from xpu_llm_bench.fingerprint import (
    Fingerprint,
    collect_fingerprint,
    hash_fingerprint_mounts,
    verify_expected,
)
from xpu_llm_bench.gpu_mem import collect_gpu_mem
from xpu_llm_bench.hold import BenchHold, quiet_hold_active
from xpu_llm_bench.hostexec import HostExec
from xpu_llm_bench.metrics import (
    ChunkStats,
    histogram_mean_ms,
    kv_usage_pct,
    metric_value,
    mtp_window,
    parse_prometheus,
    per_pos_sum,
    percentile,
    spec_counter_check,
    steady_window,
    stream_stats,
    system_window,
    true_hit_rate,
)
from xpu_llm_bench.path_classify import classify_logs
from xpu_llm_bench.planner import GRID_WARNING, feasibility, make_plan
from xpu_llm_bench.prompts import corpus, load_corpus_dir, needle_for, session_seed
from xpu_llm_bench.quality import (
    code_task_for,
    cut_at_eos,
    extract_code,
    first_divergence,
    format_check,
    load_keyed_jsonl,
    logprob_probes,
    mean_answer_logprob,
    needle_pass,
    run_code_test,
)
from xpu_llm_bench.ram import LocalMemInfo, RamFloorError, RamSource, RamWatcher, SshMemInfo
from xpu_llm_bench.results import (
    REPEAT_METRICS,
    AttentionSnapshot,
    DecodeMethod,
    GraphSnapshot,
    HardwareSnapshot,
    QualityResult,
    RungResult,
    RunManifest,
    SpecDecodeSnapshot,
    local_tz_name,
    read_jsonl,
    repeat_stat,
    stamp_pair,
    write_jsonl_row,
)
from xpu_llm_bench.safety import (
    SafetyError,
    assert_container_allowed,
    assert_endpoint_allowed,
    assert_test_container_name,
    is_protected_container,
    site_forbidden_ports,
)
from xpu_llm_bench.schedule import Rung, build_schedule, ceiling_key
from xpu_llm_bench.site import describe_site, load_site
from xpu_llm_bench.spec import (
    RunSpec,
    apply_variant,
    is_placeholder,
    resolve_contexts,
    spec_placeholders,
    with_contexts,
)

ABORTING_EXITS = {"RAM_FLOOR", "OOM", "SPEC_ASSERT", "ENGINE_ERROR"}
DEFAULT_LOAD_PROBE_S = 2.0  # pre-run load check: seconds between the two /metrics scrapes

# A failed rung whose errors read like this hit the context ceiling of its
# user count (the server refused the length or ran out of KV), not a crash.
_CTX_REJECT_RE = re.compile(
    r"maximum context length|max_model_len|context length|prompt is too long|too many tokens|"
    r"longer than the maximum|exceeds the max",
    re.I,
)
_KV_FULL_RE = re.compile(
    r"KV cache|num_gpu_blocks|not enough (?:free )?blocks|insufficient (?:KV|cache)|larger than the available",
    re.I,
)


def ctx_fill_pct(ctx_tokens: list[int], target: int) -> float | None:
    """Mean filled prompt tokens as a percentage of the rung's target."""
    vals = [c for c in ctx_tokens if c]
    if not vals or not target:
        return None
    return round(100.0 * (sum(vals) / len(vals)) / target, 2)


def ceiling_reason(row: RungResult) -> str | None:
    """Why a failed rung is this user count's context ceiling (None: a real failure)."""
    if row.valid != "invalid":
        return None
    if row.exit_reason == "RAM_FLOOR":
        return "ram_floor"
    if row.exit_reason == "OOM":
        return "oom"
    text = " ".join([row.reason or "", *(row.errors or [])])
    if _KV_FULL_RE.search(text):
        return "kv_capacity"
    if _CTX_REJECT_RE.search(text):
        return "server_rejected_context"
    return None


def kv_capacity_from_metrics(text: str) -> int | None:
    """KV capacity in tokens from vllm:cache_config_info (num_gpu_blocks x block_size)."""
    for line in text.splitlines():
        if not line.startswith("vllm:cache_config_info{"):
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', line))
        try:
            return int(labels["num_gpu_blocks"]) * int(labels["block_size"])
        except (KeyError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Sessions: one growing conversation each, persisted for --resume
# ---------------------------------------------------------------------------


class Session:
    """A conversation that only grows.

    Growth appends a new user turn (fresh corpus text + one planted needle +
    the ask). After each measured window the model's own reply (cut at EOS)
    is appended as the assistant turn. Earlier turns are never edited, so
    the rendered prefix is stable and the prefix cache is reused.
    """

    def __init__(
        self,
        sid: int,
        prompt_type: str,
        seed: int,
        chars_per_token: float,
        style: str = "realistic",
        file_text: str | None = None,
        plant_needles: bool = True,
    ):
        self.sid = sid
        self.prompt_type = prompt_type
        self.seed = seed
        self.cpt = chars_per_token
        self.plant_needles = plant_needles
        self.corpus = corpus(prompt_type, seed=seed, chars_per_token=chars_per_token, style=style, file_text=file_text)  # type: ignore[arg-type]
        self.messages: list[dict[str, str]] = []
        self.segments = 0
        self.needles: list[dict[str, Any]] = []
        self.ctx_tokens = 0  # last server-reported prompt_tokens
        # Compact, regenerable record of each turn (see to_dict).
        self.turns: list[dict[str, Any]] = []
        # back-compat with the scaffold API
        self.prefix = ""

    # -- size bookkeeping ---------------------------------------------------
    def chars(self) -> int:
        return sum(len(m["content"]) for m in self.messages)

    def est_tokens(self) -> int:
        if self.ctx_tokens:
            return self.ctx_tokens
        return int(self.chars() / max(self.cpt, 0.5))

    def calibrate(self, prompt_tokens: int | None) -> None:
        if prompt_tokens:
            self.ctx_tokens = int(prompt_tokens)
            self.cpt = max(self.chars() / max(prompt_tokens, 1), 0.5)

    # -- growth -------------------------------------------------------------
    def grow_to(self, target_tokens: int) -> list[dict[str, str]]:
        need = target_tokens - self.est_tokens() - 64  # template overhead per turn
        if need <= 0 and self.messages and self.messages[-1]["role"] == "user":
            return self.messages
        n_chars = max(int(max(need, 32) * self.cpt), 64)
        if self.plant_needles:
            _fact, question, answer = needle_for(self.seed, self.segments)
            self.needles.append({"segment": self.segments, "question": question, "answer": answer})
        opener = not self.messages
        content = self._user_text(n_chars, self.segments, opener)
        self.messages.append({"role": "user", "content": content})
        self.turns.append(
            {
                "role": "user",
                "segment": self.segments,
                "n_chars": n_chars,
                "opener": opener,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
        self.segments += 1
        return self.messages

    def _user_text(self, n_chars: int, segment: int, opener: bool) -> str:
        needle_text = needle_for(self.seed, segment)[0] if self.plant_needles else None
        body = self.corpus.segment(n_chars, segment, needle=needle_text)
        return (self.corpus.opener() if opener else "") + body + self.corpus.ask()

    def resize_last_user(self, n_chars: int) -> None:
        """Re-cut the newest (not yet sent) user turn to ``n_chars`` of corpus
        text. Deterministic, so --resume regenerates the same turn."""
        if not self.messages or self.messages[-1]["role"] != "user" or not self.turns:
            return
        turn = self.turns[-1]
        n_chars = max(int(n_chars), 64)
        content = self._user_text(n_chars, int(turn["segment"]), bool(turn["opener"]))
        self.messages[-1] = {"role": "user", "content": content}
        turn["n_chars"] = n_chars
        turn["sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()

    def grow_message(self, target_tokens: int) -> list[dict[str, str]]:  # scaffold name
        return self.grow_to(target_tokens)

    def append_reply(self, text: str) -> None:
        reply, _ = cut_at_eos(text)
        if self.messages and self.messages[-1]["role"] == "user":
            content = reply.strip() or "(no reply)"
            self.messages.append({"role": "assistant", "content": content})
            self.turns.append({"role": "assistant", "content": content})

    def to_dict(self) -> dict[str, Any]:
        """Compact state: user turns are stored as (segment, n_chars, opener,
        sha256) and regenerated from the seeded corpus on resume; only the
        model's own replies are stored verbatim."""
        return {
            "format": 2,
            "sid": self.sid,
            "prompt_type": self.prompt_type,
            "seed": self.seed,
            "cpt": self.cpt,
            "turns": self.turns,
            "segments": self.segments,
            "needles": self.needles,
            "ctx_tokens": self.ctx_tokens,
        }

    def load_state(self, d: dict[str, Any]) -> None:
        self.cpt = float(d.get("cpt") or self.cpt)
        self.segments = int(d.get("segments") or 0)
        self.needles = list(d.get("needles") or [])
        self.ctx_tokens = int(d.get("ctx_tokens") or 0)
        if "turns" not in d:  # format 1: full messages stored verbatim
            self.messages = list(d.get("messages") or [])
            self.turns = []
            return
        self.turns = list(d["turns"])
        self.messages = []
        for turn in self.turns:
            if turn["role"] == "assistant":
                self.messages.append({"role": "assistant", "content": turn["content"]})
                continue
            text = self._user_text(int(turn["n_chars"]), int(turn["segment"]), bool(turn["opener"]))
            if turn.get("sha256") and hashlib.sha256(text.encode("utf-8")).hexdigest() != turn["sha256"]:
                raise ValueError(
                    f"session {self.sid}: regenerated turn {turn['segment']} does not match the saved hash "
                    "(prompt corpus, seed or code changed); resume refused, start a fresh run"
                )
            self.messages.append({"role": "user", "content": text})


# ---------------------------------------------------------------------------
# Window sampler: /metrics and /proc during a measured window
# ---------------------------------------------------------------------------


METRICS_FETCH_ATTEMPTS = 4
METRICS_RETRY_BASE_S = 1.0
QUIESCE_POLL_S = 0.25  # /metrics poll interval while waiting for the server to go idle


@dataclass
class WindowSamples:
    running: list[float] = field(default_factory=list)
    waiting: list[float] = field(default_factory=list)
    kv_pct: list[float] = field(default_factory=list)
    rss: dict[str, int] = field(default_factory=dict)
    other_stacks: set[str] = field(default_factory=set)


class WindowSampler:
    def __init__(self, fetch: Callable[[], str], names, interval_s: float, proc_fn: Callable[[], procmem.ProcSnapshot] | None):
        self.fetch = fetch
        self.names = names
        self.interval_s = max(interval_s, 0.05)
        self.proc_fn = proc_fn
        self.samples = WindowSamples()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _proc(self) -> None:
        if not self.proc_fn:
            return
        try:
            snap = self.proc_fn()
        except Exception:  # noqa: BLE001
            return
        self.samples.rss = procmem.peak_merge(self.samples.rss, snap.ours)
        self.samples.other_stacks.update(snap.other_stacks)

    def _loop(self) -> None:
        n = 0
        while not self._stop.is_set():
            try:
                snap = parse_prometheus(self.fetch())
                for key, dst in ((self.names.running, self.samples.running), (self.names.waiting, self.samples.waiting)):
                    v = metric_value(snap, key)
                    if v is not None:
                        dst.append(v)
                kv = kv_usage_pct(snap, self.names)
                if kv is not None:
                    self.samples.kv_pct.append(kv)
            except Exception:  # noqa: BLE001
                pass
            if n % 5 == 0:
                self._proc()
            n += 1
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._proc()
        self._thread = threading.Thread(target=self._loop, name="window-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> WindowSamples:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._proc()
        return self.samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_id(spec_name: str) -> str:
    utc, _ = stamp_pair()
    short = uuid.uuid4().hex[:8]
    stamp = utc.replace(":", "").replace("-", "").split(".")[0]
    return f"{spec_name}-{stamp}-{short}"


def _mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _sum(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) if vals else None


def apply_repeats(row: RungResult, extra: list[dict[str, float | None] | None]) -> None:
    """Fold repeats 2..N into ``row`` (which holds repeat 1).

    Each REPEAT_METRICS field becomes the median across repeats and
    ``repeat_stats`` keeps the per-repeat values, min, max and (n >= 3) stdev.
    A failed repeat contributes None. Prefill is measured once per rung (the
    context is not regrown between repeats), so its stat holds that single
    value (n=1, no range).
    """
    first = {m: getattr(row, m) for m in REPEAT_METRICS}
    per = [first] + list(extra)
    row.repeats = len(per)
    stats = {}
    for m in REPEAT_METRICS:
        vals = [first[m]] if m == "prefill_tok_s" else [(p or {}).get(m) for p in per]
        st = repeat_stat(vals)
        stats[m] = st
        if st.median is not None:
            setattr(row, m, st.median)
    row.repeat_stats = stats
    users = max(row.users, 1)
    if row.decode_tok_s_per_user is not None:
        row.decode_tok_s_total = row.decode_tok_s_per_user * users
    if row.steps_s_per_user is not None:
        row.steps_s_total = row.steps_s_per_user * users
    if row.decode_tok_s_per_user_steady is not None:
        row.decode_tok_s_total_steady = row.decode_tok_s_per_user_steady * users


def _host_exec(spec: RunSpec) -> HostExec:
    return HostExec(spec.access.mode, spec.access.ssh_host, spec.access.ssh_extra_args)


def _ram_source(spec: RunSpec) -> RamSource:
    if spec.access.mode == "ssh":
        if not spec.access.ssh_host:
            raise SafetyError("access.mode is ssh but ssh_host is not set (TODO)")
        return SshMemInfo(spec.access.ssh_host, spec.access.ssh_extra_args)
    return LocalMemInfo()


def safety_preflight(spec: RunSpec) -> None:
    """Checks that need no network: ports and container names. Runs in dry-run too."""
    ep = spec.endpoint
    assert_endpoint_allowed(
        [ep.base_url, ep.metrics_url, ep.health_url, ep.tokenize_url],
        site_forbidden_ports(spec.safety.forbidden_ports),
    )
    name = spec.server.container_name
    if name:
        assert_container_allowed(name)
        if is_protected_container(name, spec.safety.protected_containers):
            raise SafetyError(f"Refusing protected container {name!r}")
        if spec.server.manage_container:
            assert_test_container_name(name, spec.safety.test_container_prefixes)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class RunRefused(SafetyError):
    """Preflight or bench-hold refused a run. The manifest is marked refused."""

    def __init__(self, message: str, manifest: RunManifest | None = None):
        super().__init__(message)
        self.manifest = manifest


class Runner:
    def __init__(
        self,
        spec: RunSpec,
        output_dir: str | Path,
        dry_run: bool = False,
        resume: bool = False,
        client: OpenAICompatClient | None = None,
        hold: BenchHold | None = None,
        containers: ContainerAdapter | None = None,
        ram_source: RamSource | None = None,
        sleep: Callable[[float], None] = time.sleep,
        fetch_metrics: Callable[[], str] | None = None,
        fetch_logs: Callable[[], str] | None = None,
        collect_mem: Callable[..., str | None] | None = None,
        host_exec: HostExec | None = None,
        fingerprint_fn: Callable[[], Fingerprint | None] | None = None,
        proc_fn: Callable[[], procmem.ProcSnapshot] | None = None,
        host_fn: Callable[[], dict] | None = None,
        ram_wait_min: float = 0.0,
        no_site_config: bool = False,
        require_container: bool = True,
        html: bool = True,
        ps_fn: Callable[[], str] | None = None,
        assume_yes: bool = False,
        confirm_fn: Callable[[str], bool] | None = None,
        load_probe_s: float | None = None,
    ):
        self.spec = spec
        # Pre-run load check: warn and ask once when the GPU or the server is
        # already busy (--yes, or a non-interactive stdin, answers yes).
        self.assume_yes = assume_yes
        self._confirm_fn = confirm_fn
        self.load_probe_s = DEFAULT_LOAD_PROBE_S if load_probe_s is None else load_probe_s
        self._pre_load_flag = False
        # Dry runs never share a directory (or a manifest) with a real run.
        self.output_dir = Path(output_dir) / "dry-run" if dry_run else Path(output_dir)
        self.dry_run = dry_run
        self.resume = resume and not dry_run
        self.no_site_config = no_site_config
        self.require_container = require_container
        self.launch_mismatches: list[str] = []
        self._run_id: str | None = None
        self.preflight_warnings: list[str] = []
        self.client = client or OpenAICompatClient(
            spec.endpoint.base_url,
            spec.endpoint.model,
            api_key_env=spec.endpoint.api_key_env,
            timeout_s=spec.endpoint.timeout_s,
        )
        self.exec = host_exec or _host_exec(spec)
        if hold is None and spec.safety.hold_command:
            cmd = spec.safety.hold_command
            if spec.access.mode == "ssh" and cmd.startswith("~/"):
                cmd = cmd[2:]  # ssh runs in $HOME; avoid expanding ~ on this machine
            hold = BenchHold(
                cmd,
                # Placeholder until run(): replaced by "<run id>" unless the spec names it.
                name=spec.safety.hold_name or f"xpu-llm-bench-{spec.name}",
                dry_run=dry_run,
                runner=(lambda c, **kw: __import__("subprocess").run(self.exec.argv(c), **kw))
                if spec.access.mode == "ssh"
                else None,
            )
        # None: no bench-hold coordination configured (safety.hold_command / site config).
        self.hold: BenchHold | None = hold
        self._hold_auto_name = hold is not None and spec.safety.hold_name is None and self._owns_hold(hold)
        self.containers = containers or ContainerAdapter(allowed=spec.safety.allowed_containers, dry_run=dry_run)
        self.ram_source = ram_source or _ram_source(spec)
        self.sleep = sleep
        self.fetch_metrics = fetch_metrics
        self.fetch_logs = fetch_logs
        self.collect_mem = collect_mem or collect_gpu_mem
        self._fingerprint_fn = fingerprint_fn
        self._proc_fn = proc_fn
        self._host_fn = host_fn
        self.html = html
        self._ps_fn = ps_fn
        self.ram_wait_min = ram_wait_min
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.k_declared = spec.server.speculative.k
        self.fp: Fingerprint | None = None
        self.boot: bootlog.BootEvidence | None = None
        self.greedy_ref = load_keyed_jsonl(spec.quality.greedy_reference)
        self.logprob_base = load_keyed_jsonl(spec.quality.logprob_baseline)
        self.diverged: dict[tuple[str, str, int], bool] = {}
        self.corpus_text: dict[str, str | None] = {}
        self.corpus_manifest: dict[str, list[dict]] = {}

    def _owns_hold(self, hold: BenchHold) -> bool:
        return hold.name == f"xpu-llm-bench-{self.spec.name}"

    @property
    def hold_reason(self) -> str:
        return self.spec.safety.hold_reason or f"xpu-llm-bench {self.spec.name} run {self._run_id or '?'}"

    # -- I/O ------------------------------------------------------------------
    def _metrics_text(self) -> str:
        if self.fetch_metrics:
            return self.fetch_metrics()
        # A busy server (e.g. several long prefills at once) can answer /metrics late; one slow or
        # dropped scrape should not abort a whole run, so retry with backoff before giving up.
        error = ""
        for attempt in range(METRICS_FETCH_ATTEMPTS):
            status, body = self.client.get_text(self.spec.endpoint.metrics_url)
            if status == 200:
                return body
            error = f"/metrics returned {status}" + (f" ({body[:200]})" if status == 0 else "")
            if attempt + 1 < METRICS_FETCH_ATTEMPTS:
                time.sleep(METRICS_RETRY_BASE_S * 2**attempt)
        raise RuntimeError(f"{error} after {METRICS_FETCH_ATTEMPTS} attempts")

    def _logs_text(self, tail: int = 4000) -> str:
        if self.fetch_logs:
            return self.fetch_logs()
        name = self.spec.server.container_name
        if not name:
            return ""
        assert_container_allowed(name)
        res = self.exec.run(["docker", "logs", "--timestamps", "--tail", str(tail), name], timeout=60)
        return res.stdout + res.stderr

    def _fingerprint(self, hash_mounts: bool = True) -> Fingerprint | None:
        """docker inspect of the test container. ``hash_mounts=False`` skips the
        mount hashing (validation and the per-rung reboot check need only state,
        ports and StartedAt)."""
        if self._fingerprint_fn:
            return self._fingerprint_fn()
        name = self.spec.server.container_name
        if not name:
            return None
        try:
            return collect_fingerprint(self.exec, name, hash_mounts=hash_mounts)
        except Exception as exc:  # noqa: BLE001
            fp = Fingerprint(name, None, None, None, [], None)
            fp.problems.append(f"docker inspect failed: {exc!r}"[:300])
            return fp

    def _procs(self) -> procmem.ProcSnapshot:
        if self._proc_fn:
            return self._proc_fn()
        cid = self.fp.container_id if self.fp else None
        return procmem.collect(self.exec, cid)

    def _host(self) -> dict:
        if self._host_fn:
            return self._host_fn()
        try:
            return hostinfo.collect(self.exec)
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}

    def _runlog(self, line: str) -> None:
        """Append one timestamped line to run.log (the launch header is at its top)."""
        if self.dry_run:
            return
        _utc, local = stamp_pair()
        with (self.output_dir / "run.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{local}  {line}\n")

    def _server_argv(self) -> list[str] | None:
        """Command line of a non-container vLLM server (read-only ``ps``)."""
        from urllib.parse import urlsplit

        try:
            text = self._ps_fn() if self._ps_fn else self.exec.run(["ps", "-eo", "pid=,args="], timeout=15).stdout
        except Exception:  # noqa: BLE001
            return None
        return launch.find_server_process(text or "", urlsplit(self.spec.endpoint.base_url).port)

    def _capture_launch(self, manifest: RunManifest, log_text: str | None) -> None:
        """Record how the server was started: launch.json, run.log header, manifest.launch.

        Read-only: docker inspect (already done for the fingerprint), ps, docker logs.
        Nothing is started, stopped or restarted.
        """
        fp = self.fp
        rec = launch.capture(
            process_argv=None if (fp and fp.launch) else self._server_argv(),
            boot_log=log_text,
            endpoint=self.spec.endpoint.base_url,
            problems=list(fp.problems) if fp else [],
        )
        if fp and fp.launch:
            head = {k: rec[k] for k in ("captured_utc", "captured_local", "display_timezone", "endpoint")}
            tail = {k: rec[k] for k in ("vllm_boot_config", "problems", "read_only")}
            rec = {**head, **fp.launch, **tail}
        manifest.launch = rec
        (self.output_dir / "launch.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        log = self.output_dir / "run.log"
        mode = "a" if (self.resume and log.exists()) else "w"
        with log.open(mode, encoding="utf-8") as fh:
            if mode == "a":
                fh.write("\n--- RESUME: launch re-captured ---\n")
            fh.write(launch.header_text(rec, f"run {manifest.run_id} ({manifest.spec_name})"))

    def _write_html(self, manifest: RunManifest) -> None:
        from xpu_llm_bench.html_report import render_html

        jsonl = self.output_dir / "results.jsonl"
        try:
            rows = read_jsonl(jsonl) if jsonl.exists() else []
            (self.output_dir / "report.html").write_text(render_html(rows, manifest), encoding="utf-8")
            self._runlog(f"wrote {self.output_dir / 'report.html'}")
        except Exception as exc:  # noqa: BLE001  (a report must never fail the run)
            self._runlog(f"report.html not written: {exc!r}")

    def _write_manifest(self, manifest: RunManifest) -> None:
        manifest.save(self.output_dir / "manifest.json")

    def _save_sessions(self, sessions: dict[tuple[str, str, int], Session]) -> None:
        data = {"|".join(map(str, k)): s.to_dict() for k, s in sessions.items()}
        tmp = self.output_dir / "sessions.json.tmp"
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.output_dir / "sessions.json")

    def _load_sessions(self) -> dict[str, dict]:
        p = self.output_dir / "sessions.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def _append_jsonl(self, name: str, obj: dict) -> None:
        with (self.output_dir / name).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")

    # -- manifest -------------------------------------------------------------
    def new_manifest(self, run_id: str, rungs: list[Rung]) -> RunManifest:
        utc, et = stamp_pair()
        names = self.spec.metrics
        method = DecodeMethod(
            window="fixed_output_tokens",
            output_tokens=self.spec.ladder.output_tokens_per_rung,
            counters=[
                names.drafts,
                names.draft_tokens,
                names.accepted_tokens,
                names.accepted_per_pos,
                names.preemptions,
                names.prefix_hits,
                names.prefix_queries,
                names.generation_tokens,
                names.gpu_cache_usage,
                names.running,
                names.waiting,
                "vllm:request_queue_time_seconds",
            ],
        )
        return RunManifest(
            run_id=run_id,
            spec_name=self.spec.name,
            profile=self.spec.profile,
            started_utc=utc,
            started_local=et,
            display_timezone=local_tz_name(),
            image=self.spec.server.image,
            image_digest=self.spec.server.image_digest,
            server_source_commit=self.spec.server.source_commit,
            hardware=HardwareSnapshot(
                host=self.spec.hardware.host,
                gpu_model=self.spec.hardware.gpu_model,
                gpu_count=self.spec.hardware.gpu_count,
                tensor_parallel=self.spec.hardware.tensor_parallel,
                host_ram_gib=self.spec.hardware.host_ram_gib,
                mem_available_floor_gib=self.spec.hardware.mem_available_floor_gib,
            ),
            config={
                "env": self.spec.server.env,
                "launch_args": self.spec.server.launch_args,
                "attention_requested": self.spec.server.attention_requested,
                "max_num_seqs": self.spec.server.max_num_seqs,
                "max_num_batched_tokens": self.spec.server.max_num_batched_tokens,
                "gpu_memory_utilization": self.spec.server.gpu_memory_utilization,
                "boot_once": self.spec.ladder.boot_once,
                "grow_monotonically": self.spec.ladder.grow_monotonically,
                "shape": self.spec.ladder.shape,
                "concurrency_steps": list(self.spec.ladder.concurrency_steps),
                "concurrency_session_steps": list(self.spec.ladder.concurrency_session_steps or self.spec.ladder.concurrency_steps),
                "temperature": self.spec.ladder.temperature,
                "ignore_eos": self.spec.ladder.ignore_eos,
                "prompt_style": self.spec.prompts.style,
            },
            attention=AttentionSnapshot(requested=self.spec.server.attention_requested),
            kv_format=self.spec.server.kv_format,
            graph=GraphSnapshot(
                mode=self.spec.server.graph.mode,
                per_gpu=self.spec.server.graph.per_gpu,
                capture_sizes=list(self.spec.server.graph.capture_sizes),
                # Evidence only: set from the boot log at preflight, else unknown.
                graphs_on=None,
                graphs_basis="unknown",
                graphs_declared=(None if self.spec.server.graph.mode is None
                                 else self.spec.server.graph.mode != "EAGER"),
                eager_fallback=False,
            ),
            speculative=SpecDecodeSnapshot(
                method=self.spec.server.speculative.method,
                num_speculative_tokens=self.spec.server.speculative.num_speculative_tokens,
                dflash_k=self.spec.server.speculative.dflash_k,
            ),
            decode_method=method,
            output_jsonl=str(self.output_dir / "results.jsonl"),
            notes=list(self.spec.notes),
            prompt_seed=self.spec.prompts.seed,
            k=self.k_declared,
        )

    # -- per-rung pieces ------------------------------------------------------
    def _stream(self, messages: list[dict[str, str]], max_tokens: int, measured: bool) -> StreamResult:
        extra: dict[str, Any] = {"skip_special_tokens": False, "return_token_ids": True}
        if measured and self.spec.ladder.ignore_eos:
            extra["ignore_eos"] = True
        return self.client.chat_stream(
            messages, max_tokens=max_tokens, temperature=self.spec.ladder.temperature, extra=extra
        )

    def _fit_to_target(self, sess: Session, target: int, tol: float = 0.01) -> int | None:
        """Make the new user turn bring the prompt to ``target`` tokens
        (within ``tol``), counted by the server's own tokenizer (/tokenize
        with the chat messages) and topped up or trimmed before anything is
        sent. Returns the counted prompt tokens (None: no tokenizer; the
        chars-per-token estimate stands and the row records what the server
        reported)."""
        if getattr(self, "_tokenize_ok", True) is False:
            return None
        count = None
        for _ in range(6):
            try:
                count = self.client.count_chat_tokens(sess.messages, url=self.spec.endpoint.tokenize_url)
            except Exception:  # noqa: BLE001
                count = None
            if count is None:
                self._tokenize_ok = False
                return None
            if abs(count - target) <= tol * target:
                break
            turn = sess.turns[-1]
            body_chars = len(sess.messages[-1]["content"])
            cpt = max(sess.chars() / max(count, 1), 0.5)
            sess.resize_last_user(int(turn["n_chars"]) + int((target - count) * cpt))
            if len(sess.messages[-1]["content"]) == body_chars:
                break
        sess.cpt = max(sess.chars() / max(count or 1, 1), 0.5)
        return count

    def _grow(self, rung: Rung, sessions: dict[int, Session], watcher: RamWatcher, metrics_ok: bool) -> tuple[list, list[str]]:
        """Grow sessions one at a time (joiners join one at a time). Returns (prefill stats, errors)."""
        stats = []
        errors: list[str] = []
        for sid in rung.grow_sessions:
            watcher.raise_if_tripped()
            sess = sessions[sid]
            n_turns = len(sess.messages)
            messages = sess.grow_to(rung.target_for(sid))
            if len(sess.messages) > n_turns:
                self._fit_to_target(sess, rung.target_for(sid))
                messages = sess.messages
            before = parse_prometheus(self._metrics_text()) if metrics_ok else {}
            res = self._stream(messages, max_tokens=max(self.spec.ladder.warmup.tokens, 1), measured=False)
            after = parse_prometheus(self._metrics_text()) if metrics_ok else {}
            if res.error or res.timed_out:
                errors.append(f"grow s{sid}: {res.error or 'timeout'}")
                break
            sess.calibrate(res.prompt_tokens)
            new_tokens = None
            if res.prompt_tokens is not None and res.cached_tokens is not None:
                new_tokens = max(int(res.prompt_tokens) - int(res.cached_tokens), 0)
            elif before and after:
                q = metric_value(after, self.spec.metrics.prefix_queries)
                h = metric_value(after, self.spec.metrics.prefix_hits)
                q0 = metric_value(before, self.spec.metrics.prefix_queries)
                h0 = metric_value(before, self.spec.metrics.prefix_hits)
                if None not in (q, h, q0, h0):
                    new_tokens = int((q - q0) - (h - h0))  # type: ignore[operator]
            stats.append(
                stream_stats(
                    res.t_send,
                    res.first_token_at,
                    res.t_end,
                    res.token_times,
                    res.output_tokens,
                    prompt_tokens=res.prompt_tokens,
                    cached_tokens=res.cached_tokens,
                    new_prompt_tokens=new_tokens,
                )
            )
        return stats, errors

    def _quality(
        self,
        rung: Rung,
        sessions: dict[int, Session],
        results: dict[int, StreamResult],
        run_key: str,
    ) -> QualityResult:
        q = QualityResult()
        qs = self.spec.quality
        if not qs.enabled:
            return q
        # 3a) format / repetition on the measured outputs (writing + chat)
        if qs.format_checks and rung.prompt_type in ("long_form", "multi_turn_chat"):
            problems: list[str] = []
            reps: list[float] = []
            for sid, res in results.items():
                fc = format_check(res.text, rung.prompt_type, qs.repetition_ngram, qs.repetition_max_ratio)
                reps.append(fc.repetition)
                problems += [f"s{sid}: {p}" for p in fc.problems]
            q.format_ok = not problems
            q.format_problems = problems[:10]
            q.repetition_max = max(reps) if reps else None
        # 2) greedy divergence vs stored reference (MTP-off / dense boot)
        if self.spec.ladder.temperature == 0.0:
            statuses = []
            for sid, res in results.items():
                probe = f"{rung.rung_id}:s{sid}"
                self._append_jsonl("greedy_outputs.jsonl", {"probe_id": probe, "token_ids": res.token_ids, "text": res.text})
                dkey = (rung.family, rung.load_mode, sid)
                ref = self.greedy_ref.get(probe)
                if self.diverged.get(dkey):
                    statuses.append("context_diverged")
                    continue
                if not ref:
                    statuses.append("no_reference")
                    continue
                a = res.token_ids or list(res.text)
                b = ref.get("token_ids") or list(ref.get("text") or "")
                idx = first_divergence(a, b)
                if idx is None:
                    statuses.append("match")
                else:
                    statuses.append("diverged")
                    self.diverged[dkey] = True
                    if q.greedy_first_divergence is None:
                        q.greedy_first_divergence = idx
            if statuses:
                order = ["diverged", "context_diverged", "no_reference", "match"]
                q.greedy_ref = next(s for s in order if s in statuses)
        s1 = sessions.get(rung.decoding_sessions[0])
        # 3b) needle at this context (deepest + newest planted fact)
        if qs.needle and s1 and s1.needles:
            picks = [s1.needles[0]] + ([s1.needles[-1]] if len(s1.needles) > 1 else [])
            outcomes = []
            for nd in picks:
                msgs = s1.messages + [{"role": "user", "content": nd["question"]}]
                r = self.client.chat_stream(msgs, max_tokens=24, temperature=0.0, extra={"skip_special_tokens": False})
                ok = (not r.error) and needle_pass(r.text, nd["answer"])
                outcomes.append(ok)
                if not ok and q.needle_detail is None:
                    q.needle_detail = f"segment {nd['segment']}: expected {nd['answer']!r}, got {cut_at_eos(r.text)[0][:60]!r}"
            q.needle_pass = all(outcomes)
        # 3c) code edit checked by a unit test, on the live code-edit context
        if qs.code_edit and s1 and rung.prompt_type == "code_edits" and rung.phase in ("S", "meas"):
            task = code_task_for(s1.seed + s1.segments)
            msgs = s1.messages + [{"role": "user", "content": task["prompt"]}]
            r = self.client.chat_stream(msgs, max_tokens=256, temperature=0.0, extra={"skip_special_tokens": False})
            code = extract_code(r.text) if not r.error else None
            if code is None:
                q.code_edit_pass, q.code_edit_detail = False, f"{task['id']}: no code in reply"
            else:
                ok, detail = run_code_test(code, task["test"], qs.code_edit_timeout_s)
                q.code_edit_pass, q.code_edit_detail = ok, f"{task['id']}: {detail}"
        # 1) logprob scoring vs dense baseline
        if qs.logprob_mode != "off":
            scores: dict[str, float] = {}
            probes = logprob_probes(self.spec.prompts.seed, int(qs.probe_ctx_tokens * (s1.cpt if s1 else 4.0)))
            if qs.logprob_mode == "ladder_ctx" and s1:
                ctx_text = "\n\n".join(m["content"] for m in s1.messages)
                probes = [type(p)(f"{p.probe_id}@{rung.rung_id}", ctx_text + "\n\n" + p.context, p.answer) for p in probes]
            for p in probes:
                ctx_ids = self.client.tokenize(p.context, url=self.spec.endpoint.tokenize_url, add_special_tokens=True)
                ans_ids = self.client.tokenize(p.answer, url=self.spec.endpoint.tokenize_url, add_special_tokens=False)
                if not ctx_ids or not ans_ids:
                    continue
                lps = self.client.prompt_logprobs(ctx_ids + ans_ids)
                if not lps:
                    continue
                m = mean_answer_logprob(lps, len(ctx_ids))
                if m is not None:
                    scores[p.probe_id] = m
                    self._append_jsonl("logprob_scores.jsonl", {"probe_id": p.probe_id, "mean_logprob": m, "rung_id": rung.rung_id, "run": run_key})
            q.logprob_scores = scores
            deltas = [scores[k] - float(self.logprob_base[k]["mean_logprob"]) for k in scores if k in self.logprob_base]
            if deltas:
                q.logprob_worst_delta = min(deltas)
                q.logprob_ok = q.logprob_worst_delta >= -qs.logprob_tolerance_nats
        return q

    def _label_flags(self, attention_runtime: str, n_decoding: int, k: int | None) -> tuple[list[str], int | None, list[int]]:
        flags: list[str] = []
        boot = self.boot
        req = self.spec.server.attention_requested
        sizes = list((boot.capture_sizes_logged if boot and boot.capture_sizes_logged else self.spec.server.graph.capture_sizes) or [])
        batch = n_decoding * (k + 1) if k else None
        if boot:
            if req == "qsa" and boot.attention_boot == "dense":
                flags.append("NOT_AS_LABELED")
            if req == "dense" and boot.attention_boot == "qsa":
                flags.append("NOT_AS_LABELED")
            if self.k_declared and boot.spec_k and boot.spec_k != self.k_declared:
                flags.append("NOT_AS_LABELED")
            if boot.kv_dtype and self.spec.server.kv_format and boot.kv_dtype != self.spec.server.kv_format:
                flags.append("NOT_AS_LABELED")
            if self.spec.server.graph.mode not in (None, "EAGER") and boot.graphs_on is False:
                flags.append("NOT_AS_LABELED")
            if boot.graphs_on is False:
                flags.append("GRAPH_MISS")
        if batch is not None and sizes and batch > max(sizes):
            flags.append("GRAPH_MISS")
        if self.launch_mismatches:
            flags.append("NOT_AS_LABELED")
        if attention_runtime == "fallback":
            flags.append("QSA_FALLBACK")
        if attention_runtime == "mixed":
            flags.append("MIXED_PATH")
        return sorted(set(flags)), batch, sizes

    def _quiesce(self, rung: Rung) -> tuple[dict[str, float], float]:
        """Scrape /metrics until no request is running or waiting, for at most
        ``ladder.quiesce_timeout_s``. vLLM can still count the runner's own
        just-finished requests (growth, previous rung) for a moment.

        Returns the last snapshot (the window's ``before``) and running + waiting in it.
        """
        names = self.spec.metrics
        timeout = self.spec.ladder.quiesce_timeout_s
        waited = 0.0
        while True:
            snap = parse_prometheus(self._metrics_text())
            busy = (metric_value(snap, names.running) or 0.0) + (metric_value(snap, names.waiting) or 0.0)
            if busy <= 0 or waited >= timeout:
                break
            step = min(QUIESCE_POLL_S, timeout - waited)
            self.sleep(step)
            waited += step
        if waited:
            state = "idle" if busy <= 0 else f"still busy ({busy:.0f} running+waiting)"
            self._runlog(f"{rung.rung_id}: waited {waited:.2f}s for the server to go idle: {state}")
        return snap, busy

    def _steady(self, rung: Rung, collected: dict[int, StreamResult]) -> tuple[list[float] | None, float | None]:
        """Per-user decode tok/s inside the all-decoding overlap, and its length.

        One session: the overlap is its own decode window, so the values equal
        ``decode_tok_s_per_user`` / ``decode_window_s`` (SHORT_WINDOW covers a
        short one). Several sessions: (None, window) when any session has fewer
        than ``ladder.min_window_steps`` verify steps inside the overlap.
        """
        sw = steady_window([(collected[s].token_times, collected[s].chunk_tokens) for s in rung.decoding_sessions])
        if len(rung.decoding_sessions) > 1 and sw.min_steps < self.spec.ladder.min_window_steps:
            return None, sw.window_s
        return sw.tok_s_each, sw.window_s

    def _window(self, rung: Rung, sessions: dict[int, Session]):
        """One measured decode window on the sessions' current context."""
        before, busy_before = self._quiesce(rung)
        t_window = time.time()
        sampler = WindowSampler(self._metrics_text, self.spec.metrics, self.spec.ladder.metrics_poll_s, self._procs)
        sampler.start()
        delay = self.spec.ladder.staggered.delay_s if rung.load_mode == "staggered" else 0.0
        collected: dict[int, StreamResult] = {}

        def _one(sid: int, start_delay: float) -> None:
            if start_delay:
                self.sleep(start_delay)
            collected[sid] = self._stream(sessions[sid].messages, rung.output_tokens, measured=True)

        threads = []
        for i, sid in enumerate(rung.decoding_sessions):
            start_delay = i * delay
            if rung.load_mode == "open_loop":
                start_delay = i / max(self.spec.ladder.open_loop.requests_per_second, 1e-6)
            th = threading.Thread(target=_one, args=(sid, start_delay), daemon=True)
            threads.append(th)
            th.start()
        for th in threads:
            th.join()
        samples = sampler.stop()
        after = parse_prometheus(self._metrics_text())
        log_lines = bootlog.lines_since(self._logs_text(), t_window - 1.0)
        return collected, samples, before, after, log_lines, t_window, busy_before

    def _repeat_window(self, rung: Rung, sessions: dict[int, Session], k: int | None) -> tuple[dict[str, float | None] | None, str | None]:
        """Re-measure the decode window on the same context (no regrow, replies not appended).

        Returns the per-repeat values of the repeated metrics, or (None, error).
        Prefill is not re-measured (the context is not regrown); the caller
        carries the rung's single prefill value.
        """
        collected, _samples, before, after, _logs, _t, _busy = self._window(rung, sessions)
        errs = []
        for sid in rung.decoding_sessions:
            r = collected.get(sid)
            if r is None or r.error:
                errs.append(f"session {sid}: {'no result' if r is None else r.error}")
        if errs:
            return None, "; ".join(errs)
        mtp = mtp_window(before, after, self.spec.metrics)
        chunk_stats, sstats, per_step_ok = [], [], True
        for sid in rung.decoding_sessions:
            r = collected[sid]
            per_step_ok = per_step_ok and r.per_step_counts
            chunk_stats.append(ChunkStats(tokens=r.output_tokens, chunks=r.chunks, t_first=r.first_token_at, t_last=r.last_token_at))
            sstats.append(stream_stats(r.t_send, r.first_token_at, r.t_end, r.token_times, r.output_tokens,
                                       prompt_tokens=r.prompt_tokens, cached_tokens=r.cached_tokens))
        E = sum(c.decode_tokens for c in chunk_stats)
        S_stream = sum(c.verify_steps for c in chunk_stats)
        S_server = int(mtp.drafts) if mtp.drafts is not None else None
        if per_step_ok and S_stream > 0:
            S = S_stream
            steps_each = [c.steps_s for c in chunk_stats]
        else:
            S = S_server
            share = (S_server or 0) / max(len(chunk_stats), 1)
            steps_each = [(share / c.window_s) if c.window_s else None for c in chunk_stats]
        return {
            "decode_tok_s_per_user": _mean([c.decode_tok_s for c in chunk_stats]),
            "decode_tok_s_per_user_steady": _mean(self._steady(rung, collected)[0] or []),
            "steps_s_per_user": _mean(steps_each),
            "accepted_drafts_per_step": (E / S - 1.0) if S else None,
            "hit_rate": true_hit_rate(E, S, k) if (k and S) else None,
            "ttft_ms": _mean([st.ttft_ms for st in sstats]),
        }, None

    def _measure_rung(
        self,
        run_id: str,
        rung: Rung,
        sessions: dict[int, Session],
        watcher: RamWatcher,
        extra_flags: list[str] | None = None,
    ) -> RungResult:
        utc0, et0 = stamp_pair()
        seeds = [sessions[s].seed for s in rung.decoding_sessions]
        base = dict(
            run_id=run_id,
            rung_id=rung.rung_id,
            users=rung.users,
            ctx_per_user=rung.ctx_per_user,
            prompt_type=rung.prompt_type,
            load_mode=rung.load_mode,
            phase=rung.phase,
            prompt_seed=self.spec.prompts.seed,
            session_seeds=seeds,
            temperature=self.spec.ladder.temperature,
            k=self.k_declared,
            kv_format=self.spec.server.kv_format,
            build_fingerprint=self.fp.build_fingerprint if self.fp else None,
            launch_fingerprint=self.fp.launch_fingerprint if self.fp else None,
            image_id=self.fp.image_id if self.fp else None,
            boot_started_at=self.fp.started_at if self.fp else None,
            attention_boot=self.boot.attention_boot if self.boot else None,
            started_utc=utc0,
            started_local=et0,
        )
        if self.dry_run:
            for sid in rung.grow_sessions:  # no prompt text is generated or stored
                sessions[sid].ctx_tokens = rung.target_for(sid)
            utc1, et1 = stamp_pair()
            return RungResult(
                **base,
                ctx_tokens=[rung.target_for(s) for s in rung.decoding_sessions],
                valid="valid",
                reason="dry-run",
                flags=["DRY_RUN"],
                finished_utc=utc1,
                finished_local=et1,
            )

        # Prefill / growth (joiners one at a time). Outside the measured window.
        grow_stats, grow_errors = self._grow(rung, sessions, watcher, metrics_ok=True)
        watcher.raise_if_tripped()
        if grow_errors:
            return self._failed_row(base, rung, sessions, grow_errors, timed_out=False, since=time.time() - 600)

        # Measured window (repeat 1; more repeats below on the same context).
        collected, samples, before, after, log_lines, t_window, busy_before = self._window(rung, sessions)

        errors: list[str] = []
        timeouts = 0
        for sid in rung.decoding_sessions:
            r = collected.get(sid)
            if r is None:
                errors.append(f"session {sid}: no result")
            elif r.error:
                errors.append(f"session {sid}: {r.error}")
            if r is not None and r.timed_out:
                timeouts += 1
        watcher.raise_if_tripped()
        if errors:
            return self._failed_row(base, rung, sessions, errors, timed_out=timeouts > 0, since=t_window, log_lines=log_lines)

        # Per-session stream accounting.
        mtp = mtp_window(before, after, self.spec.metrics)
        sysm = system_window(before, after, self.spec.metrics)
        flags: list[str] = list(extra_flags or [])
        k = self.k_declared
        if k is None and mtp.drafts and mtp.drafts > 0 and mtp.draft_tokens:
            k = round(mtp.draft_tokens / mtp.drafts)
            flags.append("K_UNDECLARED")
        chunk_stats: list[ChunkStats] = []
        sstats = []
        ctxs: list[int] = []
        per_step_ok = True
        eos_seen = False
        for sid in rung.decoding_sessions:
            r = collected[sid]
            per_step_ok = per_step_ok and r.per_step_counts
            cs = ChunkStats(tokens=r.output_tokens, chunks=r.chunks, t_first=r.first_token_at, t_last=r.last_token_at)
            chunk_stats.append(cs)
            sstats.append(
                stream_stats(r.t_send, r.first_token_at, r.t_end, r.token_times, r.output_tokens,
                             prompt_tokens=r.prompt_tokens, cached_tokens=r.cached_tokens)
            )
            if cut_at_eos(r.text)[1]:
                eos_seen = True
            sessions[sid].calibrate(r.prompt_tokens)
            ctxs.append(sessions[sid].ctx_tokens or rung.target_for(sid))

        E = sum(c.decode_tokens for c in chunk_stats)
        S_stream = sum(c.verify_steps for c in chunk_stats)
        S_server = int(mtp.drafts) if mtp.drafts is not None else None
        if per_step_ok and S_stream > 0:
            S = S_stream
            steps_user_each = [c.steps_s for c in chunk_stats]
        else:
            S = S_server
            flags.append("DERIVED_STEPS")
            share = (S_server or 0) / max(len(chunk_stats), 1)
            steps_user_each = [(share / c.window_s) if c.window_s else None for c in chunk_stats]
        if S_server and S_stream and per_step_ok and abs(S_stream - S_server) / S_server > 0.05:
            flags.append("STEP_COUNT_MISMATCH")
        tok_user_each = [c.decode_tok_s for c in chunk_stats]
        hit = true_hit_rate(E, S, k) if k else None
        check = spec_counter_check(
            mtp, k, E, S_stream, len(chunk_stats),
            per_pos_sum(before, after, self.spec.metrics.accepted_per_pos),
        )
        if k and mtp.drafts and not check.ok:
            flags.append("SPEC_COUNTER_BROKEN")
        if any(c.verify_steps < self.spec.ladder.min_window_steps for c in chunk_stats):
            flags.append("SHORT_WINDOW")
        steady_each, overlap_s = self._steady(rung, collected)
        if steady_each is None and len(chunk_stats) > 1:
            flags.append("SHORT_OVERLAP")
        if eos_seen:
            flags.append("EOS_IN_WINDOW")
        if sysm.preemptions:
            flags.append("PREEMPTED")
        running_max = max(samples.running) if samples.running else None
        waiting_max = max(samples.waiting) if samples.waiting else None
        # busy_before: still running/waiting after the pre-window quiescence wait.
        if busy_before > 0 or (running_max or 0) + (waiting_max or 0) > len(rung.decoding_sessions):
            flags.append("CONCURRENT_CLIENT")
        if samples.other_stacks:
            flags.append("FOREIGN_STACK")

        # Attention path in this window (runtime lines) on top of boot evidence.
        runtime = classify_logs(log_lines, patterns=self.spec.log_patterns, requested=self.spec.server.attention_requested)
        has_runtime = bool(runtime.active_widths or runtime.skip_widths)
        attention = runtime.path if has_runtime else (self.boot.attention_boot if self.boot and self.boot.attention_boot != "unknown" else runtime.path)
        lf, batch, sizes = self._label_flags(runtime.path if has_runtime else "", len(rung.decoding_sessions), k)
        flags += lf
        # Spec assert can be logged without failing the request.
        fail = bootlog.classify_failure(log_lines, [], patterns=self.spec.log_patterns)
        exit_reason = None
        if fail and fail.flag == "SPEC_ASSERT":
            flags.append("SPEC_ASSERT")
            exit_reason = "SPEC_ASSERT"

        # Rebooted mid-run?
        fp_now = self._fingerprint(hash_mounts=False)
        if self.fp and fp_now and fp_now.started_at and self.fp.started_at and fp_now.started_at != self.fp.started_at:
            flags.append("REBOOTED")

        # Repeats 2..N: same context (nothing regrown, no replies appended yet).
        extra: list[dict[str, float | None] | None] = []
        for _ in range(max(self.spec.ladder.repeats, 1) - 1):
            vals, err = self._repeat_window(rung, sessions, k)
            watcher.raise_if_tripped()
            if err:
                flags.append("REPEAT_FAILED")
                self._runlog(f"{rung.rung_id}: repeat failed: {err}")
            extra.append(vals)

        # Assistant replies are appended (conversation grows; never edited).
        for sid in rung.decoding_sessions:
            sessions[sid].append_reply(collected[sid].text)

        quality = self._quality(rung, sessions, {s: collected[s] for s in rung.decoding_sessions}, run_id)
        if quality.needle_pass is False or quality.code_edit_pass is False or quality.format_ok is False or quality.logprob_ok is False or quality.greedy_ref == "diverged":
            flags.append("QUALITY_FAIL")

        prefill_new = [s.prefill_tok_s for s in grow_stats]
        ttft_grow = [s.ttft_ms for s in grow_stats if s.ttft_ms]
        prefill_tokens = None
        prefill_rate = None
        if grow_stats:
            toks = [
                (s.prompt_tokens or 0) - (s.cached_tokens or 0) if s.cached_tokens is not None else None
                for s in grow_stats
            ]
            if all(t is not None for t in toks) and ttft_grow:
                prefill_tokens = int(sum(toks))  # type: ignore[arg-type]
                prefill_rate = prefill_tokens / (sum(ttft_grow) / 1000.0) if sum(ttft_grow) > 0 else None
            else:
                prefill_rate = _mean(prefill_new)
        itls = []
        for sid in rung.decoding_sessions:
            tt = collected[sid].token_times
            itls += [(b - a) * 1000.0 for a, b in zip(tt, tt[1:], strict=False)]
        windows = [c.window_s for c in chunk_stats if c.window_s]
        tpots = [((s.e2e_ms or 0) - (s.ttft_ms or 0)) / max(c.tokens - 1, 1) for s, c in zip(sstats, chunk_stats, strict=False) if s.ttft_ms is not None and c.tokens > 1]
        gpu = None
        try:
            gpu = self.collect_mem(container=None, kv_cache_usage_pct=sysm.kv_cache_usage_pct)
        except SafetyError:
            gpu = None
        utc1, et1 = stamp_pair()
        valid = "valid"
        reason = None
        fill_pct = ctx_fill_pct(ctxs, rung.ctx_per_user)
        if fill_pct is not None and abs(fill_pct - 100.0) > 2.0:
            flags.append("CTX_FILL_OFF")
        hard = {"NOT_AS_LABELED", "SPEC_COUNTER_BROKEN", "SPEC_ASSERT", "REBOOTED", "CONCURRENT_CLIENT", "QSA_FALLBACK",
                "MIXED_PATH", "PRE_EXISTING_LOAD"}
        if set(flags) & hard:
            valid = "warning"
            reason = "flags: " + " ".join(sorted(set(flags) & hard))
        if attention in ("fallback", "mixed") and valid == "valid":
            valid, reason = "warning", f"attention path={attention}"
        acc_step = (E / S - 1.0) if S else None
        row = RungResult(
            **base,
            ctx_tokens=ctxs,
            ctx_fill_pct=fill_pct,
            capture_sizes=sizes,
            decode_batch_tokens=batch,
            prefill_tok_s=prefill_rate,
            prefill_tokens=prefill_tokens,
            ttft_ms=_mean([s.ttft_ms for s in sstats]),
            decode_tok_s_per_user=_mean(tok_user_each),
            decode_tok_s_total=_sum(tok_user_each),
            decode_tok_s_per_user_steady=_mean(steady_each or []),
            decode_tok_s_total_steady=_sum(steady_each or []),
            steady_window_s=overlap_s,
            steps_s_per_user=_mean(steps_user_each),
            steps_s_total=_sum(steps_user_each),
            tpot_ms=_mean(tpots),
            decode_window_s=_mean(windows),
            accepted_drafts_per_step=acc_step,
            hit_rate=hit,
            acceptance_rate_vllm=mtp.hit_rate,
            tokens_per_step=(E / S) if S else None,
            tokens_delivered=E,
            verify_steps=S,
            verify_steps_stream=S_stream if per_step_ok else None,
            verify_steps_server=S_server,
            draft_tokens=int(mtp.draft_tokens) if mtp.draft_tokens is not None else None,
            accepted_drafts=int(mtp.accepted_drafts) if mtp.accepted_drafts is not None else None,
            drafts=S_server,
            itl_p50_ms=percentile(itls, 50),
            itl_p95_ms=percentile(itls, 95),
            itl_p99_ms=percentile(itls, 99),
            e2e_ms=_mean([s.e2e_ms for s in sstats]),
            queue_time_ms=histogram_mean_ms(before, after, "vllm:request_queue_time_seconds") or sysm.queue_time_ms,
            running_reqs=_mean(samples.running) if samples.running else sysm.running_reqs,
            waiting_reqs=_mean(samples.waiting) if samples.waiting else sysm.waiting_reqs,
            running_max=running_max,
            waiting_max=waiting_max,
            prefix_cache_hit_rate=sysm.prefix_cache_hit_rate,
            kv_cache_usage_pct=sysm.kv_cache_usage_pct,
            kv_cache_usage_max_pct=max(samples.kv_pct) if samples.kv_pct else sysm.kv_cache_usage_pct,
            preemptions=int(sysm.preemptions) if sysm.preemptions is not None else None,
            mem_available_min_gib=watcher.min_gib(),
            gpu_mem=gpu,
            rss_kib=samples.rss,
            other_vllm_stacks=sorted(samples.other_stacks),
            attention_path=attention,  # type: ignore[arg-type]
            attention_evidence=(runtime.evidence or (self.boot.evidence if self.boot else []))[:12],
            graphs_on=self.boot.graphs_on if self.boot else None,
            eager_fallback=bool(runtime.skip_widths and self.spec.server.graph.eager_fallback_when_unmatched),
            exit_reason=exit_reason,
            flags=sorted(set(flags)),
            quality=quality,
            valid=valid,  # type: ignore[arg-type]
            reason=reason,
            errors=[],
            timeouts=0,
            finished_utc=utc1,
            finished_local=et1,
        )
        if self.spec.ladder.repeats > 1:
            apply_repeats(row, extra)
        return row

    def _failed_row(
        self,
        base: dict,
        rung: Rung,
        sessions: dict[int, Session],
        errors: list[str],
        timed_out: bool,
        since: float,
        log_lines: list[str] | None = None,
        ram_floor: bool = False,
        mem_min: float | None = None,
    ) -> RungResult:
        if log_lines is None:
            try:
                log_lines = bootlog.lines_since(self._logs_text(), since - 1.0)
            except Exception:  # noqa: BLE001
                log_lines = []
        fp_now = None if ram_floor else self._fingerprint(hash_mounts=False)
        fail = bootlog.classify_failure(
            log_lines, errors, timed_out=timed_out, ram_floor_hit=ram_floor,
            oom_killed=bool(fp_now and fp_now.oom_killed), patterns=self.spec.log_patterns,
        )
        flag = fail.flag if fail else "CLIENT_ERROR"
        utc1, et1 = stamp_pair()
        detail = (fail.detail if fail else "") + (" | " + " / ".join(fail.lines[:2]) if fail and fail.lines else "")
        return RungResult(
            **base,
            ctx_tokens=[sessions[s].ctx_tokens or rung.target_for(s) for s in rung.decoding_sessions],
            mem_available_min_gib=mem_min,
            valid="invalid",
            reason=detail[:500] or "; ".join(errors)[:500],
            errors=errors[:10],
            timeouts=1 if timed_out else 0,
            exit_reason=flag,
            flags=[flag],
            finished_utc=utc1,
            finished_local=et1,
        )

    # -- preflight ------------------------------------------------------------
    def _would_run(self) -> list[str]:
        if self.hold is None:
            lines = ["(no bench-hold: safety.hold_command is not set, so no host-wide hold is taken)"]
        else:
            cmd, name = self.hold.command, self.hold.name
            lines = [
                f"{cmd} status",
                f"{cmd} acquire {self.spec.safety.hold_minutes} {self.hold_reason!r} --by {name}",
                f"{cmd} status",
                "...",
                f"{cmd} release  (only if status still shows by={name})",
            ]
        if self.spec.safety.quiet_hold_file:
            lines.insert(0, f"read {self.spec.safety.quiet_hold_file}  (an active quiet hold blocks the run)")
        if self.spec.server.container_name:
            lines.insert(0, f"docker inspect {self.spec.server.container_name}  (must exist and be running)")
        return lines

    def _preflight(self, manifest: RunManifest, rungs: list[Rung]) -> list[Rung]:
        """Safety, identity and feasibility checks. Returns the rung list,
        rebuilt when ``context_rungs_tokens: auto`` resolves against the server."""
        safety_preflight(self.spec)
        site = load_site()
        manifest.site_config = describe_site()
        problems: list[str] = []
        warnings: list[str] = []
        if not site.source:
            msg = (
                "no site config found: production containers/ports of this host are NOT protected beyond the "
                "built-in neutral defaults. Create one (README 'Site config') or pass --no-site-config to "
                "acknowledge."
            )
            if self.no_site_config:
                msg = (
                    "running WITHOUT a site config (--no-site-config): only the built-in neutral "
                    "protection applies"
                )
            (warnings if (self.no_site_config or self.dry_run) else problems).append(msg)
        holes = spec_placeholders(self.spec)
        if holes:
            msg = (
                "spec still has placeholder values (set each to the real value, or to null to skip its check):\n    "
                + "\n    ".join(holes)
            )
            (warnings if self.dry_run else problems).append(msg)
        if self.spec.ladder.shape == "grid":
            warnings.append(GRID_WARNING)
        if self.dry_run:
            manifest.hold = {"dry_run": True, "would_run": self._would_run()}
            self.preflight_warnings = warnings
            manifest.launch_check = {"checked": False, "warnings": warnings}
            if problems:
                raise RunRefused("\n".join(problems), manifest)
            return rungs
        if problems:  # stop before touching docker or the host
            raise RunRefused("\n".join(problems), manifest)
        # Inspect only: mounts are hashed after every check has passed (_hash_mounts).
        self.fp = self._fingerprint(hash_mounts=False)
        fp = self.fp
        if self.require_container:
            name = self.spec.server.container_name
            if fp is None or not fp.container_id:
                why = "; ".join(fp.problems) if fp and fp.problems else "no fingerprint"
                problems.append(
                    f"test container {name!r} not found ({why}). Boot it yourself first and point "
                    "server.container_name / endpoint.base_url at it (README 'Prerequisites')."
                )
            elif fp.state_status and fp.state_status != "running":
                problems.append(f"test container {name!r} is {fp.state_status}, not running")
        if fp and fp.container_id:
            bad = set(fp.host_ports) & site_forbidden_ports(self.spec.safety.forbidden_ports)
            if bad:
                raise SafetyError(f"Refusing: container {fp.container} is bound to production port(s) {sorted(bad)}")
            manifest.fingerprint = fp.to_dict()
            srv = self.spec.server
            self.launch_mismatches = verify_expected(
                fp, srv.env, srv.launch_args, srv.image, srv.image_digest, is_placeholder=is_placeholder
            )
            manifest.launch_check = {
                "checked": True,
                "ok": not self.launch_mismatches,
                "mismatches": self.launch_mismatches,
                "note": "env / launch_args / image are expectations verified against docker inspect, not applied",
            }
            if self.launch_mismatches:
                warnings.append(
                    "container does not match the spec; every row will be flagged NOT_AS_LABELED:\n    "
                    + "\n    ".join(self.launch_mismatches)
                )
        else:
            manifest.launch_check = {"checked": False, "reason": "no container fingerprint"}
        log_text: str | None = None
        try:
            log_text = self._logs_text(tail=200000)
            self.boot = bootlog.parse_boot(log_text, self.spec.log_patterns)
        except Exception:  # noqa: BLE001
            self.boot = None
        self._capture_launch(manifest, log_text)
        if self.boot:
            manifest.boot = self.boot.summary()
            if manifest.graph is not None and self.boot.graphs_on is not None:
                manifest.graph.graphs_on = self.boot.graphs_on
                manifest.graph.graphs_basis = "boot_log"
            manifest.attention.actual = self.boot.attention_boot  # type: ignore[assignment]
            manifest.attention.evidence_lines = self.boot.evidence[:20]
        rungs = self._resolve_ladder(manifest, rungs)
        # Feasibility against live MemAvailable on the bench host (the server
        # is booted, so this already counts it and every other process).
        try:
            mem_now = self.ram_source.read_kib() / (1024 * 1024)
        except Exception:  # noqa: BLE001
            mem_now = None
        feas = feasibility(
            self.spec, rungs, mem_now, "live", manifest.kv_capacity_tokens, manifest.kv_capacity_source or ""
        )
        manifest.launch_check["feasibility"] = {
            "gpu": feas.gpu_status, "host": feas.host_status, "host_basis": feas.host_basis,
            "mem_available_gib": round(mem_now, 2) if mem_now is not None else None,
            "host_growth_gib": round(feas.host_need_gib, 2), "gpu_need_gib": round(feas.estimated_total_gib, 2),
        }
        manifest.launch_check["feasibility"]["ceiling_estimates"] = [
            {"key": c.key, "users": c.users, "top_ok": c.top_ok, "stops_at": c.stops_at, "reason": c.reason}
            for c in feas.ceilings
        ]
        warnings.extend(n for n in feas.notes if n.startswith("WARN"))
        if feas.host_status in ("over", "tight"):
            note = (
                f"INFO host RAM estimate (never blocks): est. growth {feas.host_need_gib:.1f} GiB vs "
                f"{feas.host_budget_gib:.1f} GiB above the floor ({feas.host_status}); a user count that hits the "
                "live RAM floor stops there (CTX_CEILING)"
            )
            warnings.append(note)
            manifest.notes.append(note)
        if self.hold is None:
            warnings.append(
                "WARN no safety.hold_command configured (site or spec): running without a bench lock, "
                "so nothing stops another job from sharing the GPUs"
            )
        self.preflight_warnings = warnings
        if problems:
            raise RunRefused("\n".join(problems), manifest)
        manifest.host = self._host()
        if manifest.hardware.host_ram_gib is None and manifest.host.get("host_ram_gib"):
            manifest.hardware.host_ram_gib = manifest.host["host_ram_gib"]  # measured MemTotal
        return rungs

    def _hash_mounts(self, manifest: RunManifest) -> None:
        """Build fingerprint, after validation. Never reads model weights (large
        and weight-type files are listed by name + size); a failure is a
        fingerprint problem, never a reason not to run."""
        fp = self.fp
        if fp is None or fp.build_fingerprint is not None or not fp.container_id:
            return
        t0 = time.time()
        try:
            hash_fingerprint_mounts(self.exec, fp)
        except Exception as exc:  # noqa: BLE001
            fp.problems.append(f"mount hashing failed: {exc!r}"[:300])
        self._runlog(f"build fingerprint: {len(fp.mounts)} mount(s) hashed in {time.time() - t0:.1f}s")
        manifest.fingerprint = fp.to_dict()

    def _server_max_model_len(self) -> tuple[int | None, str | None]:
        """max_model_len from /v1/models (vLLM), else from the boot log."""
        try:
            status, body = self.client.get_text(self.client._url("models"))
            if status == 200:
                for item in json.loads(body).get("data") or []:
                    if item.get("max_model_len"):
                        return int(item["max_model_len"]), "/v1/models"
        except Exception:  # noqa: BLE001
            pass
        if self.boot and self.boot.max_model_len:
            return self.boot.max_model_len, "boot_log"
        return None, None

    def _kv_capacity(self) -> tuple[int | None, str | None]:
        """KV cache capacity in tokens: boot log line, else vllm:cache_config_info."""
        if self.boot and self.boot.kv_tokens:
            return self.boot.kv_tokens, "boot_log"
        try:
            status, body = self.client.get_text(self.spec.endpoint.metrics_url)
            if status == 200 and (cap := kv_capacity_from_metrics(body)):
                return cap, "metrics cache_config_info"
        except Exception:  # noqa: BLE001
            pass
        return None, None

    def _resolve_ladder(self, manifest: RunManifest, rungs: list[Rung]) -> list[Rung]:
        """Resolve the context ladder against the live server (``auto`` grows
        to max_model_len) and record max_model_len and KV capacity."""
        mml, src = self._server_max_model_len()
        manifest.max_model_len, manifest.max_model_len_source = mml, src
        manifest.kv_capacity_tokens, manifest.kv_capacity_source = self._kv_capacity()
        if manifest.context_rungs and manifest.completed_rung_ids:
            ctxs, note = list(manifest.context_rungs), manifest.context_note or "resumed"
        else:
            ctxs, note = resolve_contexts(self.spec.ladder, mml)
        manifest.context_rungs, manifest.context_note = ctxs, note
        self._runlog(f"context rungs {ctxs} ({note})")
        self.spec = with_contexts(self.spec, ctxs)
        return build_schedule(self.spec)

    def _refuse(self, manifest: RunManifest, exc: Exception) -> RunRefused:
        manifest.status = "refused"
        manifest.abort_reason = str(exc)
        manifest.finished_utc, manifest.finished_local = stamp_pair()
        self._write_manifest(manifest)
        return RunRefused(str(exc), manifest)

    # -- main loop -------------------------------------------------------------
    def run(self) -> RunManifest:
        rungs = build_schedule(self.spec)
        completed: set[str] = set()
        manifest: RunManifest | None = None
        jsonl = self.output_dir / "results.jsonl"
        prior_boot = None
        if self.resume and (self.output_dir / "manifest.json").exists():
            manifest = RunManifest.load(self.output_dir / "manifest.json")
            if manifest.status == "dry_run":
                manifest = None  # a dry run never counts as progress
            else:
                completed = set(manifest.completed_rung_ids)
                prior_boot = (manifest.fingerprint or {}).get("started_at")
                if manifest.context_rungs and completed:
                    # Same ladder as the interrupted run, even if the server changed.
                    self.spec = with_contexts(self.spec, manifest.context_rungs)
                    rungs = build_schedule(self.spec)
        if manifest is None:
            manifest = self.new_manifest(_run_id(self.spec.name), rungs)
            manifest.context_rungs, manifest.context_note = resolve_contexts(self.spec.ladder, None)
            if jsonl.exists():
                jsonl.unlink()
            for extra in ("sessions.json", "greedy_outputs.jsonl", "logprob_scores.jsonl"):
                (self.output_dir / extra).unlink(missing_ok=True)
        run_id = manifest.run_id
        self._run_id = run_id
        if self.hold is not None and self._hold_auto_name:
            # Unique per run (a resume keeps its run id, so its hold stays "ours").
            self.hold.name = run_id

        # Safety and identity before anything touches the server.
        try:
            rungs = self._preflight(manifest, rungs)
            if not self.dry_run:
                self._check_pre_run_load(manifest)
        except (SafetyError, RunRefused) as exc:
            raise self._refuse(manifest, exc) from None
        if not self.dry_run:
            self._hash_mounts(manifest)
        self._write_manifest(manifest)
        if not self.dry_run:
            for warn in self.preflight_warnings:
                print(warn if warn.startswith("INFO") else f"WARNING: {warn}", file=sys.stderr, flush=True)

        watcher = RamWatcher(floor_gib=self.spec.hardware.mem_available_floor_gib, source=self.ram_source, interval_s=0.5)
        try:
            watcher.check_now()
        except RamFloorError as exc:
            # A warning only: the run starts anyway. A user count that hits the
            # floor during a rung stops there (CTX_CEILING) and the run moves on.
            # A floor the host already sits under cannot tell this run's growth
            # apart, so the RAM floor guard is off for this run (MemAvailable is
            # still sampled); OOM, KV-capacity and server-rejection ceilings still apply.
            note = (f"WARN MemAvailable already under the floor before the first rung ({exc}); running anyway "
                    "with the RAM floor guard off (still sampled); OOM / KV / server ceilings still apply")
            print(f"WARNING: {note}", file=sys.stderr, flush=True)
            manifest.notes.append(note)
            self._runlog(note)
            self._write_manifest(manifest)
            watcher = RamWatcher(floor_gib=0.0, source=watcher.source, interval_s=watcher.interval_s)

        if not self.dry_run and self.hold is None:
            try:
                quiet = quiet_hold_active(
                    self.spec.safety.quiet_hold_file,
                    read_file=self._read_remote if self.spec.access.mode == "ssh" else None,
                )
            except SafetyError as exc:
                raise self._refuse(manifest, exc) from None
            if quiet:
                raise self._refuse(manifest, SafetyError(f"Refusing to start: {quiet}"))
            manifest.hold = {"skipped": "no safety.hold_command configured; no bench-hold taken"}
            self._write_manifest(manifest)
            watcher.start()
        elif not self.dry_run:
            try:
                st = self.hold.guard_and_acquire(
                    self.spec.safety.hold_minutes,
                    self.hold_reason,
                    quiet_hold_file=self.spec.safety.quiet_hold_file,
                    read_file=self._read_remote if self.spec.access.mode == "ssh" else None,
                )
            except SafetyError as exc:
                raise self._refuse(manifest, exc) from None
            manifest.hold = {"by": st.by, "until": st.until, "acquired_by_runner": True}
            self._write_manifest(manifest)
            watcher.start()

        # Corpora (real files if configured).
        if self.spec.prompts.corpus_dir:
            for ptype in self.spec.prompts.types:
                text, files = load_corpus_dir(self.spec.prompts.corpus_dir, ptype)
                self.corpus_text[ptype] = text
                self.corpus_manifest[ptype] = files
            manifest.corpus_files = self.corpus_manifest

        sessions: dict[tuple[str, str, int], Session] = {}
        saved = self._load_sessions() if self.resume else {}
        refill_pending = bool(saved) and not self.dry_run
        same_boot = bool(prior_boot and self.fp and self.fp.started_at == prior_boot)
        abort_reason = None
        family_index = {p: i for i, p in enumerate(self.spec.prompts.types)}
        last_renew = time.time()
        try:
            for rung in rungs:
                if rung.rung_id in completed:
                    continue
                ckey = ceiling_key(rung)
                if ckey in manifest.ctx_ceilings:
                    continue  # this user count already hit its ceiling
                if self._kv_precheck(manifest, run_id, rung, jsonl):
                    continue
                local: dict[int, Session] = {}
                for sid in rung.live_sessions:
                    key = (rung.family, rung.load_mode, sid)
                    if key not in sessions:
                        sess = Session(
                            sid,
                            rung.prompt_type,
                            seed=session_seed(self.spec.prompts.seed, family_index[rung.prompt_type], sid),
                            chars_per_token=self.spec.prompts.chars_per_token_guess,
                            style=self.spec.prompts.style,
                            file_text=self.corpus_text.get(rung.prompt_type),
                            plant_needles=self.spec.prompts.plant_needles,
                        )
                        skey = "|".join(map(str, key))
                        if skey in saved:
                            sess.load_state(saved[skey])
                        sessions[key] = sess
                    local[sid] = sessions[key]
                extra_flags: list[str] = ["PRE_EXISTING_LOAD"] if self._pre_load_flag else []
                if refill_pending:
                    # Re-prefill every live session to its high-water context
                    # outside the window; never shrink, never rebuild smaller.
                    for sess in local.values():
                        if sess.messages:
                            self._stream(sess.messages, 1, measured=False)
                    if not same_boot:
                        extra_flags.append("RESUMED_REFILL")
                    refill_pending = False
                if self.hold is not None and not self.dry_run and time.time() - last_renew > max(self.spec.safety.hold_renew_below_min, 1) * 60:
                    st = self.hold.state()
                    if st.active and st.by == self.hold.name:
                        self.hold.renew(self.spec.safety.hold_minutes, self.hold_reason)
                        last_renew = time.time()
                    else:
                        abort_reason = f"bench-hold no longer ours ({st.raw.strip()[:120]}); stopping"
                        break
                try:
                    watcher.raise_if_tripped()
                    row = self._measure_rung(run_id, rung, local, watcher, extra_flags)
                except RamFloorError as exc:
                    base_seeds = [local[s].seed for s in rung.decoding_sessions]
                    row = self._failed_row(
                        dict(run_id=run_id, rung_id=rung.rung_id, users=rung.users, ctx_per_user=rung.ctx_per_user,
                             prompt_type=rung.prompt_type, load_mode=rung.load_mode, phase=rung.phase,
                             prompt_seed=self.spec.prompts.seed, session_seeds=base_seeds,
                             temperature=self.spec.ladder.temperature, k=self.k_declared,
                             build_fingerprint=self.fp.build_fingerprint if self.fp else None,
                             started_utc=stamp_pair()[0], started_local=stamp_pair()[1]),
                        rung, local, [str(exc)], timed_out=False, since=time.time(), log_lines=[],
                        ram_floor=True, mem_min=watcher.min_gib(),
                    )
                if self.spec.ladder.stop_at_ceiling and not self.dry_run and ceiling_reason(row):
                    row.flags = [*row.flags, "CTX_CEILING"]
                write_jsonl_row(jsonl, row)
                self._runlog(
                    f"rung {row.rung_id} valid={row.valid} decode_tok_s_per_user={row.decode_tok_s_per_user} "
                    f"exit={row.exit_reason} flags={','.join(row.flags) or '-'}"
                )
                if self.dry_run:
                    # Dry runs record what would run; nothing is completed or resumable.
                    manifest.planned_rung_ids.append(rung.rung_id)
                    self._write_manifest(manifest)
                    continue
                self._save_sessions(sessions)
                why = ceiling_reason(row) if self.spec.ladder.stop_at_ceiling else None
                if why:
                    self._record_ceiling(manifest, rung, why, row.reason or "; ".join(row.errors)[:200])
                    self._runlog(f"ceiling {ckey}: stops at {rung.ctx_per_user} ({why}); continuing with the next user count")
                    if why == "ram_floor":
                        # Never aborts the run: give RAM a chance to recover (--ram-wait-min),
                        # note it if it did not, and go on with the next user count.
                        if not self._ram_recovered(watcher):
                            note = (f"WARN MemAvailable still under the floor after the {ckey} ceiling; continuing "
                                    "with the RAM floor guard off (still sampled)")
                            manifest.notes.append(note)
                            self._runlog(note)
                            watcher.floor_gib = 0.0
                        watcher = self._restart_watcher(watcher)
                    manifest.completed_rung_ids.append(rung.rung_id)
                    self._write_manifest(manifest)
                    continue
                if row.valid != "invalid":
                    manifest.ctx_reached[ckey] = max(manifest.ctx_reached.get(ckey, 0), rung.ctx_per_user)
                if row.exit_reason in ABORTING_EXITS and row.valid == "invalid":
                    if row.exit_reason == "RAM_FLOOR" and self.ram_wait_min > 0 and self._wait_for_ram(watcher):
                        watcher = self._restart_watcher(watcher)
                        continue  # rung not marked complete; next --resume or loop picks it up
                    abort_reason = f"{row.exit_reason}: {row.reason}"
                    break
                manifest.completed_rung_ids.append(rung.rung_id)
                if manifest.attention.actual == "unknown":
                    manifest.attention.actual = row.attention_path
                    manifest.attention.evidence_lines = list(row.attention_evidence)
                if row.quality.greedy_ref == "diverged" and not manifest.first_greedy_divergence:
                    manifest.first_greedy_divergence = {"rung_id": row.rung_id, "token_index": row.quality.greedy_first_divergence}
                self._write_manifest(manifest)
        finally:
            watcher.stop()
            if not self.dry_run and self.hold is not None:
                released, note = self.hold.release_if_ours()
                manifest.hold = {**(manifest.hold or {}), "released": released, "release_note": note}

        manifest.finished_utc, manifest.finished_local = stamp_pair()
        if abort_reason:
            manifest.status = "aborted"
            manifest.abort_reason = abort_reason
        elif self.dry_run:
            manifest.status = "dry_run"
        else:
            manifest.status = "completed"
        self._write_manifest(manifest)
        self._runlog(f"run {manifest.status}" + (f": {manifest.abort_reason}" if manifest.abort_reason else ""))
        if self.html and not self.dry_run:
            self._write_html(manifest)
        return manifest

    def _pre_run_load(self) -> dict:
        """Is the server already busy, or the GPU already loaded, before we start?

        Two /metrics scrapes ``load_probe_s`` apart: requests running or
        waiting (server busy), generation tokens produced in between or KV
        cache in use (GPU loaded by this server's other clients). Read-only;
        no process watching.
        """
        names = self.spec.metrics
        try:
            a = parse_prometheus(self._metrics_text())
            self.sleep(self.load_probe_s)
            b = parse_prometheus(self._metrics_text())
        except Exception as exc:  # noqa: BLE001
            return {"checked": False, "error": repr(exc)[:200]}
        running = max(metric_value(a, names.running) or 0.0, metric_value(b, names.running) or 0.0)
        waiting = max(metric_value(a, names.waiting) or 0.0, metric_value(b, names.waiting) or 0.0)
        gen_a, gen_b = metric_value(a, names.generation_tokens), metric_value(b, names.generation_tokens)
        gen_delta = (gen_b - gen_a) if gen_a is not None and gen_b is not None else None
        kv_pct = kv_usage_pct(b, names)
        server_busy = running > 0 or waiting > 0
        gpu_loaded = bool(gen_delta and gen_delta > 0) or bool(kv_pct and kv_pct >= 1.0)
        return {
            "checked": True,
            "probe_s": self.load_probe_s,
            "requests_running": running,
            "requests_waiting": waiting,
            "generation_tokens_in_probe": gen_delta,
            "kv_cache_usage_pct": kv_pct,
            "server_busy": server_busy,
            "gpu_loaded": gpu_loaded,
            "loaded": server_busy or gpu_loaded,
        }

    def _check_pre_run_load(self, manifest: RunManifest) -> None:
        """Warn and ask once if something is already running. Never blocks
        unless the operator answers no; the starting load is recorded and
        every row is flagged PRE_EXISTING_LOAD."""
        load = self._pre_run_load()
        manifest.pre_run_load = load
        if not load.get("loaded"):
            self._runlog("pre-run load: idle")
            return
        parts = []
        if load["server_busy"]:
            parts.append(f"server busy: {load['requests_running']:.0f} running, {load['requests_waiting']:.0f} waiting")
        if load["gpu_loaded"]:
            parts.append(
                f"GPU loaded: {load['generation_tokens_in_probe'] or 0:.0f} tokens generated in "
                f"{self.load_probe_s:.0f}s, KV cache {load['kv_cache_usage_pct'] or 0:.1f}%"
            )
        msg = "WARNING: pre-existing load before the run (" + "; ".join(parts) + "). Results will be flagged PRE_EXISTING_LOAD."
        print(msg, file=sys.stderr, flush=True)
        self._runlog(msg)
        if self.assume_yes:
            answer = True
        elif self._confirm_fn is not None:
            answer = self._confirm_fn(msg + " Run anyway? [y/N] ")
        elif sys.stdin is not None and sys.stdin.isatty():
            answer = input("Run anyway? [y/N] ").strip().lower() in ("y", "yes")
        else:
            answer = True  # non-interactive: warn, record, run
        load["confirmed"] = answer
        if not answer:
            raise RunRefused("operator declined to run with pre-existing load (" + "; ".join(parts) + ")", manifest)
        self._pre_load_flag = True
        manifest.notes.append("PRE_EXISTING_LOAD: " + "; ".join(parts))

    def _record_ceiling(self, manifest: RunManifest, rung: Rung, reason: str, detail: str) -> None:
        key = ceiling_key(rung)
        manifest.ctx_ceilings[key] = {
            "family": rung.family,
            "load_mode": rung.load_mode,
            "users": rung.users,
            "max_ok_ctx": manifest.ctx_reached.get(key),
            "stopped_at": rung.ctx_per_user,
            "reason": reason,
            "detail": (detail or "")[:300],
            "rung_id": rung.rung_id,
        }

    def _kv_precheck(self, manifest: RunManifest, run_id: str, rung: Rung, jsonl: Path) -> bool:
        """Before sending anything: if this rung's live KV tokens exceed the
        server's KV capacity, record the user count's ceiling (a CTX_CEILING
        row, nothing sent) and return True."""
        cap = manifest.kv_capacity_tokens
        if self.dry_run or not self.spec.ladder.stop_at_ceiling:
            return False
        need = sum(rung.target_for(s) for s in rung.live_sessions) + len(rung.decoding_sessions) * rung.output_tokens
        if cap and need > cap:
            why = "kv_capacity"
            reason = (
                f"KV need {need:,} tokens ({rung.users} x {rung.ctx_per_user:,} + output) exceeds the server KV "
                f"capacity {cap:,} tokens ({manifest.kv_capacity_source}); not sent"
            )
        else:
            return False
        utc, et = stamp_pair()
        row = RungResult(
            run_id=run_id, rung_id=rung.rung_id, users=rung.users, ctx_per_user=rung.ctx_per_user,
            ctx_tokens=[], prompt_type=rung.prompt_type, load_mode=rung.load_mode, phase=rung.phase,
            prompt_seed=self.spec.prompts.seed, temperature=self.spec.ladder.temperature, k=self.k_declared,
            build_fingerprint=self.fp.build_fingerprint if self.fp else None,
            started_utc=utc, started_local=et, finished_utc=utc, finished_local=et,
            valid="invalid", reason=reason, exit_reason="CTX_CEILING", flags=["CTX_CEILING"],
        )
        write_jsonl_row(jsonl, row)
        self._record_ceiling(manifest, rung, why, reason)
        self._runlog(f"ceiling {ceiling_key(rung)}: {reason}")
        manifest.completed_rung_ids.append(rung.rung_id)
        self._write_manifest(manifest)
        return True

    def _ram_recovered(self, watcher: RamWatcher) -> bool:
        """After a RAM-floor ceiling: MemAvailable back above floor + 0.5 GiB
        (waiting up to --ram-wait minutes if set)."""
        try:
            now = self.ram_source.read_kib() / (1024 * 1024)
        except Exception:  # noqa: BLE001
            return False
        if now >= self.spec.hardware.mem_available_floor_gib + 0.5:
            return True
        return self.ram_wait_min > 0 and self._wait_for_ram(watcher)

    def _read_remote(self, path: str) -> str | None:
        p = path[2:] if path.startswith("~/") else path
        return self.exec.read_file(p)

    def _wait_for_ram(self, watcher: RamWatcher) -> bool:
        """After a floor abort: wait (bounded) for MemAvailable >= floor + 1 GiB."""
        deadline = time.time() + self.ram_wait_min * 60
        target = self.spec.hardware.mem_available_floor_gib + 1.0
        while time.time() < deadline:
            try:
                if self.ram_source.read_kib() / (1024 * 1024) >= target:
                    return True
            except Exception:  # noqa: BLE001
                return False
            self.sleep(5.0)
        return False

    def _restart_watcher(self, old: RamWatcher) -> RamWatcher:
        old.stop()
        w = RamWatcher(floor_gib=old.floor_gib, source=old.source, interval_s=old.interval_s)
        try:
            w.check_now()
        except RamFloorError:
            w = RamWatcher(floor_gib=old.floor_gib, source=old.source, interval_s=old.interval_s)
        if not self.dry_run:
            w.start()
        return w


def run_spec(
    spec: RunSpec,
    output_root: str | Path,
    dry_run: bool = False,
    resume: bool = False,
    variant: str | None = None,
    pause_between_variants: bool = False,
    prompt: Callable[[str], str] | None = None,
    say: Callable[[str], None] | None = None,
    **kwargs: Any,
) -> list[RunManifest]:
    """Run a spec, expanding comparison variants into sequential one-boot runs.

    Each variant must run against its own booted container: pick one with
    ``variant``, give every variant its own ``endpoint.base_url``, or set
    ``pause_between_variants`` to be prompted to reboot the test container
    between variants.
    """
    say = say or (lambda msg: print(msg, flush=True))
    manifests: list[RunManifest] = []
    chosen = list(spec.variants)
    if variant is not None:
        chosen = [v for v in spec.variants if v.name == variant]
        if not chosen:
            names = ", ".join(v.name for v in spec.variants) or "(spec has no variants)"
            raise ValueError(f"unknown variant {variant!r}; choose one of: {names}")
    elif len(chosen) > 1 and not dry_run and not pause_between_variants:
        expanded = [apply_variant(spec, v) for v in chosen]
        urls = [e.endpoint.base_url for e in expanded]
        if len(set(urls)) != len(urls):
            raise RunRefused(
                "this spec has variants that share one endpoint, so they would all hit the same container. "
                "Run one variant at a time with --variant NAME (booting its container first), give each variant "
                "its own endpoint.base_url, or pass --pause-between-variants."
            )
    variants = chosen or [None]
    for i, var in enumerate(variants):
        one = apply_variant(spec, var) if var is not None else spec
        if var is not None and pause_between_variants and not dry_run:
            env = " ".join(f"{k}={v}" for k, v in one.server.env.items()) or "(none)"
            args = " ".join(one.server.launch_args) or "(none)"
            msg = (
                f"\n[{i + 1}/{len(variants)}] Variant {var.name!r}: boot container "
                f"{one.server.container_name!r} at {one.endpoint.base_url} with env {env} and args {args}, "
                "then press Enter (Ctrl-C to stop) "
            )
            (prompt or input)(msg)
        out = Path(output_root) / one.name
        runner = Runner(one, out, dry_run=dry_run, resume=resume, **kwargs)
        manifests.append(runner.run())
    return manifests


def plan_and_explain(spec: RunSpec) -> str:
    from xpu_llm_bench.planner import format_plan

    return format_plan(make_plan(spec))
