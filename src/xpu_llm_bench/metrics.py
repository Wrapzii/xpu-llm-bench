"""Prometheus /metrics scrape, deltas, and MTP accounting."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from xpu_llm_bench.spec import MetricNameMap

# name{labels} value
_LINE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse a Prometheus text exposition.

    Values for the same metric name are summed across label sets, except
    per-position counters which are kept as name{position=N}.
    """
    out: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name = m.group("name")
        labels = m.group("labels") or ""
        value = float(m.group("value"))
        key = name
        if "per_pos" in name and "position=" in labels:
            pos = re.search(r'position="?([^",}]+)"?', labels)
            key = f"{name}{{position={pos.group(1)}}}" if pos else name
        out[key] = out.get(key, 0.0) + value
    return out


def metric_value(snap: Mapping[str, float], name: str) -> float | None:
    if name in snap:
        return float(snap[name])
    # Fall back to prefix match when labels were kept on the key.
    total = 0.0
    found = False
    for key, val in snap.items():
        base = key.split("{", 1)[0]
        if base == name:
            total += float(val)
            found = True
    return total if found else None


def delta(before: Mapping[str, float], after: Mapping[str, float], name: str) -> float | None:
    a = metric_value(after, name)
    b = metric_value(before, name)
    if a is None or b is None:
        return None
    return a - b


@dataclass
class MTPWindow:
    drafts: float | None
    draft_tokens: float | None
    accepted_drafts: float | None
    accepted_drafts_per_step: float | None
    hit_rate: float | None
    tokens_per_step: float | None
    formula: str = (
        "num_accepted_tokens counts accepted drafts only (excludes the bonus / verified token). "
        "tokens_per_step = 1 + accepted_drafts/drafts; "
        "hit_rate = accepted_drafts / draft_tokens."
    )


def mtp_window(before: Mapping[str, float], after: Mapping[str, float], names: MetricNameMap) -> MTPWindow:
    drafts = delta(before, after, names.drafts)
    draft_tokens = delta(before, after, names.draft_tokens)
    # Exclude per-position series — those are a breakdown of the same accepts.
    accepted = delta(before, after, names.accepted_tokens)
    acc_per_step = None
    hit = None
    toks_per_step = None
    if drafts and drafts > 0 and accepted is not None:
        acc_per_step = accepted / drafts
        toks_per_step = 1.0 + acc_per_step
    if draft_tokens and draft_tokens > 0 and accepted is not None:
        hit = accepted / draft_tokens
    return MTPWindow(
        drafts=drafts,
        draft_tokens=draft_tokens,
        accepted_drafts=accepted,
        accepted_drafts_per_step=acc_per_step,
        hit_rate=hit,
        tokens_per_step=toks_per_step,
    )


@dataclass
class SystemWindow:
    preemptions: float | None = None
    prefix_hits: float | None = None
    prefix_queries: float | None = None
    prefix_cache_hit_rate: float | None = None
    kv_cache_usage_pct: float | None = None
    running_reqs: float | None = None
    waiting_reqs: float | None = None
    queue_time_ms: float | None = None
    generation_tokens: float | None = None
    prompt_tokens: float | None = None


def kv_usage_pct(snap: Mapping[str, float], names: MetricNameMap) -> float | None:
    """KV cache usage in percent.

    ``vllm:kv_cache_usage_perc`` (V1) is a 0..1 fraction despite its name and is
    scaled to percent. The legacy ``gpu_cache_usage_perc`` is passed through as-is.
    """
    val = metric_value(snap, names.gpu_cache_usage)
    if val is not None:
        return val * 100.0 if names.gpu_cache_usage.endswith("kv_cache_usage_perc") else val
    for alias in names.gpu_cache_usage_aliases:
        val = metric_value(snap, alias)
        if val is not None:
            return val * 100.0 if alias.endswith("kv_cache_usage_perc") else val
    return None


def system_window(
    before: Mapping[str, float],
    after: Mapping[str, float],
    names: MetricNameMap,
) -> SystemWindow:
    hits = delta(before, after, names.prefix_hits)
    queries = delta(before, after, names.prefix_queries)
    hit_rate = None
    if queries is not None and queries > 0 and hits is not None:
        hit_rate = hits / queries
    q_sum = delta(before, after, names.queue_time_sum)
    q_count = delta(before, after, names.queue_time_count)
    queue_ms = None
    if q_sum is not None and q_count and q_count > 0:
        queue_ms = (q_sum / q_count) * 1000.0
    return SystemWindow(
        preemptions=delta(before, after, names.preemptions),
        prefix_hits=hits,
        prefix_queries=queries,
        prefix_cache_hit_rate=hit_rate,
        kv_cache_usage_pct=kv_usage_pct(after, names),
        running_reqs=metric_value(after, names.running),
        waiting_reqs=metric_value(after, names.waiting),
        queue_time_ms=queue_ms,
        generation_tokens=delta(before, after, names.generation_tokens),
        prompt_tokens=delta(before, after, names.prompt_tokens),
    )


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile. p is in 0..100."""
    if not values:
        return None
    xs = sorted(values)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def inter_token_latencies_ms(token_times: list[float]) -> list[float]:
    """token_times are epoch seconds of each output token (including the first).

    ITL is the gap between consecutive output tokens; TTFT is not included.
    """
    if len(token_times) < 2:
        return []
    ordered = sorted(token_times)
    return [(b - a) * 1000.0 for a, b in zip(ordered, ordered[1:], strict=False)]


@dataclass
class StreamStats:
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    itl_p50_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None
    output_tokens: int = 0
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    decode_tok_s: float | None = None
    prefill_tok_s: float | None = None
    errors: list[str] = field(default_factory=list)
    timed_out: bool = False


def stream_stats(
    t_send: float,
    first_token_at: float | None,
    t_end: float,
    token_times: list[float],
    output_tokens: int,
    prompt_tokens: int | None = None,
    cached_tokens: int | None = None,
    new_prompt_tokens: int | None = None,
) -> StreamStats:
    ttft = (first_token_at - t_send) * 1000.0 if first_token_at is not None else None
    e2e = (t_end - t_send) * 1000.0
    itls = inter_token_latencies_ms(token_times)
    decode_tok_s = None
    if first_token_at is not None and t_end > first_token_at and output_tokens > 1:
        # Remaining tokens after the first, over the decode interval.
        decode_tok_s = (output_tokens - 1) / (t_end - first_token_at)
    elif first_token_at is not None and t_end > first_token_at and output_tokens == 1:
        decode_tok_s = 0.0
    prefill_tok_s = None
    billed = new_prompt_tokens
    if billed is None and prompt_tokens is not None:
        cached = cached_tokens or 0
        billed = max(prompt_tokens - cached, 0)
    if ttft and ttft > 0 and billed:
        prefill_tok_s = billed / (ttft / 1000.0)
    return StreamStats(
        ttft_ms=ttft,
        e2e_ms=e2e,
        itl_p50_ms=percentile(itls, 50),
        itl_p95_ms=percentile(itls, 95),
        itl_p99_ms=percentile(itls, 99),
        output_tokens=output_tokens,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        decode_tok_s=decode_tok_s,
        prefill_tok_s=prefill_tok_s,
    )


# ---------------------------------------------------------------------------
# True hit rate and speculative-counter identity checks
# ---------------------------------------------------------------------------


def true_hit_rate(delivered: float | None, verify_steps: float | None, k: int | None) -> float | None:
    """(delivered - verify_steps) / (k * verify_steps).

    ``delivered`` = decode tokens the client actually received (the first token
    of each request comes from prefill and is excluded). ``verify_steps`` =
    target-model verify passes that produced them. Each verify step yields one
    bonus/verified token plus 0..k accepted drafts, so this is the fraction of
    proposed draft slots that became delivered tokens. It does not read the
    accepted-tokens counter, which some builds over-report (accepted > drafted).
    """
    if delivered is None or verify_steps is None or not k or verify_steps <= 0:
        return None
    return (float(delivered) - float(verify_steps)) / (float(k) * float(verify_steps))


@dataclass
class SpecCheck:
    ok: bool
    k_observed: float | None
    problems: list[str] = field(default_factory=list)


def spec_counter_check(
    mtp: MTPWindow,
    k: int | None,
    delivered: float | None,
    verify_steps_stream: float | None,
    n_requests: int,
    per_pos_sum: float | None = None,
    rel_tol: float = 0.02,
) -> SpecCheck:
    """Identities that must hold if the spec counters are trustworthy.

    T = k*D; A <= T; sum_p A_p = A (when per-position series exist);
    A ~= delivered - D, allowing up to k discarded drafts per request on the
    final step (max_tokens truncation) plus rel_tol of delivered.
    Stream steps vs D is reported separately (STEP_COUNT_MISMATCH).
    """
    problems: list[str] = []
    k_obs = None
    if mtp.drafts and mtp.drafts > 0 and mtp.draft_tokens is not None:
        k_obs = mtp.draft_tokens / mtp.drafts
        if k and abs(k_obs - k) > 1e-6:
            problems.append(f"draft_tokens/drafts={k_obs:.3f} != k={k}")
    if mtp.accepted_drafts is not None and mtp.draft_tokens is not None:
        if mtp.accepted_drafts > mtp.draft_tokens:
            problems.append(
                f"accepted {mtp.accepted_drafts:.0f} > draft_tokens {mtp.draft_tokens:.0f}"
            )
    if per_pos_sum is not None and mtp.accepted_drafts is not None:
        if abs(per_pos_sum - mtp.accepted_drafts) > max(1.0, rel_tol * mtp.accepted_drafts):
            problems.append(
                f"sum(per_pos)={per_pos_sum:.0f} != accepted {mtp.accepted_drafts:.0f}"
            )
    if (
        delivered is not None
        and mtp.drafts is not None
        and mtp.accepted_drafts is not None
        and mtp.drafts > 0
    ):
        implied = delivered - mtp.drafts
        slack = (k or 0) * max(n_requests, 1) + rel_tol * max(delivered, 1.0)
        if mtp.accepted_drafts - implied > slack or implied - mtp.accepted_drafts > rel_tol * max(delivered, 1.0) + 1:
            problems.append(
                f"accepted {mtp.accepted_drafts:.0f} vs delivered-drafts {implied:.0f}"
            )
    return SpecCheck(ok=not problems, k_observed=k_obs, problems=problems)


def per_pos_sum(before: Mapping[str, float], after: Mapping[str, float], name: str) -> float | None:
    total = 0.0
    found = False
    for key, val in after.items():
        if key.split("{", 1)[0] == name and "position=" in key:
            total += float(val) - float(before.get(key, 0.0))
            found = True
    return total if found else None


def histogram_mean_ms(
    before: Mapping[str, float], after: Mapping[str, float], base: str
) -> float | None:
    s = delta(before, after, base + "_sum")
    c = delta(before, after, base + "_count")
    if s is None or not c or c <= 0:
        return None
    return s / c * 1000.0


@dataclass
class ChunkStats:
    """Per-request stream facts used for steps/s and the true hit rate.

    vLLM streams one SSE chunk per engine step per request (stream_interval=1).
    A step can carry 1..k+1 tokens, so the number of token-bearing chunks after
    the first is the request's verify-step count. The first chunk is the
    prefill step.
    """

    tokens: int
    chunks: int
    t_first: float | None
    t_last: float | None

    @property
    def decode_tokens(self) -> int:
        return max(self.tokens - 1, 0)

    @property
    def verify_steps(self) -> int:
        return max(self.chunks - 1, 0)

    @property
    def window_s(self) -> float | None:
        if self.t_first is None or self.t_last is None or self.t_last <= self.t_first:
            return None
        return self.t_last - self.t_first

    @property
    def decode_tok_s(self) -> float | None:
        w = self.window_s
        return self.decode_tokens / w if w else None

    @property
    def steps_s(self) -> float | None:
        w = self.window_s
        return self.verify_steps / w if w else None


@dataclass
class SteadyWindow:
    """The interval in which every session of a window is decoding.

    Sessions are sent together, but each one first prefills, so the start of
    one session's decode overlaps the others' prefill. The steady window is
    ``[max(first token), min(last token)]``: after the last session's first
    token and before the first session's last token, all of them decode.
    ``tokens[i]`` / ``steps[i]`` are session i's tokens and token-bearing
    chunks (verify steps) that arrive inside it, counted the same way as
    ChunkStats (a chunk exactly at the start belongs to the step before).
    """

    t_start: float | None
    t_end: float | None
    tokens: list[int]
    steps: list[int]

    @property
    def window_s(self) -> float | None:
        if self.t_start is None or self.t_end is None or self.t_end <= self.t_start:
            return None
        return self.t_end - self.t_start

    @property
    def min_steps(self) -> int:
        return min(self.steps) if self.steps else 0

    @property
    def tok_s_each(self) -> list[float] | None:
        w = self.window_s
        return [n / w for n in self.tokens] if w else None


def steady_window(streams: list[tuple[list[float], list[int]]]) -> SteadyWindow:
    """Overlap of the sessions' decode intervals.

    ``streams`` holds one ``(token_times, chunk_tokens)`` pair per session:
    the arrival time of each token-bearing chunk and the tokens it carried.
    """
    if not streams or any(not times for times, _ in streams):
        return SteadyWindow(t_start=None, t_end=None, tokens=[0] * len(streams), steps=[0] * len(streams))
    t_start = max(times[0] for times, _ in streams)
    t_end = min(times[-1] for times, _ in streams)
    tokens: list[int] = []
    steps: list[int] = []
    for times, counts in streams:
        inside = [n for t, n in zip(times, counts, strict=False) if t_start < t <= t_end]
        tokens.append(sum(inside))
        steps.append(len(inside))
    return SteadyWindow(t_start=t_start, t_end=t_end, tokens=tokens, steps=steps)
