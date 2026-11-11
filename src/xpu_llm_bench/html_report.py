"""Single-file HTML report with inline SVG charts drawn here.

No JavaScript chart library, no CDN, no fonts or images from the network: the
file opens offline. Each headline metric gets one chart per prompt type and
load mode: X = context length per user (log scale when the ladder spans a
wide range), Y = the metric, one line per concurrency (users). A line ends at
its user count's context ceiling (marker + tooltip with the reason). Flagged
rows are drawn hollow; every point has an SVG <title> tooltip with the exact
value and its flags. Each group shows its quality pass counts, and an
"Average across prompt types" group averages the clean rows.
"""

from __future__ import annotations

import html
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from xpu_llm_bench.results import FLAGS, REPEAT_METRICS, RepeatStat, RungResult, RunManifest

# (field, title, unit, scale, needs_spec_decode)
METRICS: list[tuple[str, str, str, float, bool]] = [
    ("decode_tok_s_per_user", "Decode speed", "tok/s per user", 1.0, False),
    ("prefill_tok_s", "Prefill speed", "tok/s", 1.0, False),
    ("ttft_ms", "Time to first token", "ms", 1.0, False),
    ("steps_s_per_user", "Verify steps", "steps/s per user", 1.0, False),
    ("accepted_drafts_per_step", "MTP accepted drafts per step", "drafts/step", 1.0, True),
    ("hit_rate", "True hit rate", "%", 100.0, True),
]
# Flags that make a point hollow (the number is suspect or not what it says).
HOLLOW_FLAGS = {"NOT_AS_LABELED", "QSA_FALLBACK", "MIXED_PATH", "GRAPH_MISS", "HIT_RATE_UNKNOWN",
                "SPEC_COUNTER_BROKEN", "DRY_RUN"}
TABLE_COLUMNS = [
    "rung_id", "phase", "users", "ctx_per_user", "prompt_type", "load_mode", "attention_path",
    "decode_tok_s_per_user", "steps_s_per_user", "decode_tok_s_total", "decode_tok_s_per_user_steady",
    "decode_tok_s_total_steady", "tokens_per_step", "accepted_drafts_per_step", "hit_rate", "prefill_tok_s", "ttft_ms", "tpot_ms", "itl_p50_ms", "itl_p99_ms",
    "kv_cache_usage_max_pct", "mem_available_min_gib", "exit_reason", "flags", "valid",
]
# Plain-words names for group headings.
PROMPT_LABELS = {"code_edits": "Code edits", "long_form": "Long-form writing", "multi_turn_chat": "Multi-turn chat"}
LOAD_LABELS = {"closed_all_decoding": "closed loop", "staggered": "staggered arrivals", "open_loop": "open loop"}
# Metrics averaged across prompt types: (field, column title, scale).
AVG_METRICS: list[tuple[str, str, float]] = [
    ("decode_tok_s_per_user", "decode tok/s per user", 1.0),
    ("steps_s_per_user", "steps/s per user", 1.0),
    ("accepted_drafts_per_step", "accepted drafts/step", 1.0),
    ("hit_rate", "true hit rate %", 100.0),
    ("ttft_ms", "TTFT ms", 1.0),
    ("prefill_tok_s", "prefill tok/s", 1.0),
]
AVG_LABEL = "average"
# Series colours by users (works on light and dark backgrounds).
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
W, H = 520, 312
ML, MR, MT, MB = 64, 22, 16, 58

CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1d1d1f;--muted:#6b6b70;--grid:#e3e3e8;--card:#f6f6f8;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf1;--muted:#a0a0aa;--grid:#33333a;--card:#1d1d21;--bad:#ff8a80}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;margin:24px;max-width:1700px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px}h3{font-size:15px;margin:16px 0 6px}
.muted{color:var(--muted)}.bad{color:var(--bad)}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid var(--grid);padding:3px 6px;text-align:left;vertical-align:top}
th{background:var(--card);position:sticky;top:0}
.hdr td:first-child{font-weight:600;white-space:nowrap}
.grid{display:flex;flex-wrap:wrap;gap:14px}.card{background:var(--card);border-radius:8px;padding:8px 10px}
.card h3{margin:2px 0 4px}
svg text{fill:var(--fg);font-size:11px}svg .axis{stroke:var(--muted)}svg .gl{stroke:var(--grid)}
pre{white-space:pre-wrap;word-break:break-all;background:var(--card);padding:8px;border-radius:6px;font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin:4px 0 8px;font-size:12px}
.legend span{display:inline-flex;align-items:center;gap:4px}
.note{background:var(--card);border-left:3px solid var(--muted);padding:6px 10px;margin:6px 0}
.wide{overflow-x:auto}
.qline{font-size:13px;margin:2px 0 8px}.qline b{font-weight:600}
.acards{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;margin:8px 0 14px}
.acard{background:var(--card);border-radius:8px;padding:8px 10px;border-top:3px solid var(--muted)}
.acard.partial{opacity:.85;border-top-style:dashed}
.ac-h{font-weight:600;font-size:13px}.ac-of{font-size:11px;color:var(--muted);margin:1px 0 4px}
.ac-big{display:flex;gap:14px;margin:2px 0}.ac-big div{display:flex;flex-direction:column}
.ac-big b{font-size:20px;line-height:1.1}.ac-big span{font-size:10px;color:var(--muted)}
.ac-small{font-size:11px;margin:3px 0 4px}
.ac-t{font-size:11px;width:100%}.ac-t td,.ac-t th{padding:1px 4px}
svg .ceil{stroke:var(--bad);stroke-width:2}
"""
# Only added when a run used repeats, so default output stays byte-identical.
CSS_REPEATS = "svg .ebar{stroke-width:1.5;opacity:.75}\n.ac-big em{font-style:normal;font-size:10px;color:var(--muted)}\n"


def group_label(prompt_type: str, load_mode: str) -> str:
    """'Code edits, closed loop' rather than 'code_edits · closed_all_decoding'."""
    if prompt_type == AVG_LABEL:
        return f"Average across prompt types, {LOAD_LABELS.get(load_mode, load_mode)}"
    return f"{PROMPT_LABELS.get(prompt_type, prompt_type.replace('_', ' ').capitalize())}, " \
        f"{LOAD_LABELS.get(load_mode, load_mode.replace('_', ' '))}"


# Quality checks: (label, getter returning True/False or None when the check did not run, suffix style)
def _greedy(r: RungResult) -> bool | None:
    g = r.quality.greedy_ref
    if g in (None, "no_reference"):
        return None
    return g == "match"


QUALITY_CHECKS: list[tuple[str, Any, str]] = [
    ("Code edit", lambda r: r.quality.code_edit_pass, "pass"),
    ("Needle", lambda r: r.quality.needle_pass, "pass"),
    ("Format", lambda r: r.quality.format_ok, "pass"),
    ("Greedy match", _greedy, ""),
    ("Logprob", lambda r: r.quality.logprob_ok, "ok"),
]


def quality_counts(rows: list[RungResult]) -> list[tuple[str, int, int, str]]:
    """(check, passed, ran, style) for every check that ran on at least one row (None = not run)."""
    out = []
    for label, get, style in QUALITY_CHECKS:
        vals = [get(r) for r in rows]
        ran = [v for v in vals if v is not None]
        if ran:
            out.append((label, sum(1 for v in ran if v), len(ran), style))
    return out


def quality_line(rows: list[RungResult]) -> str:
    """'Code edit: 5 of 5 pass · Needle: 9 of 10 pass · ... · Logprob: ok 3 of 3' (text)."""
    parts = []
    for label, ok, ran, style in quality_counts(rows):
        if style == "ok":
            parts.append(f"{label}: ok {ok} of {ran}")
        elif style:
            parts.append(f"{label}: {ok} of {ran} {style}")
        else:
            parts.append(f"{label}: {ok} of {ran}")
    return " · ".join(parts) if parts else "no quality checks ran"


def _qline_html(rows: list[RungResult], prefix: str = "Quality") -> str:
    bad = any(ok < ran for _, ok, ran, _ in quality_counts(rows))
    return (f'<p class="qline{" bad" if bad else ""}"><b>{_e(prefix)}:</b> {_e(quality_line(rows))}</p>')


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def ctx_label(n: int | None) -> str:
    if n is None:
        return "?"
    if n >= 1024 and n % 1024 == 0:
        k = n // 1024
        # Project convention: 131072 reads "131K", 196608 "196K", 262144
        # "262K" (thousands, rounded down); 8192 -> 8K, 65536 -> 64K.
        return f"{n // 1000}K" if k >= 100 else f"{k}K"
    if n >= 102400:
        return f"{n // 1000}K"
    return f"{n / 1024:.1f}K" if n >= 1024 else str(n)


def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _tick(v: float, step: float) -> str:
    decimals = max(0, -math.floor(math.log10(step))) if step < 1 else 0
    if step < 1 and round(step * 10 ** decimals) % 10 and step * 10 ** decimals % 1:
        decimals += 1
    return f"{v:,.{decimals}f}"


def _nice_max(v: float) -> tuple[float, float]:
    """Axis top and tick step for data maximum v (>0)."""
    if v <= 0:
        return 1.0, 0.25
    raw = v / 5
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return step * math.ceil(v / step * 1.0001), step


@dataclass
class Point:
    ctx: int
    value: float
    flags: list[str]
    rung_ids: list[str]
    valid: str
    lo: float | None = None  # min across repeats (only when repeats > 1)
    hi: float | None = None
    n: int = 1


def repeat_range(r: RungResult, field: str) -> RepeatStat | None:
    """The row's repeat stats for ``field`` when it was measured more than once."""
    if (r.repeats or 1) <= 1 or not r.repeat_stats:
        return None
    st = r.repeat_stats.get(field)
    if st is None or st.n <= 1 or st.min is None or st.max is None:
        return None
    return st


def fmt_repeat(st: RepeatStat, scale: float = 1.0, unit: str = "") -> str:
    """'median (min to max, n=N)'. Percentages keep one decimal like the plain cells."""
    def f(v: float) -> str:
        return f"{v * scale:.1f}" if unit == "%" else _fmt(v * scale)
    return f"{f(st.median)}{unit} ({f(st.min)} to {f(st.max)}{unit}, n={st.n})"


def _series(rows: list[RungResult], field: str, scale: float) -> dict[int, list[Point]]:
    """users -> points ordered by context. Duplicate (users, ctx) rows are averaged."""
    acc: dict[tuple[int, int], list[RungResult]] = defaultdict(list)
    for r in rows:
        if r.ctx_per_user is None or "CTX_CEILING" in r.flags and r.exit_reason == "CTX_CEILING":
            continue
        acc[(r.users, r.ctx_per_user)].append(r)
    out: dict[int, list[Point]] = defaultdict(list)
    for (users, ctx), group in sorted(acc.items()):
        vals = [getattr(r, field) for r in group]
        nums = [float(v) * scale for v in vals if isinstance(v, (int, float))]
        if not nums:
            continue
        flags = sorted({f for r in group for f in r.flags})
        valid = "valid" if all(r.valid == "valid" for r in group) else "invalid"
        pt = Point(ctx, sum(nums) / len(nums), flags, [r.rung_id for r in group], valid)
        rngs = [st for st in (repeat_range(r, field) for r in group) if st is not None]
        if rngs:
            pt.lo = min(st.min for st in rngs) * scale  # type: ignore[operator]
            pt.hi = max(st.max for st in rngs) * scale  # type: ignore[operator]
            pt.n = max(st.n for st in rngs)
        out[users].append(pt)
    return out


def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def svg_chart(
    runs: list[tuple[str, list[RungResult]]],
    field: str,
    title: str,
    unit: str,
    scale: float,
    users_order: list[int],
    ceilings: dict[int, dict[str, Any]] | None = None,
) -> str:
    """One chart. ``runs`` holds (label, rows); run 0 is solid, others dashed.

    X is the context rungs that ran: log-scaled when the largest is 8x the
    smallest or more, otherwise evenly spaced. ``ceilings`` (users -> ceiling
    info) puts a marker at the end of that user count's line (run 0)."""
    data = [(label, _series(rows, field, scale)) for label, rows in runs]
    ctxs = sorted({p.ctx for _, s in data for pts in s.values() for p in pts})
    vals = [v for _, s in data for pts in s.values() for p in pts for v in (p.value, p.hi) if v is not None]
    head = f'<div class="card"><h3>{_e(title)} <span class="muted">({_e(unit)})</span></h3>'
    if not ctxs:
        return head + '<p class="muted">No data for this metric.</p></div>'
    top, step = _nice_max(max(vals))
    pw, ph = W - ML - MR, H - MT - MB

    log_x = len(ctxs) > 1 and ctxs[-1] / max(ctxs[0], 1) >= 8

    def x(ctx: int) -> float:
        if len(ctxs) == 1:
            return ML + pw / 2
        if log_x:
            lo, hi = math.log2(ctxs[0]), math.log2(ctxs[-1])
            return ML + pw * (math.log2(ctx) - lo) / (hi - lo)
        return ML + pw * ctxs.index(ctx) / (len(ctxs) - 1)

    gaps = [x(b) - x(a) for a, b in zip(ctxs, ctxs[1:], strict=False)]
    rotate = bool(gaps) and min(gaps) < 34

    def y(v: float) -> float:
        return MT + ph * (1 - v / top)

    parts = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
             f'aria-label="{_e(title)} by context length, one line per user count">']
    t = 0.0
    while t <= top * 1.0001:
        parts.append(f'<line class="gl" x1="{ML}" x2="{W - MR}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        parts.append(f'<text x="{ML - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{_e(_tick(t, step))}</text>')
        t += step
    parts.append(f'<line class="axis" x1="{ML}" x2="{ML}" y1="{MT}" y2="{MT + ph}"/>')
    parts.append(f'<line class="axis" x1="{ML}" x2="{W - MR}" y1="{MT + ph}" y2="{MT + ph}"/>')
    for c in ctxs:
        if rotate:
            parts.append(f'<text transform="translate({x(c) + 4:.1f} {MT + ph + 12}) rotate(-40)" '
                         f'text-anchor="end">{ctx_label(c)}</text>')
        else:
            parts.append(f'<text x="{x(c):.1f}" y="{MT + ph + 16}" text-anchor="middle">{ctx_label(c)}</text>')
    parts.append(f'<text x="{ML + pw / 2:.1f}" y="{H - 6}" text-anchor="middle">context per user (tokens'
                 f'{", log scale" if log_x else ""})</text>')
    parts.append(f'<text transform="translate(14 {MT + ph / 2:.1f}) rotate(-90)" text-anchor="middle">'
                 f'{_e(title)} ({_e(unit)})</text>')
    for ri, (label, series) in enumerate(data):
        dash = "" if ri == 0 else ' stroke-dasharray="6 4"'
        for users in users_order:
            pts = sorted(series.get(users, []), key=lambda p: ctxs.index(p.ctx))
            if not pts:
                continue
            col = _color(users_order.index(users))
            name = f"{users} user{'s' if users != 1 else ''}" + (f" [{label}]" if len(data) > 1 else "")
            cls = f"s u{users} r{ri}"
            if len(pts) > 1:
                d = " ".join(f"{'M' if i == 0 else 'L'}{x(p.ctx):.1f},{y(p.value):.1f}" for i, p in enumerate(pts))
                parts.append(f'<path class="{cls}" data-users="{users}" d="{d}" fill="none" stroke="{col}" '
                             f'stroke-width="2"{dash}><title>{_e(name)}</title></path>')
            for p in pts:
                hollow = bool(HOLLOW_FLAGS & set(p.flags)) or p.valid != "valid"
                fill = "none" if hollow else col
                tip = f"{name}, {ctx_label(p.ctx)} context: {_fmt(p.value)} {unit}"
                if len(p.rung_ids) > 1:
                    tip += f" (mean of {len(p.rung_ids)} rows)"
                if p.lo is not None and p.hi is not None:
                    tip += f"\nmedian of {p.n} repeats; min to max {_fmt(p.lo)} to {_fmt(p.hi)} {unit}"
                    xc, ylo, yhi = x(p.ctx), y(p.lo), y(p.hi)
                    parts.append(f'<g class="ebar" data-users="{users}" stroke="{col}">'
                                 f'<line x1="{xc:.1f}" x2="{xc:.1f}" y1="{yhi:.1f}" y2="{ylo:.1f}"/>'
                                 f'<line x1="{xc - 4:.1f}" x2="{xc + 4:.1f}" y1="{yhi:.1f}" y2="{yhi:.1f}"/>'
                                 f'<line x1="{xc - 4:.1f}" x2="{xc + 4:.1f}" y1="{ylo:.1f}" y2="{ylo:.1f}"/>'
                                 f'<title>{_e(tip)}</title></g>')
                if p.flags:
                    tip += "\nflags: " + ", ".join(p.flags)
                if p.valid != "valid":
                    tip += "\nrow not valid"
                tip += "\nrung: " + ", ".join(p.rung_ids)
                shape = (f'<rect x="{x(p.ctx) - 4:.1f}" y="{y(p.value) - 4:.1f}" width="8" height="8"'
                         if ri else f'<circle cx="{x(p.ctx):.1f}" cy="{y(p.value):.1f}" r="4"')
                parts.append(f'{shape} class="pt{" hollow" if hollow else ""}" data-users="{users}" fill="{fill}" '
                             f'stroke="{col}" stroke-width="2"><title>{_e(tip)}</title>'
                             f'{"</rect>" if ri else "</circle>"}')
            ceil = (ceilings or {}).get(users) if ri == 0 else None
            if ceil:
                last = pts[-1]
                cx, cy = x(last.ctx), y(last.value)
                tip = (f"{name}: context ceiling. Last measured {ctx_label(last.ctx)}"
                       + (f"; stopped at {ctx_label(ceil['stopped_at'])}" if ceil.get("stopped_at") else "")
                       + f" ({ceil.get('reason') or 'unknown'})"
                       + (f"\n{ceil['detail']}" if ceil.get("detail") else ""))
                parts.append(f'<g class="ceil-mark" data-users="{users}"><title>{_e(tip)}</title>'
                             f'<line class="ceil" x1="{cx + 7:.1f}" x2="{cx + 7:.1f}" y1="{cy - 9:.1f}" '
                             f'y2="{cy + 9:.1f}" stroke="{col}"/>'
                             f'<line class="ceil" x1="{cx + 3:.1f}" x2="{cx + 11:.1f}" y1="{cy - 9:.1f}" '
                             f'y2="{cy - 9:.1f}" stroke="{col}"/>'
                             f'<line class="ceil" x1="{cx + 3:.1f}" x2="{cx + 11:.1f}" y1="{cy + 9:.1f}" '
                             f'y2="{cy + 9:.1f}" stroke="{col}"/></g>')
    parts.append("</svg>")
    return head + "".join(parts) + "</div>"


def _legend(users_order: list[int], labels: list[str], caption: str | None = None, repeats: int = 1) -> str:
    items = [f"<span><b>{_e(caption)}</b></span>"] if caption else []
    for i, u in enumerate(users_order):
        items.append(f'<span><svg width="22" height="10"><line x1="0" x2="22" y1="5" y2="5" stroke="{_color(i)}" '
                     f'stroke-width="3"/></svg>{u} user{"s" if u != 1 else ""}</span>')
    if len(labels) > 1:
        for ri, lab in enumerate(labels):
            dash = "" if ri == 0 else ' stroke-dasharray="6 4"'
            items.append(f'<span><svg width="22" height="10"><line x1="0" x2="22" y1="5" y2="5" stroke="currentColor" '
                         f'stroke-width="2"{dash}/></svg>{_e(lab)} ({"solid, circles" if ri == 0 else "dashed, squares"})'
                         "</span>")
    items.append('<span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="none" stroke="currentColor" '
                 'stroke-width="2"/></svg>hollow = flagged or invalid row (hover for flags)</span>')
    items.append('<span><svg width="14" height="20"><g stroke="currentColor" stroke-width="2"><line x1="7" x2="7" '
                 'y1="1" y2="19"/><line x1="3" x2="11" y1="1" y2="1"/><line x1="3" x2="11" y1="19" y2="19"/></g>'
                 '</svg>line ends at that user count\'s context ceiling (hover for the reason)</span>')
    if repeats > 1:
        items.append('<span><svg width="10" height="20"><g stroke="currentColor" stroke-width="1.5"><line x1="5" x2="5" '
                     'y1="2" y2="18"/><line x1="1" x2="9" y1="2" y2="2"/><line x1="1" x2="9" y1="18" y2="18"/></g>'
                     f'</svg>point = median of {repeats} repeats, bar = min to max</span>')
    return '<div class="legend">' + "".join(items) + "</div>"


def ctx_ceilings(rows: list[RungResult], manifest: RunManifest | None) -> dict[tuple[str, str, int], dict[str, Any]]:
    """(prompt type, load mode, users) -> {max_ok_ctx, stopped_at, reason, detail}.

    From manifest.ctx_ceilings, else from rows flagged CTX_CEILING."""
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for v in ((manifest.ctx_ceilings if manifest else {}) or {}).values():
        out[(v.get("family"), v.get("load_mode"), int(v.get("users") or 0))] = dict(v)
    for r in rows:
        key = (r.prompt_type, r.load_mode, r.users)
        if "CTX_CEILING" in r.flags and key not in out:
            ok = [x.ctx_per_user for x in rows if (x.prompt_type, x.load_mode, x.users) == key
                  and x.valid != "invalid" and x.ctx_per_user < r.ctx_per_user]
            out[key] = {"max_ok_ctx": max(ok) if ok else None, "stopped_at": r.ctx_per_user,
                        "reason": (r.exit_reason or "").lower(), "detail": r.reason}
    return out


def _clean(r: RungResult) -> bool:
    """Rows used for averages: valid and carrying no flag at all."""
    return r.valid == "valid" and not r.flags


def _type_label(pt: str) -> str:
    return PROMPT_LABELS.get(pt, pt.replace("_", " ").capitalize())


def _exclusion_reason(rs: list[RungResult]) -> str:
    """Why a prompt type's rows at one (users, context) cell are left out of the average."""
    if not rs:
        return "no row"
    bits = []
    for r in rs:
        if r.valid != "valid" and r.valid != "warning":
            bits.append(f"invalid ({r.exit_reason or 'failed'})")
        if r.flags:
            bits.append("flagged " + ", ".join(r.flags))
        elif r.valid == "warning":
            bits.append("warning row")
    return "; ".join(dict.fromkeys(bits)) or "not clean"


def average_rows(rows: list[RungResult]) -> tuple[list[RungResult], list[dict[str, Any]]]:
    """Average clean rows across prompt types, per load mode, users and context.

    Only (load mode, users, context) cells where EVERY prompt type of that load
    mode has a clean row are averaged (``averaged`` true). Cells where at least
    one type is clean but another is not are returned too (``averaged`` false)
    with the excluded types and why, so the report can name them; they are not
    averaged. Returns synthetic rows for the averaged cells (prompt_type
    "average", for the chart group) and one record per cell."""
    types_by_mode: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.prompt_type != AVG_LABEL:
            types_by_mode[r.load_mode].add(r.prompt_type)
    every: dict[tuple[str, int, int], dict[str, list[RungResult]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.prompt_type != AVG_LABEL:
            every[(r.load_mode, r.users, r.ctx_per_user)][r.prompt_type].append(r)
    avg_rows: list[RungResult] = []
    table: list[dict[str, Any]] = []
    for (mode, users, ctx), all_by_type in sorted(every.items()):
        types = sorted(types_by_mode[mode])
        if len(types) < 2:
            continue
        clean = {t: [x for x in all_by_type.get(t, []) if _clean(x)] for t in types}
        clean = {t: rs for t, rs in clean.items() if rs}
        if not clean:
            continue
        excluded = {t: _exclusion_reason(all_by_type.get(t, [])) for t in types if t not in clean}
        per_type: dict[str, dict[str, float | None]] = {}
        type_rng: dict[str, dict[str, RepeatStat]] = {}
        for t, rs in clean.items():
            per_type[t] = {}
            type_rng[t] = {}
            for field, _title, _scale in AVG_METRICS:
                nums = [float(getattr(x, field)) for x in rs if isinstance(getattr(x, field), (int, float))]
                per_type[t][field] = sum(nums) / len(nums) if nums else None
                sts = [st for st in (repeat_range(x, field) for x in rs) if st is not None]
                if sts and per_type[t][field] is not None and len(sts) == len(rs):
                    type_rng[t][field] = RepeatStat(
                        n=min(st.n for st in sts), median=per_type[t][field],
                        min=sum(st.min for st in sts) / len(sts),  # type: ignore[misc]
                        max=sum(st.max for st in sts) / len(sts),  # type: ignore[misc]
                    )
        averaged = not excluded
        rec: dict[str, Any] = {"load_mode": mode, "users": users, "ctx": ctx, "types": types,
                               "averaged": averaged, "breakdown": per_type, "excluded": excluded,
                               "breakdown_ranges": type_rng}
        vals: dict[str, float | None] = {}
        avg_rng: dict[str, RepeatStat] = {}
        for field, _title, _scale in AVG_METRICS:
            got = [per_type[t][field] for t in types if t in per_type and per_type[t][field] is not None]
            vals[field] = sum(got) / len(got) if averaged and len(got) == len(types) else None
            # Range of the averaged value: mean of the per-type minima to mean of the maxima.
            rg = [type_rng.get(t, {}).get(field) for t in types]
            if vals[field] is not None and all(rg):
                avg_rng[field] = RepeatStat(
                    n=min(r.n for r in rg), median=vals[field],  # type: ignore[union-attr]
                    min=sum(r.min for r in rg) / len(rg), max=sum(r.max for r in rg) / len(rg),  # type: ignore[misc,union-attr]
                )
        rec.update(vals)
        rec["ranges"] = avg_rng
        table.append(rec)
        if averaged:
            avg_rows.append(RungResult.model_construct(
                run_id="average",
                rung_id=f"average of {' · '.join(_type_label(t) for t in types)} ({users}u, {ctx_label(ctx)})",
                users=users, ctx_per_user=ctx, ctx_tokens=[], prompt_type=AVG_LABEL, load_mode=mode,
                flags=[], valid="valid", exit_reason=None, **vals,
                repeats=max((r.n for r in avg_rng.values()), default=1), repeat_stats=avg_rng,
            ))
    return avg_rows, table


def _avg_val(v: float | None, scale: float = 1.0) -> str:
    return _fmt(v * scale) if isinstance(v, (int, float)) else "n/a"


def _rng_txt(rng: RepeatStat | None, scale: float = 1.0, unit: str = "") -> str:
    """' (min to max, n=N)' after a value measured more than once; '' otherwise."""
    if rng is None or rng.min is None or rng.max is None or rng.n <= 1:
        return ""
    return f" ({_fmt(rng.min * scale)} to {_fmt(rng.max * scale)}{unit}, n={rng.n})"


def averages_cards(table: list[dict[str, Any]]) -> str:
    """One compact card per (load mode, users, context) cell."""
    if not table:
        return '<p class="muted">No (users, context) cell has a clean row, so there is nothing to average.</p>'
    cards = []
    for rec in table:
        users, ctx, types = rec["users"], rec["ctx"], rec["types"]
        head = (f'<div class="ac-h">{users} user{"s" if users != 1 else ""} · {_e(ctx_label(ctx))}'
                f'<span class="muted"> · {_e(LOAD_LABELS.get(rec["load_mode"], rec["load_mode"]))}</span></div>')
        if rec["averaged"]:
            of = "Avg of " + " · ".join(_type_label(t) for t in types)
            rg = rec.get("ranges") or {}

            def _em(field: str, rg: dict = rg) -> str:
                t = _rng_txt(rg.get(field))
                return f"<em>{_e(t.strip())}</em>" if t else ""

            big = (f'<div class="ac-big"><div><b>{_avg_val(rec["steps_s_per_user"])}</b>{_em("steps_s_per_user")}'
                   f'<span>steps/s per user</span></div><div><b>{_avg_val(rec["decode_tok_s_per_user"])}</b>'
                   f'{_em("decode_tok_s_per_user")}<span>decode tok/s per user</span></div></div>')
            small = (f'<div class="ac-small">drafts/step {_avg_val(rec["accepted_drafts_per_step"])}'
                     f'{_e(_rng_txt(rg.get("accepted_drafts_per_step")))} · '
                     f'hit rate {_avg_val(rec["hit_rate"], 100.0)}%{_e(_rng_txt(rg.get("hit_rate"), 100.0, "%"))} · '
                     f'TTFT {_avg_val(rec["ttft_ms"])} ms{_e(_rng_txt(rg.get("ttft_ms")))} · '
                     f'prefill {_avg_val(rec["prefill_tok_s"])} tok/s{_e(_rng_txt(rg.get("prefill_tok_s")))}</div>')
            cls = "acard"
        else:
            kept = [t for t in types if t in rec["breakdown"]]
            of = "Not averaged: " + " · ".join(_type_label(t) for t in kept) + " only"
            big = '<div class="ac-big muted"><div><b>—</b><span>needs a clean row in every prompt type</span></div></div>'
            small = ""
            cls = "acard partial"
        rows_html = []
        for t in types:
            b = rec["breakdown"].get(t)
            if b is not None:
                br = (rec.get("breakdown_ranges") or {}).get(t, {})
                rows_html.append(f'<tr><td>{_e(_type_label(t))}</td><td>{_avg_val(b["steps_s_per_user"])}'
                                 f'{_e(_rng_txt(br.get("steps_s_per_user")))}</td>'
                                 f'<td>{_avg_val(b["decode_tok_s_per_user"])}'
                                 f'{_e(_rng_txt(br.get("decode_tok_s_per_user")))}</td></tr>')
            else:
                rows_html.append(f'<tr class="bad"><td>{_e(_type_label(t))}</td><td colspan="2">excluded: '
                                 f'{_e(rec["excluded"].get(t, ""))}</td></tr>')
        breakdown = ('<table class="ac-t"><thead><tr><th>prompt type</th><th>steps/s</th><th>tok/s</th></tr></thead>'
                     f'<tbody>{"".join(rows_html)}</tbody></table>')
        cards.append(f'<div class="{cls}" data-users="{users}" data-ctx="{ctx}">{head}'
                     f'<div class="ac-of">{_e(of)}</div>{big}{small}{breakdown}</div>')
    return f'<div class="acards">{"".join(cards)}</div>'


def has_spec_decode(rows: list[RungResult], manifest: RunManifest | None) -> bool:
    if manifest is not None:
        sp = manifest.speculative
        if manifest.k or (sp is not None and getattr(sp, "method", None)):
            return True
    return any(r.k or r.accepted_drafts_per_step is not None or r.hit_rate is not None for r in rows)


def _cell(r: RungResult, col: str) -> str:
    v = getattr(r, col, None)
    if col in REPEAT_METRICS:
        st = repeat_range(r, col)
        if st is not None:
            return fmt_repeat(st, 100.0, "%") if col == "hit_rate" else fmt_repeat(st)
    if col == "hit_rate" and isinstance(v, (int, float)):
        return f"{v * 100:.1f}%"
    if isinstance(v, float):
        return _fmt(v)
    if isinstance(v, list):
        return ", ".join(map(str, v))
    return "" if v is None else str(v)


def _stdev_title(r: RungResult, col: str) -> str:
    st = repeat_range(r, col) if col in REPEAT_METRICS else None
    if st is None or st.stdev is None:
        return ""
    sc = 100.0 if col == "hit_rate" else 1.0
    return f' title="stdev {_e(_fmt(st.stdev * sc))} over {st.n} repeats"'


def numbers_table(rows: list[RungResult]) -> str:
    head = "".join(f"<th>{_e(c)}</th>" for c in TABLE_COLUMNS)
    body = []
    for r in rows:
        cls = ' class="bad"' if (HOLLOW_FLAGS & set(r.flags)) or r.valid != "valid" else ""
        body.append(f"<tr{cls}>" + "".join(f"<td{_stdev_title(r, c)}>{_e(_cell(r, c))}</td>" for c in TABLE_COLUMNS)
                    + "</tr>")
    return f'<div class="wide"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def context_reach(rows: list[RungResult], manifest: RunManifest | None) -> str:
    """Per load mode and users: highest context measured (all prompt types) and the ceiling, if any."""
    ceil = ctx_ceilings(rows, manifest)
    top = max(((manifest.context_rungs if manifest else None) or [r.ctx_per_user for r in rows]), default=None)
    keys = sorted({(r.load_mode, r.users) for r in rows if r.prompt_type != AVG_LABEL})
    if not keys or top is None:
        return ""
    body = []
    for mode, users in keys:
        mine = [r for r in rows if (r.load_mode, r.users) == (mode, users) and r.valid != "invalid"
                and not (r.exit_reason == "CTX_CEILING")]
        reached = max((r.ctx_per_user for r in mine), default=None)
        stops = sorted({(v.get("stopped_at"), v.get("reason")) for (pt, lm, u), v in ceil.items()
                        if lm == mode and u == users})
        if stops:
            what = "; ".join(f"stopped at {ctx_label(s)} ({rs})" for s, rs in stops)
        elif reached == top:
            what = "reached the top rung"
        else:
            what = "did not reach the top rung (run incomplete)"
        body.append(f"<tr><td>{_e(LOAD_LABELS.get(mode, mode))}</td><td>{users}</td>"
                    f"<td>{_e(ctx_label(reached) if reached else 'none')}</td><td>{_e(what)}</td></tr>")
    return ("<table><thead><tr><th>load</th><th>users</th><th>max context measured</th><th>ceiling</th></tr>"
            f"</thead><tbody>{''.join(body)}</tbody></table>")


def _kv_table(items: list[tuple[str, Any]]) -> str:
    return '<table class="hdr">' + "".join(
        f"<tr><td>{_e(k)}</td><td>{v if isinstance(v, Markup) else _e(v)}</td></tr>" for k, v in items
    ) + "</table>"


class Markup(str):
    """Already-escaped HTML."""


def _header(rows: list[RungResult], manifest: RunManifest | None) -> str:
    out: list[str] = []
    m = manifest
    r0 = rows[0] if rows else None
    if m is None:
        out.append('<p class="note">No manifest.json next to the results: run fingerprint unknown '
                   "(imported or copied rows).</p>")
    fp = (m.fingerprint if m else {}) or {}
    boot = (m.boot if m else {}) or {}
    launch = (m.launch if m else {}) or {}
    att = m.attention if m else None
    graph = m.graph if m else None
    sp = m.speculative if m else None
    hw = m.hardware if m else None
    caps = (graph.capture_sizes if graph and getattr(graph, "capture_sizes", None) else None) or (
        r0.capture_sizes if r0 else None)
    att_txt = "unknown"
    if att is not None:
        att_txt = f"{att.actual} (requested {att.requested}; from boot/runtime log lines, {len(att.evidence_lines)} quoted)"
    elif r0 is not None:
        att_txt = f"{r0.attention_path} (from rows)"
    graphs_txt = "unknown"
    if graph is not None:
        graphs_txt = (f"on={graph.graphs_on} (basis {graph.graphs_basis}; declared {graph.graphs_declared}), "
                      f"mode={graph.mode}, capture sizes={caps}")
    items: list[tuple[str, Any]] = []
    if m:
        items += [
            ("run", f"{m.run_id} — spec {m.spec_name} / profile {m.profile} — status {m.status}"
             + (f" ({m.abort_reason})" if m.abort_reason else "")),
            ("started", f"{m.started_utc} UTC / {m.started_local} ({m.display_timezone or 'local'})"),
            ("finished", f"{m.finished_utc or 'n/a'} UTC / {m.finished_local or 'n/a'}"),
        ]
    items += [
        ("image", f"{fp.get('image_tag') or (m.image if m else None) or 'n/a'}"
         f"{' — digest ' + ', '.join(fp.get('repo_digests') or []) if fp.get('repo_digests') else ''}"
         f"{' — id ' + str(fp.get('image_id'))[:19] if fp.get('image_id') else ''}"),
        ("build fingerprint", fp.get("build_fingerprint") or (r0.build_fingerprint if r0 else None) or "n/a"),
        ("attention path", att_txt),
        ("KV format", f"{boot.get('kv_dtype') or (m.kv_format if m else None) or (r0.kv_format if r0 else None) or 'unknown'}"
         + (f" ({boot.get('kv_tokens')} KV tokens)" if boot.get("kv_tokens") else "")),
        ("graphs", graphs_txt),
        ("speculative decoding", (f"{getattr(sp, 'method', None)} k={m.k if m else None}" if sp else "none declared")
         + (f"; boot log: {boot.get('spec_method')} k={boot.get('spec_k')}" if boot.get("spec_method") else "")),
    ]
    if m and (m.max_model_len or m.context_rungs):
        items.append(("context ladder",
                      f"max_model_len {m.max_model_len or 'unknown'}"
                      + (f" (from {m.max_model_len_source})" if m.max_model_len_source else "")
                      + (f"; rungs {', '.join(ctx_label(c) for c in m.context_rungs)}" if m.context_rungs else "")
                      + (f"; {m.context_note}" if m.context_note else "")))
    n_rep = max((r.repeats or 1 for r in rows), default=1)
    if n_rep > 1:
        items.append(("repeats", f"{n_rep} decode windows per rung on the same context; numbers are medians with "
                      "(min to max, n=N); quality checks ran once per rung; prefill is measured once per rung"))
    if m and m.kv_capacity_tokens:
        items.append(("KV capacity", f"{m.kv_capacity_tokens:,} tokens (from {m.kv_capacity_source})"))
    if m and m.pre_run_load:
        pl = m.pre_run_load
        items.append(("pre-run load",
                      ("busy/loaded: " if pl.get("loaded") else "idle: ")
                      + f"{pl.get('requests_running', 0):.0f} running, {pl.get('requests_waiting', 0):.0f} waiting, "
                      f"{pl.get('generation_tokens_in_probe') or 0:.0f} tokens generated in {pl.get('probe_s', 0):.0f}s, "
                      f"KV {pl.get('kv_cache_usage_pct') or 0:.1f}%" if pl.get("checked") else "not checked"))
    if hw is not None:
        items.append(("hardware", f"host {hw.host}, GPU {hw.gpu_model} x{hw.gpu_count or '?'}, TP {hw.tensor_parallel or '?'}, "
                                  f"host RAM {hw.host_ram_gib or '?'} GiB, MemAvailable floor {hw.mem_available_floor_gib} GiB"))
    out.append(_kv_table(items))
    reach = context_reach(rows, manifest)
    if reach:
        out.append("<h3>Context reached per user count</h3>")
        out.append(reach)
    # Launch: how the server was started (captured read-only at preflight).
    out.append("<h3>Server launch</h3>")
    if launch:
        out.append(f'<p class="muted">Captured {_e(launch.get("captured_utc"))} UTC / '
                   f'{_e(launch.get("captured_local"))} from {_e(launch.get("source"))} (read-only; secrets redacted).</p>')
        out.append(f"<pre>{_e(launch.get('command_line') or '(start command not captured)')}</pre>")
        env = launch.get("env") or {}
        if env:
            out.append("<details><summary>environment</summary><pre>"
                       + _e("\n".join(f"{k}={v}" for k, v in sorted(env.items()))) + "</pre></details>")
        extra = [(k, ", ".join(map(str, launch[k])) if isinstance(launch.get(k), list) else launch.get(k))
                 for k in ("container", "devices", "group_add", "ipc_mode", "shm_size", "port_bindings",
                           "container_started_at") if launch.get(k)]
        if extra:
            out.append(_kv_table(extra))
        if launch.get("vllm_boot_config"):
            out.append("<details><summary>vLLM boot config (log)</summary><pre>"
                       + _e("\n".join(launch["vllm_boot_config"])) + "</pre></details>")
    else:
        out.append('<p class="muted">No launch record (dry run, imported rows, or a run from before launch '
                   "capture).</p>")
    # Caveats
    cav: list[str] = []
    if m and m.status != "completed":
        cav.append(f"run status is {m.status}")
    lc = (m.launch_check if m else {}) or {}
    for mm in lc.get("mismatches", []) or []:
        cav.append(f"NOT_AS_LABELED: {mm}")
    for w in lc.get("warnings", []) or []:
        cav.append(str(w))
    flagged = sorted({f for r in rows for f in r.flags})
    for f in flagged:
        n = sum(f in r.flags for r in rows)
        cav.append(f"{f} on {n} row(s): {FLAGS.get(f, '')}")
    invalid = sum(r.valid != "valid" for r in rows)
    if invalid:
        cav.append(f"{invalid} row(s) not valid")
    if cav:
        out.append("<h3>Caveats</h3><ul>" + "".join(f"<li>{_e(c)}</li>" for c in cav) + "</ul>")
    return "".join(out)


def render_html(
    rows: list[RungResult],
    manifest: RunManifest | None = None,
    title: str | None = None,
    compare: list[tuple[str, list[RungResult]]] | None = None,
) -> str:
    """Whole report. ``compare`` adds more runs, drawn dashed over the first."""
    label0 = (manifest.run_id if manifest else None) or (rows[0].run_id if rows else "run")
    runs = [(label0, rows)] + list(compare or [])
    all_rows = [r for _, rs in runs for r in rs]
    users_order = sorted({r.users for r in all_rows})
    groups = sorted({(r.prompt_type, r.load_mode) for r in all_rows})
    spec_on = any(has_spec_decode(rs, manifest if i == 0 else None) for i, (_, rs) in enumerate(runs))
    ceilings = ctx_ceilings(rows, manifest)
    ttl = title or f"xpu-llm-bench report — {label0}"
    css = CSS + (CSS_REPEATS if any((r.repeats or 1) > 1 for r in all_rows) else "")
    out = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_e(ttl)}</title><style>{css}</style></head><body>",
        f"<h1>{_e(ttl)}</h1>",
        f'<p class="muted">{len(all_rows)} rows. Charts: X = context per user, one line per concurrency. '
        "Hover a point for the exact value. Self-contained file: no scripts, no network assets.</p>",
        _qline_html(rows, "Quality, all rows"),
        "<h2>Run</h2>",
        _header(rows, manifest),
    ]
    if len(runs) > 1:
        out.append('<p class="note">Compare view: ' + ", ".join(
            f"<b>{_e(lab)}</b> ({'solid' if i == 0 else 'dashed'})" for i, (lab, _) in enumerate(runs)) + ".</p>")
    out.append("<h2>Charts</h2>")
    n_rep = max((r.repeats or 1 for r in all_rows), default=1)
    out.append(_legend(users_order, [lab for lab, _ in runs], repeats=n_rep))
    if not spec_on:
        out.append('<p class="note">No speculative decoding in this run: the accepted-drafts and hit-rate charts '
                   "are skipped.</p>")
    if not all_rows:
        out.append('<p class="muted">No rows.</p>')
    for pt, lm in groups:
        group_rows = [r for r in rows if (r.prompt_type, r.load_mode) == (pt, lm)]
        out.append(f"<h3>{_e(group_label(pt, lm))}</h3>")
        out.append(_qline_html(group_rows))
        out.append('<div class="grid">')
        gceil = {u: v for (p_, l_, u), v in ceilings.items() if (p_, l_) == (pt, lm)}
        for field, mtitle, unit, scale, needs_spec in METRICS:
            if needs_spec and not spec_on:
                continue
            sub = [(lab, [r for r in rs if (r.prompt_type, r.load_mode) == (pt, lm)]) for lab, rs in runs]
            out.append(svg_chart(sub, field, mtitle, unit, scale, users_order, gceil))
        out.append("</div>")
    avg_rows, avg_table = average_rows(rows)
    if len({r.prompt_type for r in rows}) > 1:
        out.append("<h2>Averages across prompt types</h2>")
        out.append('<p class="note">Each card: the mean of the per-prompt-type values for one load mode, user count '
                   "and context rung that has a clean row in every prompt type ("
                   + _e(", ".join(sorted({PROMPT_LABELS.get(r.prompt_type, r.prompt_type) for r in rows})))
                   + "). Flagged and invalid rows are excluded, so a cell where any prompt type was flagged or "
                   "failed is not averaged over fewer types: its card says which type was excluded and why. Each card "
                   "also lists every prompt type's own steps/s and tok/s."
                   + (" With repeats, each value is a median; '(a to b, n=N)' runs from the mean of the per-type "
                      "minima to the mean of the maxima." if n_rep > 1 else "") + "</p>")
        out.append(averages_cards(avg_table))
        for lm in sorted({r.load_mode for r in avg_rows}):
            mode_rows = [r for r in avg_rows if r.load_mode == lm]
            mode_types = " · ".join(_type_label(t) for t in sorted(
                {r.prompt_type for r in rows if r.load_mode == lm and r.prompt_type != AVG_LABEL}))
            out.append(f"<h3>{_e(group_label(AVG_LABEL, lm))} ({_e(mode_types)})</h3>")
            out.append(_legend(users_order, [], f"each line = average of {mode_types} for that user count",
                               repeats=n_rep))
            out.append('<div class="grid">')
            for field, mtitle, unit, scale, needs_spec in METRICS:
                if needs_spec and not spec_on:
                    continue
                out.append(svg_chart([("average", mode_rows)], field, mtitle, unit, scale, users_order))
            out.append("</div>")
    out.append("<h2>All numbers</h2>")
    if groups:
        out.append('<div class="note"><b>Quality pass counts per group</b><ul>' + "".join(
            f"<li>{_e(group_label(pt, lm))}: "
            f"{_e(quality_line([r for r in rows if (r.prompt_type, r.load_mode) == (pt, lm)]))}</li>"
            for pt, lm in groups) + "</ul></div>")
    for lab, rs in runs:
        if len(runs) > 1:
            out.append(f"<h3>{_e(lab)}</h3>")
        out.append(numbers_table(rs))
    flagged = sorted({f for r in all_rows for f in r.flags})
    if flagged:
        out.append("<h2>Flags</h2><ul>" + "".join(
            f"<li><code>{_e(f)}</code>: {_e(FLAGS.get(f, ''))}</li>" for f in flagged) + "</ul>")
    out.append("</body></html>\n")
    return "\n".join(out)
