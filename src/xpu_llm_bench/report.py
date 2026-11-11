"""Markdown and CSV reports, plus A/B comparison with non-comparable flags."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from xpu_llm_bench.results import RungResult, RunManifest, read_jsonl

# Fixed column order for every table. Do not reorder without a format bump.
COLUMNS = [
    "run_id",
    "rung_id",
    "phase",
    "users",
    "ctx_per_user",
    "ctx_tokens",
    "prompt_type",
    "load_mode",
    "prompt_seed",
    "k",
    "build_fingerprint",
    "attention_path",
    "attention_boot",
    "graphs_on",
    "capture_sizes",
    "decode_batch_tokens",
    "kv_format",
    # Headline speed: steps/s first, tok/s next to it.
    "steps_s_per_user",
    "decode_tok_s_per_user",
    "steps_s_total",
    "decode_tok_s_total",
    "tokens_per_step",
    "accepted_drafts_per_step",
    "hit_rate",
    "acceptance_rate_vllm",
    "verify_steps",
    "verify_steps_server",
    "tokens_delivered",
    "decode_window_s",
    "prefill_tok_s",
    "ttft_ms",
    "tpot_ms",
    "itl_p50_ms",
    "itl_p95_ms",
    "itl_p99_ms",
    "e2e_ms",
    "queue_time_ms",
    "running_reqs",
    "waiting_reqs",
    "running_max",
    "waiting_max",
    "prefix_cache_hit_rate",
    "kv_cache_usage_pct",
    "kv_cache_usage_max_pct",
    "preemptions",
    "mem_available_min_gib",
    "rss_kib",
    "other_vllm_stacks",
    "gpu_mem",
    "eager_fallback",
    "quality",
    "errors",
    "timeouts",
    "exit_reason",
    "flags",
    "valid",
    "reason",
    # Added later; appended so earlier CSV column positions stay put.
    "decode_tok_s_per_user_steady",
    "decode_tok_s_total_steady",
    "steady_window_s",
]

# Short table for humans: one comparable table, steps/s next to tok/s.
SUMMARY_COLUMNS = [
    "rung_id",
    "users",
    "ctx_tokens",
    "attention_path",
    "steps_s_per_user",
    "decode_tok_s_per_user",
    "decode_tok_s_total",
    "decode_tok_s_per_user_steady",
    "decode_tok_s_total_steady",
    "tokens_per_step",
    "hit_rate",
    "prefill_tok_s",
    "ttft_ms",
    "itl_p95_ms",
    "kv_cache_usage_max_pct",
    "mem_available_min_gib",
    "quality",
    "flags",
]

NUMERIC = {
    "prefill_tok_s",
    "ttft_ms",
    "tpot_ms",
    "decode_tok_s_per_user",
    "decode_tok_s_total",
    "decode_tok_s_per_user_steady",
    "decode_tok_s_total_steady",
    "steps_s_per_user",
    "steps_s_total",
    "tokens_per_step",
    "accepted_drafts_per_step",
    "hit_rate",
    "acceptance_rate_vllm",
    "itl_p50_ms",
    "itl_p95_ms",
    "itl_p99_ms",
    "e2e_ms",
    "queue_time_ms",
    "running_reqs",
    "waiting_reqs",
    "prefix_cache_hit_rate",
    "kv_cache_usage_pct",
    "preemptions",
    "mem_available_min_gib",
    "timeouts",
}


def _quality_cell(row: RungResult) -> str:
    q = row.quality
    parts = []
    if q.needle_pass is not None:
        parts.append("needle " + ("ok" if q.needle_pass else "FAIL"))
    if q.code_edit_pass is not None:
        parts.append("code " + ("ok" if q.code_edit_pass else "FAIL"))
    if q.format_ok is not None:
        parts.append("format " + ("ok" if q.format_ok else "FAIL"))
    if q.greedy_ref:
        g = q.greedy_ref
        if q.greedy_first_divergence is not None:
            g += f"@{q.greedy_first_divergence}"
        parts.append("greedy " + g)
    if q.logprob_ok is not None:
        d = f"{q.logprob_worst_delta:+.3f}" if q.logprob_worst_delta is not None else ""
        parts.append("logprob " + ("ok" if q.logprob_ok else "FAIL") + d)
    return ", ".join(parts)


def _cell(row: RungResult, col: str) -> str:
    val = getattr(row, col, None)
    if col == "errors":
        return "; ".join(row.errors) if row.errors else ""
    if col == "quality":
        return _quality_cell(row)
    if col == "flags":
        return " ".join(row.flags)
    if col == "build_fingerprint":
        return (val or "")[:12]
    if col == "rss_kib":
        return " ".join(f"{k}={v // 1024}M" for k, v in sorted((val or {}).items()))
    if col in {"ctx_tokens", "capture_sizes", "other_vllm_stacks"}:
        return "/".join(str(x) for x in (val or []))
    if val is None:
        return ""
    if isinstance(val, float):
        if col in {"hit_rate", "prefix_cache_hit_rate"}:
            return f"{val:.3f}"
        if col.endswith("_ms"):
            return f"{val:.1f}"
        if col.endswith("_s") or "tok_s" in col or col.startswith("steps_"):
            return f"{val:.2f}"
        return f"{val:.3f}"
    if isinstance(val, bool):
        return "yes" if val else "no"
    return str(val)


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def rows_to_markdown(rows: list[RungResult], columns: list[str] | None = None) -> str:
    cols = columns or COLUMNS
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_md_escape(_cell(row, c)) for c in cols) + " |")
    return "\n".join([header, sep, *body, ""])


def rows_to_csv(rows: list[RungResult], columns: list[str] | None = None) -> str:
    cols = columns or COLUMNS
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(cols)
    for row in rows:
        writer.writerow([_cell(row, c) for c in cols])
    return buf.getvalue()


def non_comparable_reasons(a: RungResult, b: RungResult) -> list[str]:
    """Loud reasons two rows must not be treated as a clean A/B."""
    reasons: list[str] = []
    if a.prompt_type != b.prompt_type:
        reasons.append(f"prompt_type mismatch ({a.prompt_type} vs {b.prompt_type})")
    if a.load_mode != b.load_mode:
        reasons.append(f"load_mode mismatch ({a.load_mode} vs {b.load_mode})")
    if a.users != b.users:
        reasons.append(f"users mismatch ({a.users} vs {b.users})")
    if a.ctx_per_user != b.ctx_per_user:
        reasons.append(f"context mismatch ({a.ctx_per_user} vs {b.ctx_per_user})")
    if a.attention_path != b.attention_path:
        reasons.append(f"attention path mismatch ({a.attention_path} vs {b.attention_path})")
    if a.phase != b.phase:
        reasons.append(f"ladder phase mismatch ({a.phase} vs {b.phase})")
    if a.k is not None and b.k is not None and a.k != b.k:
        reasons.append(f"k mismatch ({a.k} vs {b.k}); compare steps/s only, not hit rate")
    if a.prompt_seed is not None and b.prompt_seed is not None and a.prompt_seed != b.prompt_seed:
        reasons.append(f"prompt seed mismatch ({a.prompt_seed} vs {b.prompt_seed})")
    if a.temperature is not None and b.temperature is not None and a.temperature != b.temperature:
        reasons.append(f"temperature mismatch ({a.temperature} vs {b.temperature})")
    if a.ctx_tokens and b.ctx_tokens and len(a.ctx_tokens) == len(b.ctx_tokens):
        drift = max(abs(x - y) / max(x, y, 1) for x, y in zip(a.ctx_tokens, b.ctx_tokens, strict=False))
        if drift > 0.05:
            reasons.append(f"session contexts differ by {drift:.0%}")
    both_converted = "CONVERTED" in a.flags and "CONVERTED" in b.flags
    if both_converted:
        reasons.append(
            "both rows imported from legacy files (CONVERTED): definitions, windows and attention "
            "path are not verified the way live rows are; treat as a caveated, not clean, A/B"
        )
    for side, row in (("A", a), ("B", b)):
        for flag in row.flags:
            if flag in {"K_UNDECLARED"} or (both_converted and flag == "CONVERTED"):
                continue
            reasons.append(f"{side} flagged {flag}")
        if row.attention_path == "fallback":
            reasons.append(f"{side} ran a silent fallback path")
        if row.attention_path == "mixed":
            reasons.append(f"{side} ran a mixed QSA/dense path")
        if row.eager_fallback:
            reasons.append(f"{side} hit an eager graph fallback")
        if row.valid == "invalid":
            reasons.append(f"{side} is invalid: {row.reason or 'unspecified'}")
        if row.reason and "ram floor" in (row.reason or "").lower():
            reasons.append(f"{side} hit the MemAvailable floor")
        if row.timeouts:
            reasons.append(f"{side} had {row.timeouts} timeout(s)")
        if row.errors:
            reasons.append(f"{side} had errors")
    return reasons


@dataclass
class ComparePair:
    key: tuple
    a: RungResult | None
    b: RungResult | None
    reasons: list[str]
    deltas: dict[str, float | None]


def compare_key(row: RungResult) -> tuple[int, int, str]:
    """Rows pair on users + context + prompt type. load_mode / phase / k
    differences do not split the pair; they mark it NON-COMPARABLE instead."""
    return (row.users, row.ctx_per_user, row.prompt_type)


def _index(rows: list[RungResult]) -> dict[tuple, RungResult]:
    """Index by compare_key plus an occurrence number, so repeated keys in one
    run (e.g. two load modes) are kept as separate pairs, never collapsed."""
    out: dict[tuple, RungResult] = {}
    seen: dict[tuple, int] = {}
    for row in rows:
        base = compare_key(row)
        n = seen.get(base, 0)
        seen[base] = n + 1
        out[(*base, n)] = row
    return out


def compare_runs(run_a: list[RungResult], run_b: list[RungResult]) -> list[ComparePair]:
    ia, ib = _index(run_a), _index(run_b)
    keys = sorted(set(ia) | set(ib))
    pairs: list[ComparePair] = []
    for key in keys:
        a, b = ia.get(key), ib.get(key)
        reasons: list[str] = []
        deltas: dict[str, float | None] = {}
        if a is None:
            reasons.append("missing in A")
        if b is None:
            reasons.append("missing in B")
        if a is not None and b is not None:
            reasons.extend(non_comparable_reasons(a, b))
            for col in NUMERIC:
                va, vb = getattr(a, col), getattr(b, col)
                if va is None or vb is None:
                    deltas[col] = None
                else:
                    deltas[col] = float(vb) - float(va)
        pairs.append(ComparePair(key=key, a=a, b=b, reasons=reasons, deltas=deltas))
    return pairs


def format_compare(pairs: list[ComparePair], name_a: str = "A", name_b: str = "B") -> str:
    lines = [
        f"# Compare {name_a} → {name_b}",
        "",
        "Deltas are B minus A. Rows flagged **NON-COMPARABLE** must not be read as a speedup.",
        "",
        "| users | ctx | prompt_type | load_mode | phase | Δ steps/s /user | Δ decode tok/s /user | Δ total | Δ hit_rate | flags |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pair in pairs:
        users, ctx, ptype = pair.key[:3]
        ref = pair.a or pair.b
        mode = ref.load_mode if ref else ""
        phase = ref.phase if ref else ""
        if pair.a is not None and pair.b is not None:
            if pair.a.load_mode != pair.b.load_mode:
                mode = f"{pair.a.load_mode}/{pair.b.load_mode}"
            if pair.a.phase != pair.b.phase:
                phase = f"{pair.a.phase}/{pair.b.phase}"
        if pair.a is None or pair.b is None:
            flag = "NON-COMPARABLE: " + "; ".join(pair.reasons)
            lines.append(f"| {users} | {ctx} | {ptype} | {mode} | {phase} |  |  |  |  | {flag} |")
            continue
        def _fmt(deltas: dict, col: str) -> str:
            val = deltas.get(col)
            return "" if val is None else f"{val:+.3f}"

        flag = ""
        if pair.reasons:
            flag = "NON-COMPARABLE: " + "; ".join(pair.reasons)
        lines.append(
            f"| {users} | {ctx} | {ptype} | {mode} | {phase} | "
            f"{_fmt(pair.deltas, 'steps_s_per_user')} | {_fmt(pair.deltas, 'decode_tok_s_per_user')} | "
            f"{_fmt(pair.deltas, 'decode_tok_s_total')} | {_fmt(pair.deltas, 'hit_rate')} | {flag} |"
        )
    lines.append("")
    return "\n".join(lines)


def load_run(path: str | Path) -> list[RungResult]:
    return read_jsonl(path)


def run_report(rows: list[RungResult], manifest: RunManifest | None = None) -> str:
    """One markdown file per run: header, evidence, one comparable table, quality, flags."""
    from xpu_llm_bench.results import FLAGS

    out: list[str] = []
    title = manifest.run_id if manifest else (rows[0].run_id if rows else "empty run")
    out.append(f"# {title}")
    out.append("")
    if manifest:
        fp = manifest.fingerprint or {}
        boot = manifest.boot or {}
        out += [
            "## Header",
            "",
            "| field | value |",
            "| --- | --- |",
            f"| spec / profile | {manifest.spec_name} / {manifest.profile} |",
            f"| status | {manifest.status}{' — ' + manifest.abort_reason if manifest.abort_reason else ''} |",
            f"| started ({manifest.display_timezone or 'local'}) | {manifest.started_local} |",
            f"| container | {fp.get('container', '')} {str(fp.get('container_id') or '')[:12]} |",
            f"| image tag / ID | {manifest.image} / {fp.get('image_id') or manifest.image_digest} |",
            f"| build fingerprint | {fp.get('build_fingerprint') or 'n/r'} |",
            f"| boot started | {fp.get('started_at') or 'n/r'} |",
            f"| attention (boot log) | {boot.get('attention_boot', 'n/r')} (requested {manifest.attention.requested}) |",
            f"| graphs (boot log) | on={boot.get('graphs_on')} mode={boot.get('cudagraph_mode')} sizes={boot.get('capture_sizes_logged')} counts={boot.get('capture_counts')} |",
            f"| KV | {boot.get('kv_dtype') or manifest.kv_format} — {boot.get('kv_tokens')} tokens |",
            f"| speculative | {boot.get('spec_method')} k={boot.get('spec_k')} (spec k={manifest.k}) |",
            f"| max model len | {boot.get('max_model_len')} |",
            f"| prompt seed | {manifest.prompt_seed} |",
            f"| decode method | {manifest.decode_method.method_id if manifest.decode_method else 'n/r'} |",
            f"| hold | {manifest.hold} |",
            "",
        ]
        if manifest.attention.evidence_lines:
            out += ["## Evidence (quoted log lines)", "", "```"]
            out += manifest.attention.evidence_lines[:20]
            out += ["```", ""]
    out += ["## Ladder", "", rows_to_markdown(rows, SUMMARY_COLUMNS)]
    flagged = sorted({f for r in rows for f in r.flags})
    if flagged:
        out += ["## Flags", ""]
        out += [f"- `{f}`: {FLAGS.get(f, '')}" for f in flagged]
        out.append("")
    if manifest and manifest.first_greedy_divergence:
        out += ["## Greedy divergence", "", f"First divergence: {manifest.first_greedy_divergence}", ""]
    out += ["## Full rows", "", "CSV with every column: `bench report <results.jsonl> --csv out.csv`.", ""]
    return "\n".join(out)
