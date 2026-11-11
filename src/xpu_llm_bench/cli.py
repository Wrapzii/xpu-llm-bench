"""Command-line entry: bench plan | run | report | compare | import-legacy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xpu_llm_bench import __version__
from xpu_llm_bench.html_report import render_html
from xpu_llm_bench.import_legacy import import_legacy_file
from xpu_llm_bench.planner import (
    GRID_WARNING,
    baseline_mem_available_from_rows,
    format_plan,
    local_mem_available_gib,
    make_plan,
)
from xpu_llm_bench.report import (
    COLUMNS,
    compare_runs,
    format_compare,
    load_run,
    rows_to_csv,
    rows_to_markdown,
    run_report,
)
from xpu_llm_bench.results import RunManifest
from xpu_llm_bench.runner import RunRefused, run_spec
from xpu_llm_bench.safety import SafetyError
from xpu_llm_bench.site import SiteConfigError, describe_site
from xpu_llm_bench.spec import load_spec


def _run_label(path: Path) -> str:
    return path.parent.name if path.name == "results.jsonl" else path.stem


def _add_profile(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--profile",
        choices=["quick", "full", "custom"],
        default=None,
        help="Override the spec profile. quick ≈ 10 min; full is the complete matrix.",
    )


def _add_repeats(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--repeats",
        type=int,
        default=None,
        metavar="N",
        help="Measure each rung's decode window N times on the same context (default: spec ladder.repeats, 1)",
    )


def _apply_repeats(spec, n: int | None) -> None:
    if n is None:
        return
    if n < 1 or n > 50:
        raise SystemExit("--repeats must be between 1 and 50")
    spec.ladder.repeats = n


def _add_concurrency(p: argparse.ArgumentParser) -> None:
    p.add_argument("--concurrency", metavar="N[,N...]", default=None,
                   help="Measure only these increasing concurrency steps from the resolved spec (e.g. 1,4)")


def _apply_concurrency(spec, value: str | None) -> None:
    if value is None:
        return
    parts = value.split(",")
    if any(not part.isascii() or not part.isdigit() for part in parts):
        raise ValueError("--concurrency must be comma-separated positive integers")
    steps = [int(part) for part in parts]
    if not steps or any(n < 1 for n in steps) or steps != sorted(set(steps)):
        raise ValueError("--concurrency must contain unique, strictly increasing positive integers")
    if not set(steps).issubset(spec.ladder.concurrency_steps):
        raise ValueError("--concurrency values must be steps in the resolved spec")
    spec.ladder.concurrency_session_steps = list(spec.ladder.concurrency_steps)
    spec.ladder.concurrency_steps = steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Consistent, comparable LLM-serving benchmarks on Intel XPU (vLLM).",
    )
    parser.add_argument("--version", action="version", version=f"xpu-llm-bench {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan", help="Print rung order, time estimate, and KV/RAM feasibility")
    plan.add_argument("spec", type=Path)
    plan.add_argument(
        "--measured",
        type=Path,
        default=None,
        help="results.jsonl from an earlier run on this host: use its measured MemAvailable baseline",
    )
    plan.add_argument(
        "--mem-available",
        type=float,
        default=None,
        metavar="GIB",
        help="Current MemAvailable on the bench host (GiB) with the test server booted",
    )
    plan.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Server max_model_len, to show the full `auto` context ladder (a live run reads it from /v1/models)",
    )
    plan.add_argument(
        "--kv-capacity-tokens",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Server KV cache capacity in tokens (boot log 'KV cache size: N tokens'), for per-user-count ceilings",
    )
    _add_profile(plan)
    _add_repeats(plan)
    _add_concurrency(plan)

    run = sub.add_parser("run", help="Drive the ladder against an OpenAI-compatible endpoint")
    run.add_argument("spec", type=Path)
    run.add_argument("--output-dir", type=Path, default=Path("results"))
    _add_repeats(run)
    _add_concurrency(run)
    run.add_argument("--resume", action="store_true", help="Skip rungs already completed in the output dir")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Safety preflight + plan + placeholder rows. No HTTP, no bench-hold, no GPU.",
    )
    run.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Run without asking when the server is already busy or the GPU already loaded (still flagged)",
    )
    run.add_argument(
        "--ram-wait-min",
        type=float,
        default=0.0,
        help="After a RAM-floor abort, wait up to N minutes for MemAvailable >= floor+1 GiB, then retry the rung",
    )
    run.add_argument(
        "--no-site-config",
        action="store_true",
        help="Acknowledge running without a site config (only built-in neutral protection applies)",
    )
    run.add_argument("--variant", default=None, help="Run only this variant (boot its container first)")
    run.add_argument("--no-html", action="store_true", help="Do not write report.html at the end of the run")
    run.add_argument(
        "--pause-between-variants",
        action="store_true",
        help="Prompt before each variant so you can boot that variant's container",
    )
    _add_profile(run)

    report = sub.add_parser(
        "report", help="Print a run's Markdown report to stdout (--output FILE to write it instead)"
    )
    report.add_argument("run", type=Path, help="results.jsonl from a bench run, or its run directory")
    report.add_argument("--csv", type=Path, default=None, help="Write every column as CSV")
    report.add_argument("--output", type=Path, default=None, help="Write the markdown here instead of stdout")
    report.add_argument("--wide", action="store_true", help="Markdown table with every column")
    report.add_argument(
        "--html",
        action="store_true",
        help="Write a self-contained HTML report with SVG charts (default: report.html beside the results)",
    )
    report.add_argument("--html-out", type=Path, default=None, help="Path for the HTML report (implies --html)")

    compare = sub.add_parser("compare", help="Compare two or more runs; flag non-comparable rows")
    compare.add_argument("runs", nargs="+", type=Path, help="results.jsonl files (or run dirs), A B [C...]")
    compare.add_argument("--output", type=Path, default=None)
    compare.add_argument(
        "--html", type=Path, default=None, help="Also write an HTML overlay: A solid, the others dashed"
    )

    imp = sub.add_parser("import-legacy", help="Convert older ad-hoc rung JSONL into the current schema")
    imp.add_argument("src", type=Path)
    imp.add_argument("dest", type=Path)
    imp.add_argument("--run-id", default="legacy")
    imp.add_argument(
        "--correct-mtp6-hit-rate",
        action="store_true",
        help="Multiply the imported acceptance counter by 7/12 (old MTP-6 summed ratio). "
        "Lands in acceptance_rate_vllm; hit_rate is always derived from tokens_per_step/drafts",
    )
    imp.add_argument(
        "--correct-summed-ratio",
        type=int,
        default=None,
        metavar="K",
        help="Convert the old summed counter ratio 2A/((k+1)D) to A/(kD) in acceptance_rate_vllm: multiply by (K+1)/(2K)",
    )
    imp.add_argument("--port", type=int, default=None, help="Server port of the legacy run (a protected production port -> PROTECTED_PORT flag)")
    imp.add_argument("--prompt-type", default="unknown")
    imp.add_argument(
        "--attention-path",
        choices=["qsa", "dense", "fallback", "unknown"],
        default=None,
        help="Attention path the legacy run used. Recorded as DECLARED (flag ATTENTION_DECLARED), "
        "not proven by log evidence",
    )
    return parser


def _fail(msg: str, code: int) -> int:
    print(f"bench: {msg}", file=sys.stderr)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _main(parser, args)
    except RunRefused as exc:
        where = ""
        if exc.manifest is not None:
            where = f"\n(manifest marked {exc.manifest.status}: {exc.manifest.output_jsonl or ''})".replace(
                "results.jsonl", "manifest.json"
            )
        return _fail(f"run refused:\n{exc}{where}", 3)
    except (SafetyError, SiteConfigError) as exc:
        return _fail(f"refused: {exc}", 3)
    except (ValueError, FileNotFoundError) as exc:
        return _fail(str(exc), 2)


def _no_results(run_path: Path, manifest: RunManifest | None) -> int:
    """`bench report` on a run with no rows: say why instead of a raw I/O error."""
    where = run_path.parent
    if manifest is None:
        print(f"no results in {where}: no results.jsonl and no manifest.json (not a run directory?)", file=sys.stderr)
        return 2
    status = manifest.status
    reason = (manifest.abort_reason or "").strip()
    if status == "refused":
        msg = f"run {manifest.run_id} was refused before any rung ran"
    elif status == "dry_run":
        msg = f"run {manifest.run_id} is a dry run: nothing was sent, so there are no results"
    elif status == "in_progress":
        msg = f"run {manifest.run_id} has no results yet (in progress, or stopped before its first rung)"
    else:
        msg = f"run {manifest.run_id} ({status}) has no results"
    print(msg + (f": {reason}" if reason else ""), file=sys.stderr)
    return 1


def _main(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.cmd == "plan":
        spec = load_spec(args.spec, profile=args.profile)
        _apply_repeats(spec, args.repeats)
        _apply_concurrency(spec, args.concurrency)
        mem, basis = None, "live"
        if args.mem_available is not None:
            mem, basis = args.mem_available, "given"
        elif args.measured:
            mem, basis = baseline_mem_available_from_rows(load_run(args.measured)), "measured baseline"
        elif spec.access.mode == "local":
            mem = local_mem_available_gib()
        plan = make_plan(
            spec, mem_available_gib=mem, basis=basis,
            max_model_len=args.max_model_len, kv_capacity_tokens=args.kv_capacity_tokens,
        )
        print(describe_site())
        sys.stdout.write(format_plan(plan))
        return 0  # estimates are informational; nothing in a plan blocks a run
    if args.cmd == "run":
        spec = load_spec(args.spec, profile=args.profile)
        _apply_repeats(spec, args.repeats)
        _apply_concurrency(spec, args.concurrency)
        print(describe_site())
        if spec.ladder.shape == "grid":
            print(GRID_WARNING, file=sys.stderr)
        if args.dry_run:
            mem = local_mem_available_gib() if spec.access.mode == "local" else None
            sys.stdout.write(format_plan(make_plan(spec, mem_available_gib=mem)))
        manifests = run_spec(
            spec,
            args.output_dir,
            dry_run=args.dry_run,
            resume=args.resume,
            ram_wait_min=args.ram_wait_min,
            assume_yes=args.yes,
            no_site_config=args.no_site_config,
            variant=args.variant,
            pause_between_variants=args.pause_between_variants,
            html=not args.no_html,
        )
        for man in manifests:
            if man.status == "dry_run":
                out = Path(man.output_jsonl or "").parent
                print(f"{man.spec_name}: DRY RUN run_id={man.run_id} planned_rungs={len(man.planned_rung_ids)} -> {out}")
                print("  nothing was sent; this manifest is ignored by --resume")
                print("  safety commands a live run would execute:")
                for line in (man.hold or {}).get("would_run", []):
                    print(f"    {line}")
                for warn in (man.launch_check or {}).get("warnings", []):
                    print(f"  WARNING: {warn}")
                continue
            print(f"{man.spec_name}: {man.status} run_id={man.run_id} rungs={len(man.completed_rung_ids)}")
            for item in (man.launch_check or {}).get("mismatches", []):
                print(f"  NOT_AS_LABELED: {item}", file=sys.stderr)
            if man.abort_reason:
                print(f"  aborted: {man.abort_reason}", file=sys.stderr)
                return 2
        return 0
    if args.cmd == "report":
        run_path = args.run / "results.jsonl" if args.run.is_dir() else args.run
        man_path = run_path.parent / "manifest.json"
        manifest = RunManifest.load(man_path) if man_path.exists() else None
        rows = load_run(run_path) if run_path.exists() else []
        if not rows:
            return _no_results(run_path, manifest)
        if args.html or args.html_out:
            out_html = args.html_out or run_path.parent / "report.html"
            out_html.write_text(render_html(rows, manifest), encoding="utf-8")
            print(f"wrote {out_html}", file=sys.stderr)
            if not (args.output or args.csv):
                return 0
        md = rows_to_markdown(rows, COLUMNS) if args.wide else run_report(rows, manifest)
        if args.output:
            args.output.write_text(md, encoding="utf-8")
            print(f"wrote {args.output}", file=sys.stderr)
        else:
            sys.stdout.write(md)  # markdown on stdout by default
        if args.csv:
            args.csv.write_text(rows_to_csv(rows, COLUMNS), encoding="utf-8")
            print(f"wrote {args.csv}", file=sys.stderr)
        return 0
    if args.cmd == "compare":
        if len(args.runs) < 2:
            parser.error("compare needs at least two run files")
        texts = []
        paths = [r / "results.jsonl" if r.is_dir() else r for r in args.runs]
        args.runs = paths
        base = load_run(args.runs[0])
        for other in args.runs[1:]:
            texts.append(
                format_compare(
                    compare_runs(base, load_run(other)),
                    name_a=str(args.runs[0]),
                    name_b=str(other),
                )
            )
        out = "\n".join(texts)
        if args.html:
            man_path = paths[0].parent / "manifest.json"
            man0 = RunManifest.load(man_path) if man_path.exists() else None
            others = [(_run_label(p), load_run(p)) for p in paths[1:]]
            args.html.write_text(
                render_html(
                    base,
                    man0,
                    title=f"xpu-llm-bench compare — {_run_label(paths[0])} vs " + ", ".join(lab for lab, _ in others),
                    compare=others,
                ),
                encoding="utf-8",
            )
            print(f"wrote {args.html}", file=sys.stderr)
        if args.output:
            args.output.write_text(out, encoding="utf-8")
        else:
            sys.stdout.write(out)
        return 0
    if args.cmd == "import-legacy":
        n = import_legacy_file(
            args.src,
            args.dest,
            run_id=args.run_id,
            correct_mtp6_hit_rate=args.correct_mtp6_hit_rate,
            prompt_type=args.prompt_type,
            correct_k=args.correct_summed_ratio,
            port=args.port,
            attention_path=args.attention_path,
        )
        print(f"imported {n} rows -> {args.dest}")
        return 0
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

