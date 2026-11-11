from pathlib import Path

from xpu_llm_bench.cli import main
from xpu_llm_bench.schedule import build_schedule, validate_progressive

SPECS = Path(__file__).resolve().parents[1] / "specs"


def test_plan_and_run_select_only_c1_c4(monkeypatch, capsys):
    import xpu_llm_bench.cli as cli
    captured = []
    original = cli.make_plan
    def plan(spec, **kwargs):
        captured.append(spec)
        return original(spec, **kwargs)
    def run(spec, *args, **kwargs):
        captured.append(spec)
        return []
    monkeypatch.setattr(cli, "make_plan", plan)
    monkeypatch.setattr(cli, "run_spec", run)
    path = str(SPECS / "generic-vllm.yaml")
    assert main(["plan", path, "--concurrency", "1,4", "--mem-available", "64"]) == 0
    assert main(["run", path, "--concurrency", "1,4"]) == 0
    for spec in captured:
        assert spec.ladder.concurrency_steps == [1, 4]
        rungs = build_schedule(spec)
        assert {r.users for r in rungs} == {1, 4}
        assert all(len(r.decoding_sessions) == r.users for r in rungs)
        assert all(len(r.live_sessions) == r.users for r in rungs)
        assert validate_progressive(rungs) == []


def test_c4_only_does_not_measure_c1(monkeypatch):
    import xpu_llm_bench.cli as cli
    captured = []
    monkeypatch.setattr(cli, "run_spec", lambda spec, *a, **kw: captured.append(spec) or [])
    assert main(["run", str(SPECS / "generic-vllm.yaml"), "--concurrency", "4"]) == 0
    rungs = build_schedule(captured[0])
    assert rungs and {r.users for r in rungs} == {4}
    assert all(len(r.live_sessions) == len(r.decoding_sessions) == 4 for r in rungs)
    assert validate_progressive(rungs) == []


def test_invalid_concurrency_is_rejected(capsys):
    for value in ("", "0", "-1", "1,1", "4,1", "1,,4", "1,3", "999999", "a"):
        assert main(["plan", str(SPECS / "generic-vllm.yaml"), "--concurrency", value]) == 2
        assert "--concurrency" in capsys.readouterr().err


def test_manifest_records_subset_for_dry_run(tmp_path):
    import json
    spec = tmp_path / "s.yaml"
    spec.write_text("name: s\nprofile: custom\nladder: {context_rungs_tokens: [64], concurrency_steps: [1, 2, 4]}\n")
    assert main(["run", str(spec), "--concurrency", "1,4", "--dry-run", "--output-dir", str(tmp_path / "out")]) == 0
    manifest = json.loads((tmp_path / "out" / "s" / "dry-run" / "manifest.json").read_text())
    assert manifest["config"]["concurrency_steps"] == [1, 4]
    assert all(":1u:" in r or ":4u:" in r for r in manifest["planned_rung_ids"])


def test_subset_preserves_full_schedule_session_ids(monkeypatch):
    import xpu_llm_bench.cli as cli
    from xpu_llm_bench.spec import load_spec
    path = SPECS / "generic-vllm.yaml"
    original = build_schedule(load_spec(path))
    captured = []
    monkeypatch.setattr(cli, "run_spec", lambda spec, *a, **kw: captured.append(spec) or [])
    assert main(["run", str(path), "--concurrency", "1,4"]) == 0
    selected = build_schedule(captured[0])
    assert [(r.rung_id, r.live_sessions) for r in selected] == [(r.rung_id, r.live_sessions) for r in original if r.users in (1, 4)]
