"""Steady-state (all sessions decoding) window and pre-window quiescence."""

import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_runner_harness import _runner, _spec
from xpu_llm_bench.client import StreamResult
from xpu_llm_bench.metrics import ChunkStats, steady_window
from xpu_llm_bench.results import FLAGS, read_jsonl
from xpu_llm_bench.runner import Runner
from xpu_llm_bench.spec import LadderSpec, MetricNameMap


def _stream(t_first: float, step_s: float, steps: int, per_step: int = 4) -> tuple[list[float], list[int]]:
    """Prefill chunk (1 token) at t_first, then ``steps`` verify chunks."""
    times = [t_first + i * step_s for i in range(steps + 1)]
    return times, [1] + [per_step] * steps


def test_overlap_equals_whole_window_when_sessions_start_together() -> None:
    a, b = _stream(10.0, 0.1, 40), _stream(10.0, 0.1, 40)
    sw = steady_window([a, b])
    assert sw.t_start == 10.0 and sw.window_s == pytest.approx(4.0)
    assert sw.steps == [40, 40] and sw.tokens == [160, 160]
    assert sw.tok_s_each == pytest.approx([40.0, 40.0])
    # Same number ChunkStats gives for each session alone.
    cs = ChunkStats(tokens=161, chunks=41, t_first=a[0][0], t_last=a[0][-1])
    assert sw.tok_s_each[0] == pytest.approx(cs.decode_tok_s)


def test_staggered_first_tokens_drop_the_prefill_overlap() -> None:
    # Session 1 decodes slowly (0.2 s/step) while session 2 is still prefilling,
    # then both decode at 0.1 s/step once session 2's first token is out at t=12.
    t1 = [10.0 + 0.2 * i for i in range(11)] + [12.0 + 0.1 * i for i in range(1, 31)]
    c1 = [1] + [4] * 40
    t2, c2 = _stream(12.0, 0.1, 40)
    sw = steady_window([(t1, c1), (t2, c2)])
    assert sw.t_start == pytest.approx(12.0) and sw.t_end == pytest.approx(15.0)
    assert sw.steps == [30, 30]
    assert sw.tok_s_each == pytest.approx([40.0, 40.0])
    # The per-request window of session 1 mixes in the slow prefill-contended part.
    whole = ChunkStats(tokens=sum(c1), chunks=len(c1), t_first=t1[0], t_last=t1[-1])
    assert whole.decode_tok_s == pytest.approx(160 / 5.0)


def test_no_overlap_gives_no_rate() -> None:
    sw = steady_window([_stream(0.0, 0.1, 10), _stream(5.0, 0.1, 10)])
    assert sw.window_s is None and sw.tok_s_each is None and sw.steps == [0, 0]
    assert steady_window([([], []), _stream(0.0, 0.1, 10)]).tok_s_each is None


def _bare_runner(min_window_steps: int = 20, quiesce_timeout_s: float = 10.0, metrics=None) -> Runner:
    runner = Runner.__new__(Runner)
    runner.spec = SimpleNamespace(
        ladder=LadderSpec(min_window_steps=min_window_steps, quiesce_timeout_s=quiesce_timeout_s),
        metrics=MetricNameMap(),
    )
    runner.fetch_metrics = metrics
    runner.sleeps = []
    runner.sleep = runner.sleeps.append
    runner._runlog = lambda line: None
    return runner


def _result(times: list[float], counts: list[int]) -> StreamResult:
    return StreamResult(status=200, text="", t_send=times[0] - 0.5, t_end=times[-1], first_token_at=times[0],
                        token_times=times, chunk_tokens=counts, output_tokens=sum(counts))


def test_short_overlap_returns_none_but_keeps_window() -> None:
    runner = _bare_runner(min_window_steps=20)
    rung = SimpleNamespace(rung_id="r", decoding_sessions=(1, 2))
    # Overlap of 10 steps (t 9..10) < 20.
    collected = {1: _result(*_stream(0.0, 0.1, 100)), 2: _result(*_stream(9.0, 0.1, 100))}
    each, window = runner._steady(rung, collected)
    assert each is None and window == pytest.approx(1.0)
    long_enough = {1: _result(*_stream(0.0, 0.1, 100)), 2: _result(*_stream(5.0, 0.1, 100))}
    each, window = runner._steady(rung, long_enough)
    assert each == pytest.approx([40.0, 40.0]) and window == pytest.approx(5.0)
    assert "SHORT_OVERLAP" in FLAGS


def test_single_session_steady_equals_per_request_value() -> None:
    runner = _bare_runner(min_window_steps=20)
    rung = SimpleNamespace(rung_id="r", decoding_sessions=(1,))
    times, counts = _stream(0.0, 0.1, 5)  # short, but one user: SHORT_WINDOW covers it
    each, window = runner._steady(rung, {1: _result(times, counts)})
    cs = ChunkStats(tokens=sum(counts), chunks=len(counts), t_first=times[0], t_last=times[-1])
    assert each == pytest.approx([cs.decode_tok_s]) and window == pytest.approx(cs.window_s)


def _metrics(running: float, waiting: float = 0.0) -> str:
    return f"vllm:num_requests_running {running}\nvllm:num_requests_waiting {waiting}\n"


def test_quiesce_waits_for_the_harness_own_requests_to_clear() -> None:
    seq = iter([_metrics(2), _metrics(1, 1), _metrics(0)])
    runner = _bare_runner(metrics=lambda: next(seq))
    snap, busy = runner._quiesce(SimpleNamespace(rung_id="r"))
    assert busy == 0 and snap["vllm:num_requests_running"] == 0
    assert len(runner.sleeps) == 2


def test_quiesce_gives_up_after_the_timeout() -> None:
    runner = _bare_runner(quiesce_timeout_s=1.0, metrics=lambda: _metrics(3))
    _snap, busy = runner._quiesce(SimpleNamespace(rung_id="r"))
    assert busy == 3 and sum(runner.sleeps) == pytest.approx(1.0)


def _ghost_runner(mock_server, tmp_path: Path, out: str, ghost_scrapes: int, **ladder):
    """The mock reports `ghost_scrapes` extra running requests on the scrapes
    right after each growth phase, like vLLM still counting the runner's own
    just-finished requests."""
    state = {"left": 0}
    url = mock_server["base"] + "/metrics"

    def fetch() -> str:
        text = urllib.request.urlopen(url, timeout=5).read().decode()
        if state["left"] > 0:
            state["left"] -= 1
            text = text.replace("vllm:num_requests_running 0.0", "vllm:num_requests_running 2.0")
        return text

    spec = _spec(mock_server["base"], ladder=ladder)
    runner = _runner(spec, tmp_path / out, mock_server, tmp_path, fetch_metrics=fetch, sleep=lambda s: None)
    grow = runner._grow

    def grow_then_ghost(*a, **kw):
        res = grow(*a, **kw)
        state["left"] = ghost_scrapes
        return res

    runner._grow = grow_then_ghost
    return runner


def test_busy_before_window_that_clears_is_not_concurrent_client(mock_server, tmp_path: Path) -> None:
    _ghost_runner(mock_server, tmp_path, "o", ghost_scrapes=2).run()
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    assert rows and all("CONCURRENT_CLIENT" not in r.flags for r in rows)
    log = (tmp_path / "o" / "run.log").read_text()
    assert "for the server to go idle: idle" in log


def test_busy_before_window_that_stays_busy_is_flagged(mock_server, tmp_path: Path) -> None:
    _ghost_runner(mock_server, tmp_path, "o", ghost_scrapes=10_000, quiesce_timeout_s=0.5).run()
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    assert rows and all("CONCURRENT_CLIENT" in r.flags and r.valid == "warning" for r in rows)


def test_rows_carry_steady_fields(mock_server, tmp_path: Path) -> None:
    spec = _spec(mock_server["base"])
    _runner(spec, tmp_path / "o", mock_server, tmp_path).run()
    rows = read_jsonl(tmp_path / "o" / "results.jsonl")
    for r in rows:
        if r.users == 1:
            assert r.decode_tok_s_per_user_steady == pytest.approx(r.decode_tok_s_per_user)
            assert r.steady_window_s == pytest.approx(r.decode_window_s)
        elif "SHORT_OVERLAP" in r.flags:
            assert r.decode_tok_s_per_user_steady is None and r.decode_tok_s_total_steady is None
        else:
            assert r.decode_tok_s_total_steady == pytest.approx(r.decode_tok_s_per_user_steady * r.users, rel=0.01)
