"""Progressive rung schedule: grow context monotonically, add concurrency in place."""

from __future__ import annotations

from dataclasses import dataclass

from xpu_llm_bench.spec import LoadMode, ProfileName, PromptType, RunSpec, ladder_contexts


@dataclass(frozen=True)
class Rung:
    """One measurement point in a progressive ladder."""

    rung_id: str
    prompt_type: PromptType
    load_mode: LoadMode
    users: int
    ctx_per_user: int
    # Sessions that should be at this context after the grow step (1-based ids,
    # unique within a prompt-type family).
    live_sessions: tuple[int, ...]
    # Sessions that actually decode during the measurement window.
    decoding_sessions: tuple[int, ...]
    # Sessions that need to be grown (append) to reach ctx_per_user before measure.
    grow_sessions: tuple[int, ...]
    # True when this rung also measures the prefill of a growth step.
    measures_prefill: bool
    output_tokens: int
    family: str  # prompt type family; sessions are not reused across families
    # "S" (session 1 grows), "C" (sessions join), or "meas" (grid shape).
    phase: str = "meas"
    # Target context per live session, aligned with live_sessions. Empty means
    # every live session is at ctx_per_user.
    ctx_targets: tuple[int, ...] = ()

    @property
    def same_context(self) -> bool:
        return not self.ctx_targets or len(set(self.ctx_targets)) == 1

    def target_for(self, sid: int) -> int:
        if self.ctx_targets:
            return self.ctx_targets[self.live_sessions.index(sid)]
        return self.ctx_per_user


def _rung_id(prompt_type: str, load_mode: str, users: int, ctx: int, phase: str = "meas") -> str:
    short = {
        "closed_all_decoding": "closed",
        "staggered": "staggered",
        "open_loop": "open",
    }.get(load_mode, load_mode)
    return f"{prompt_type}:{short}:{users}u:{ctx}:{phase}"


def build_session1_then_join(spec: RunSpec) -> list[Rung]:
    """The progressive ladder. Per prompt type and load mode, on one boot:

    Phase S: session 1 grows through every context rung in one conversation
    (append only, prefix cache reused), up to the top rung (the server's
    max_model_len minus headroom with ``context_rungs_tokens: auto``).

    Phase C: for each concurrency step n > 1, n users join at the first rung
    (each prefills on its own) and then grow together, rung by rung, with all
    n decoding in every window, until the top rung or until that user count
    stops fitting (the runner records CTX_CEILING and moves to the next user
    count). Each user count has its own conversations, so all n users sit at
    the same context: one clean curve per concurrency. A conversation only
    ever grows (append-only, prefix reuse); earlier ones are released, never
    shrunk or refilled.
    """
    ladder = spec.ladder
    out: list[Rung] = []
    ctxs = ladder_contexts(spec)
    for ptype in spec.prompts.types:
        for mode in ladder.load_modes:
            if 1 in ladder.concurrency_steps:
                for ctx in ctxs:
                    out.append(
                        Rung(
                            rung_id=_rung_id(ptype, mode, 1, ctx, "S"),
                            prompt_type=ptype,
                            load_mode=mode,
                            users=1,
                            ctx_per_user=ctx,
                            live_sessions=(1,),
                            decoding_sessions=(1,),
                            grow_sessions=(1,),
                            measures_prefill=True,
                            output_tokens=ladder.output_tokens_per_rung,
                            family=str(ptype),
                            phase="S",
                            ctx_targets=(ctx,),
                        )
                    )
            join = ladder.join_ctx_tokens or ctxs[0]
            c_ctxs = [c for c in ctxs if c >= join] or ctxs[-1:]
            if ladder.join_max_context_tokens:
                c_ctxs = [c for c in c_ctxs if c <= ladder.join_max_context_tokens] or c_ctxs[:1]
            next_sid = 2
            for n in (ladder.concurrency_session_steps or ladder.concurrency_steps):
                if n <= 1:
                    continue
                group = tuple(range(next_sid, next_sid + n))
                next_sid += n
                if n not in ladder.concurrency_steps:
                    continue
                for ctx in c_ctxs:
                    out.append(
                        Rung(
                            rung_id=_rung_id(ptype, mode, n, ctx, "C"),
                            prompt_type=ptype,
                            load_mode=mode,
                            users=n,
                            ctx_per_user=ctx,
                            live_sessions=group,
                            decoding_sessions=group,
                            grow_sessions=group,
                            measures_prefill=True,
                            output_tokens=ladder.output_tokens_per_rung,
                            family=str(ptype),
                            phase="C",
                            ctx_targets=tuple([ctx] * n),
                        )
                    )
    return out


def ceiling_key(rung: Rung) -> str:
    """Key of the (family, load mode, users) line a ceiling stops."""
    return f"{rung.family}|{rung.load_mode}|{rung.users}"


def build_schedule(spec: RunSpec, profile: ProfileName | None = None) -> list[Rung]:
    """Build a cache-reusing, never-shrink schedule (shape from ``ladder.shape``).

    For each prompt type (a session family) and each load mode:

    1. At the smallest context rung, add sessions to reach 1, 2, 4, 8, ... users
       and measure each concurrency with all of those sessions at that context.
    2. Grow *all* live sessions together to the next context (append, prefix-cache
       reuse). Measure the max concurrency, then quiesce down through the lower
       concurrency steps (sessions stay live at the new context; only a subset
       decodes). Context never decreases.
    3. Finish the family with all sessions decoding together at the high-water
       context (already measured as the max-concurrency rung; not repeated).
    4. Drop the family and start the next prompt type with fresh sessions.
       That is not a shrink of the same conversation; the prefixes cannot be
       reused across prompt types.

    Open-loop and staggered modes reuse the same session growth order; only the
    arrival pattern in the measurement window changes.
    """
    ladder = spec.ladder
    if ladder.shape == "session1_then_join":
        return build_session1_then_join(spec)
    rungs_out: list[Rung] = []
    output = ladder.output_tokens_per_rung
    for ptype in spec.prompts.types:
        family = str(ptype)
        for mode in ladder.load_modes:
            live: list[int] = []
            next_sid = 1
            ctxs = ladder_contexts(spec)
            first = True
            for ctx in ctxs:
                if first:
                    for n in ladder.concurrency_steps:
                        grow: list[int] = []
                        while len(live) < n:
                            live.append(next_sid)
                            grow.append(next_sid)
                            next_sid += 1
                        decoding = tuple(live[:n])
                        rungs_out.append(
                            Rung(
                                rung_id=_rung_id(ptype, mode, n, ctx),
                                prompt_type=ptype,
                                load_mode=mode,
                                users=n,
                                ctx_per_user=ctx,
                                live_sessions=tuple(live),
                                decoding_sessions=decoding,
                                grow_sessions=tuple(grow),
                                measures_prefill=bool(grow),
                                output_tokens=output,
                                family=family,
                            )
                        )
                    first = False
                    continue
                # Grow every live session to the new context, then measure
                # from high concurrency down so the all-decoding point comes first.
                grow_all = tuple(live)
                steps = list(reversed(ladder.concurrency_steps))
                # Only measure concurrencies we actually have sessions for.
                steps = [n for n in steps if n <= len(live)]
                for i, n in enumerate(steps):
                    rungs_out.append(
                        Rung(
                            rung_id=_rung_id(ptype, mode, n, ctx),
                            prompt_type=ptype,
                            load_mode=mode,
                            users=n,
                            ctx_per_user=ctx,
                            live_sessions=tuple(live),
                            decoding_sessions=tuple(live[:n]),
                            grow_sessions=grow_all if i == 0 else (),
                            measures_prefill=i == 0,
                            output_tokens=output,
                            family=family,
                        )
                    )
    return rungs_out


def validate_progressive(rungs: list[Rung]) -> list[str]:
    """Return a list of violations of the progressive-growth rules."""
    errors: list[str] = []
    # Per family+mode, session context must be non-decreasing, and a session
    # id is never reused after the family ends.
    highwater: dict[tuple[str, str, int], int] = {}
    last_family_mode: tuple[str, str] | None = None
    seen_ids: set[str] = set()
    for rung in rungs:
        if rung.rung_id in seen_ids:
            errors.append(f"duplicate rung_id {rung.rung_id}")
        seen_ids.add(rung.rung_id)
        key_fm = (rung.family, rung.load_mode)
        if last_family_mode and key_fm != last_family_mode:
            # New family/mode: session contexts may reset.
            pass
        last_family_mode = key_fm
        if not rung.decoding_sessions:
            errors.append(f"{rung.rung_id}: no decoding sessions")
        if set(rung.decoding_sessions) - set(rung.live_sessions):
            errors.append(f"{rung.rung_id}: decoding not a subset of live")
        if set(rung.grow_sessions) - set(rung.live_sessions):
            errors.append(f"{rung.rung_id}: grow not a subset of live")
        if len(rung.decoding_sessions) != rung.users:
            errors.append(f"{rung.rung_id}: users={rung.users} but decoding={rung.decoding_sessions}")
        for sid in rung.live_sessions:
            prev = highwater.get((rung.family, rung.load_mode, sid), 0)
            target = rung.target_for(sid)
            if target < prev:
                errors.append(
                    f"{rung.rung_id}: session {sid} would shrink from {prev} to {target}"
                )
            if sid in rung.grow_sessions or sid in rung.decoding_sessions:
                highwater[(rung.family, rung.load_mode, sid)] = max(prev, target)
            else:
                # Quiesced sessions stay at the previous high-water mark; after a
                # family-wide grow they are already at ctx_per_user.
                highwater[(rung.family, rung.load_mode, sid)] = max(prev, prev)
    return errors


def peak_live_tokens(rungs: list[Rung]) -> int:
    """Peak of (number of live sessions × context) over the schedule."""
    peak = 0
    for rung in rungs:
        total = sum(rung.target_for(sid) for sid in rung.live_sessions)
        peak = max(peak, total)
    return peak
