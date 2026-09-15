"""Run-artifact reference integrity: a cited run id must RESOLVE, must not be a dead run quoted
as a result, and must not be truncated into something unresolvable.

WHY THIS EXISTS. Provenance pointers here are written by hand, and hand-written pointers have
failed repeatedly, each time caught only because someone went looking: four ids transcribed from
memory that resolve to nothing (decision log 2026-08-22T11:25Z), two citations given as a bare
timestamp with no run name (00:55Z and 01:35Z, stated stamps), one entry citing the run whose
control FAILED while the passing run sat beside it (00:55Z), and then six MORE dangling ids that
all three hand-audits missed, found by this file on its first run (2026-08-22T21:26Z). The
documentation rules require mechanically-checkable cross-references to be checked mechanically.
This is that check, and its first act was to prove the hand-audits were not sufficient.

THREE RULES.
  RULE 1  every `20\\d{6}T\\d{6}Z_<name>` cited in a scanned file exists under `results/runs/`
  RULE 2  a cited run that produced NO payload is flagged, because that is how a killed or crashed
          attempt gets quoted as a finding — the defect that reached the N1 entry. "Payload" is
          any file that is not run boilerplate: runners here write `results.json`,
          `data_efficiency.json` or `report_summary.json` depending on the job, so keying on one
          filename would flag a dozen healthy runs. A `results_partial.json` alone is NOT a
          payload — a partial means the run did not finish, which is exactly the dangerous case.
  RULE 3  a citation truncated with an ellipsis (`20260822T081917Z...`) is unresolvable and
          therefore invisible to RULE 1. Two existed; both were wrong in some other way too.
  RULE 4  a BARE compact stamp (`20260822T184418Z`, no run name) that resolves to exactly one run
          directory by prefix IS a citation, written unresolvably. Prose timestamps are unaffected:
          they carry ISO separators (`2026-08-22T18:44Z`) and never match the compact pattern, and
          a stamp matching zero or several runs is skipped as a moment rather than an id.

EXEMPTIONS, narrow and explicit.
  (a) A CORRECTION entry must quote the wrong id to correct it, so an id is exempt from RULE 1
      when a RESOLVING id appears on the same line. A prototype without this reported 8 failures
      where only 4 were real.
  (b) `DOCUMENTED_DANGLING` pins ids that are known-wrong and whose correction is logged. Each
      entry names its true artifact, so the allowlist is a receipt, not an amnesty: a NEW dangling
      id still fails. Entries are never edited out of the log (append-only), which is why the
      historical wrong ids must live here instead.
  (c) A run that died may be cited as having died — RULE 2 is waived when the citing line or
      either line beside it says so in words.

`results/runs/` is gitignored, so every rule SKIPS rather than fails when it is absent: on a fresh
clone or in CI there is nothing to resolve against and a red suite there would be noise.

BARE STAMPS AND THE GRANDFATHER LINE. RULE 4 closes the gap this file originally declared out of
scope. Of 43 bare compact stamps, 38 resolve to exactly one run and are therefore citations written
unresolvably; 5 resolve to nothing and are moments. The five outside the decision log were FIXED
rather than excused (two `BACKLOG.md` citations written out in full, two experiment-script comments,
and one illustrative example switched to a non-resolving stamp). The 33 inside the log cannot be
fixed — entries are immutable — so the log is grandfathered up to `LOG_GRANDFATHER_LINE` and
enforced strictly beyond it. A line cutoff is sound here for one specific reason: the log is
append-only by contract, so existing line numbers never move, and if that contract were ever broken
the test starts failing loudly rather than passing silently.

NOT IN SCOPE, stated so the gap is not mistaken for coverage. (i) Entry timestamps written from the
wrong clock (the 2026-08-22T19:48Z timezone defect). (ii) Backlog rows left open after their finding
landed. (iii) The reverse direction — a run that exists but is cited nowhere; deliberately not a
rule, because a two-trial smoke pass does not need a log entry and a rule demanding one would
generate noise that trains the operator to ignore the suite (see the 2026-08-23T00:46Z entry). All
three stay on operator discipline.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
RUN_ID = re.compile(r"20\d{6}T\d{6}Z_[A-Za-z0-9_]+")
TRUNCATED = re.compile(r"20\d{6}T\d{6}Z_?[A-Za-z0-9_]*\.\.\.")
BARE = re.compile(r"20\d{6}T\d{6}Z(?![_A-Za-z0-9])")

# The decision log is append-only, so line numbers of existing content never move. Everything at or
# below this line predates RULE 4 and is grandfathered; everything appended after it is enforced.
LOG_GRANDFATHER_LINE = 10834

# Files a runner writes for every run regardless of outcome. Anything else is a payload.
BOILERPLATE = {
    "git_sha.txt",
    "lock_hash.txt",
    "params.yaml",
    "metrics.json",
    "compute_env.json",
    "results_partial.json",  # a partial means the run did not finish — not a result
}

# Known-wrong ids whose correction is logged. Value = the artifact actually meant.
# Added 2026-08-22T21:26Z; see the correction entry of that stamp for how each was resolved.
DOCUMENTED_DANGLING = {
    "20260815T104438Z_probe_operating_point_full_binary": (
        "20260815T113832Z_probe_operating_point_full_binary"
    ),
    "20260815T121109Z_probe_hsj_restarts_binary_q8": (
        "20260815T113707Z_probe_hsj_restarts_binary_q8"
    ),
    "20260822T075224Z_shot_attack_bridge_binary_q8": (
        "20260822T075249Z_shot_attack_bridge_binary_q8"
    ),
    "20260822T080845Z_projected_kernel_binary_q8": "20260822T080846Z_projected_kernel_binary_q8",
    "20260822T081235Z_evasion_max_agg_binary_q8": "20260822T081217Z_evasion_max_agg_binary_q8",
    "20260822T090450Z_group_blackbox_control_binary_q8": (
        "20260822T084835Z_group_blackbox_control_binary_q8"
    ),
}

# Dead runs cited as the artifact behind a finding, where the citing entry is immutable.
# Value = the run that actually holds the numbers. Same shape of defect as the N1 citation.
DOCUMENTED_DEAD_CITED = {
    "20260822T075536Z_poisoning_seeds_binary_q8": "20260822T082913Z_poisoning_seeds_binary_q8",
}

# Truncated citations that predate RULE 3. Both are also corrected in the 21:26Z entry.
DOCUMENTED_TRUNCATED = {"20260822T081917Z...", "20260822T080113Z..."}

# Words that admit, next to the pointer, that the run produced nothing.
DEAD_RUN_WORDS = (
    "killed",
    "kill",
    "crashed",
    "crash",
    "discarded",
    "discard",
    "failed",
    "failing",
    "incomplete",
    "absent",
    "did not reach",
    "no results",
    "never wrote",
    "wrote nothing",
    "superseded",
    "stopped",
    "aborted",
    "partial",
)


def _scanned_files() -> list[Path]:
    """Docs and scripts carrying provenance pointers a reader is expected to follow."""
    files = [ROOT / "experiment_decision_log.md", ROOT / "BACKLOG.md"]
    files += sorted((ROOT / "reports").glob("*.md"))
    files += sorted((ROOT / "experiments").glob("*.py"))
    files += sorted((ROOT / "paper" / "tables").glob("*_meta.json"))
    return [f for f in files if f.exists()]


def _on_disk() -> set[str]:
    return {d.name for d in RUNS.iterdir() if d.is_dir()}


def _has_payload(rid: str) -> bool:
    """True when the run wrote something beyond boilerplate.

    Deliberately NOT gated on `metrics.json` status: a run can write a complete payload and then
    raise in a later stage — `20260813T183453Z_adversarial_binary_q8_blackbox` wrote `results.json`
    and then died on `KeyError('evasion')`, and the paper's 2.818 sigma figure comes from it. A
    status veto flagged that healthy artifact while catching nothing a payload check misses, since
    a run that errored before writing anything already fails here.
    """
    return any(f.name not in BOILERPLATE for f in (RUNS / rid).iterdir() if f.is_file())


def _cited(path: Path) -> list[tuple[int, str, str]]:
    """(line number, id, line text) for every run id cited in the file."""
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        for m in RUN_ID.finditer(line):
            out.append((i + 1, m.group(0), line))
    return out


def _skip() -> bool:
    return not RUNS.exists() or not _on_disk()


def test_every_cited_run_id_resolves():
    """RULE 1 — a pointer to a run that does not exist is a dangling reference."""
    if _skip():
        return
    on_disk = _on_disk()

    failures = []
    for path in _scanned_files():
        for lineno, rid, line in _cited(path):
            if rid in on_disk or rid in DOCUMENTED_DANGLING:
                continue
            others = [o for o in RUN_ID.findall(line) if o != rid]
            if any(o in on_disk for o in others):
                continue  # exemption (a): a correction quotes the wrong id beside the right one
            failures.append(f"{path.relative_to(ROOT)}:{lineno} cites {rid}, which does not exist")

    assert not failures, "dangling run-artifact references:\n" + "\n".join(failures)


def test_no_cited_run_is_a_dead_run_quoted_as_a_result():
    """RULE 2 — a cited run that wrote no payload must be labelled as having produced none.

    This is the half that matters. The N1 entry cited the run whose control FAILED while the
    passing run sat beside it on disk, and any existence check would have passed that citation.
    """
    if _skip():
        return
    on_disk = _on_disk()

    failures = []
    for path in _scanned_files():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for lineno, rid, line in _cited(path):
            if rid not in on_disk or _has_payload(rid) or rid in DOCUMENTED_DEAD_CITED:
                continue
            window = " ".join(lines[max(0, lineno - 3) : lineno + 2]).lower()
            if any(w in window for w in DEAD_RUN_WORDS):
                continue  # exemption (c): the death is written down next to the pointer
            failures.append(
                f"{path.relative_to(ROOT)}:{lineno} cites {rid}, which wrote no payload "
                "and is not described as killed/discarded/failed"
            )

    assert not failures, "dead runs cited as results:\n" + "\n".join(failures)


def test_no_citation_is_truncated_into_unresolvability():
    """RULE 3 — `20260822T081917Z...` is a pointer no reader and no linter can follow."""
    if _skip():
        return
    failures = []
    for path in _scanned_files():
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
            for m in TRUNCATED.finditer(line):
                if m.group(0) in DOCUMENTED_TRUNCATED:
                    continue
                failures.append(f"{path.relative_to(ROOT)}:{i + 1} cites {m.group(0)}, truncated")

    assert not failures, "truncated run-artifact citations:\n" + "\n".join(failures)


def test_no_run_is_cited_by_a_bare_timestamp():
    """RULE 4 — a bare stamp that resolves to exactly one run is a citation written unresolvably."""
    if _skip():
        return
    on_disk = sorted(_on_disk())

    failures = []
    for path in _scanned_files():
        is_log = path.name == "experiment_decision_log.md"
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
            if is_log and i + 1 <= LOG_GRANDFATHER_LINE:
                continue
            for m in BARE.finditer(line):
                hits = [d for d in on_disk if d.startswith(m.group(0))]
                if len(hits) != 1:
                    continue  # zero or several matches: a moment, or ambiguous — not a citation
                failures.append(
                    f"{path.relative_to(ROOT)}:{i + 1} cites {m.group(0)} bare; write it in full "
                    f"as {hits[0]}"
                )

    assert not failures, "run ids cited by bare timestamp:\n" + "\n".join(failures)


def test_the_linter_would_actually_catch_the_defects_it_was_built_for():
    """Positive control — a linter that silently matched nothing would read as coverage."""
    on_disk = _on_disk() if RUNS.exists() else set()

    fake = "20260101T000000Z_this_run_never_existed"
    assert fake not in on_disk
    assert RUN_ID.fullmatch(fake), "the id pattern no longer matches a well-formed run id"
    assert TRUNCATED.search("artifact 20260822T081917Z..."), "RULE 3's pattern no longer fires"
    assert BARE.search("run 20260822T184418Z here"), "RULE 4's bare pattern no longer fires"
    assert not BARE.search("logged at 2026-08-22T18:44Z"), (
        "RULE 4 now matches ISO prose timestamps; it would flag every entry header"
    )
    assert not BARE.search("20260822T184418Z_taxonomy_scope_binary_q8"), (
        "RULE 4 matches full run ids; it would flag every correct citation"
    )

    # the correction shape must stay exempt: wrong id and right id on one line
    if on_disk:
        real = sorted(on_disk)[0]
        others = [o for o in RUN_ID.findall(f"cites {fake}, correct id {real}") if o != fake]
        assert any(o in on_disk for o in others), "the correction exemption no longer fires"

    # every documented-dangling id must point at an artifact that really exists, or the
    # allowlist has become a place to hide unresolved pointers
    if on_disk:
        for wrong, right in DOCUMENTED_DANGLING.items():
            assert wrong not in on_disk, f"{wrong} now exists; remove it from DOCUMENTED_DANGLING"
            assert right in on_disk, f"{wrong} is excused by {right}, which does not exist either"
        for dead, live in DOCUMENTED_DEAD_CITED.items():
            assert not _has_payload(dead), f"{dead} has a payload now; drop the exemption"
            assert _has_payload(live), f"{dead} is excused by {live}, which has no payload either"


def test_paper_sources_carry_no_mangled_escapes():
    """A raw CR byte where a macro's backslash-letter should be typesets as garbled text.

    THE DEFECT THIS PINS (BACKLOG, pass-4 audit 2026-08-22): commit e2adf69 turned the
    backslash-r of ``\ref{sec:rq1}`` in methodology.tex into a literal 0x0D byte, and
    neither build failed -- pdflatex typeset the mangled text silently. Two greps: no bare
    carriage return in any paper source, and no orphaned ``ef{`` (the residue such a
    mangling leaves) outside a real ``\ref{``/``\\eqref{``/similar macro.
    """
    import re as _re

    paper = Path(__file__).resolve().parents[1] / "paper"
    sources = sorted(paper.glob("sections/*.tex")) + [
        p for p in (paper / "main.tex", paper / "arxiv" / "main.tex") if p.exists()
    ]
    assert sources, "no paper sources found"
    for src in sources:
        raw = src.read_bytes()
        bare_cr = _re.findall(rb"\x0d(?!\x0a)", raw)
        assert not bare_cr, f"{src.name}: {len(bare_cr)} bare CR byte(s) -- a mangled escape"
        orphans = [m.start() for m in _re.finditer(rb"(?<![\\A-Za-z])ef\{", raw)]
        assert not orphans, f"{src.name}: orphaned 'ef{{' at byte(s) {orphans}"
