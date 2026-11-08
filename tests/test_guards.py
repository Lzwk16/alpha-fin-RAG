from alpha_audit.guards import check_grounding, check_numbers
from alpha_audit.guards.grounding import marker_ids
from alpha_audit.schemas import Answer, Citation, Conflict, Evidence

E1 = Evidence("c1", "d1", "Fund Review", "KMLM = 5%", "The net expense ratio is 1.27% since 2020.", 0.9, 0.5)
E2 = Evidence("c2", "d2", "Other", "Fees", "Another article says the fee is 1.30%.", 0.8, 0.5)


def answer(text="It costs 1.27%. [E1]", citations=(), status="ANSWERED", conflicts=()):
    return Answer(analysis="", status=status, answer=text, citations=list(citations), conflicts=list(conflicts))


def kinds(flags):
    return [f.kind for f in flags]


def test_verbatim_quote_is_kept_ignoring_case_and_space():
    a, flags = check_grounding(answer(citations=[Citation(evidence_id="E1", quote="the NET  expense ratio")]), [E1])
    assert len(a.citations) == 1 and flags == []


def test_heading_quote_is_kept():
    a, flags = check_grounding(answer(citations=[Citation(evidence_id="E1", quote="KMLM = 5%")]), [E1])
    assert len(a.citations) == 1 and flags == []


def test_paraphrase_and_wrong_excerpt_are_dropped():
    cits = [Citation(evidence_id="E1", quote="the fee is 1.27%"), Citation(evidence_id="E2", quote="KMLM = 5%")]
    a, flags = check_grounding(answer(citations=cits), [E1, E2])
    assert a.citations == []
    assert kinds(flags) == ["quote_not_in_evidence", "quote_not_in_evidence", "ungrounded"]


def test_unknown_ids_are_flagged():
    a, flags = check_grounding(answer("See [E9].", [Citation(evidence_id="E9", quote="x")]), [E1])
    assert a.citations == [] and "unknown_evidence_id" in kinds(flags) and "unknown_marker" in kinds(flags)


def test_conflict_positions_are_checked():
    c = Conflict(topic="fee", positions=[Citation(evidence_id="E1", quote="made up"),
                                         Citation(evidence_id="E2", quote="the fee is 1.30%")], likely_reason="date")
    a, _ = check_grounding(answer(citations=[Citation(evidence_id="E1", quote="net expense")], conflicts=[c]), [E1, E2])
    assert [p.evidence_id for p in a.conflicts[0].positions] == ["E2"]


def test_markers_handle_two_digits_and_lists():
    assert marker_ids("a [E10] b [E1, E9]") == ["E10", "E1", "E9"]


def test_numeric_accepts_cited_rounding_and_heading_figures():
    assert check_numbers(answer("It costs 1.3%. [E1]"), [E1], "q") == []
    assert check_numbers(answer("KMLM is 5%. [E1]"), [E1], "q") == []


def test_numeric_flags_uncited_and_changed_figures():
    assert [f.kind for f in check_numbers(answer("It costs 1.30%. [E1]"), [E1, E2], "q")] == ["unsupported_percent"]


def test_numeric_allows_figures_from_the_question():
    assert check_numbers(answer("It is not 0.45%; it is 1.27%. [E1]"), [E1], "Why is it 0.45%?") == []
