"""Tests for `omytea.joint_belief` — JointWaveFunction + OffDiagonalEntry."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import product

import pytest

from omytea.joint_belief import (
    JointBranchHypothesis,
    JointWaveFunction,
    OffDiagonalEntry,
)
from omytea.models import Position
from omytea.quantum import StateHypothesis, WaveFunction


_NOW = datetime(2026, 5, 19, tzinfo=timezone.utc)


def _make_3hyp_wf(entity_id: str) -> WaveFunction:
    hyps = tuple(
        StateHypothesis(
            object_id=entity_id, label=name, stream_id="t",
            timestamp=_NOW, position=Position(x=cx, y=0.5),
            weight=w, branch_label=name,
        )
        for name, cx, w in [
            ("continue",   0.6, 0.55),
            ("accelerate", 0.9, 0.25),
            ("decelerate", 0.3, 0.20),
        ]
    )
    return WaveFunction(
        object_id=entity_id, label=entity_id, stream_id="t",
        timestamp=_NOW, hypotheses=hyps, action_arm=None,
    )


def _cartesian_joint(wfs: list[WaveFunction]) -> JointWaveFunction:
    """Helper: build a JointWaveFunction by Cartesian product."""
    entity_ids = tuple(wf.object_id for wf in wfs)
    joint_hyps: list[JointBranchHypothesis] = []
    for combo in product(*(wf.hypotheses for wf in wfs)):
        branch_refs = {
            wf.object_id: h.hypothesis_id for wf, h in zip(wfs, combo)
        }
        weight = 1.0
        for h in combo:
            weight *= h.weight
        joint_hyps.append(JointBranchHypothesis(
            branch_refs=branch_refs, weight=weight,
        ))
    return JointWaveFunction(
        entity_ids=entity_ids,
        hypotheses=tuple(joint_hyps),
        off_diagonal_couplings=(),
    )


# --------------------------------------------------------------
# JointBranchHypothesis
# --------------------------------------------------------------


def test_joint_branch_hypothesis_construction() -> None:
    jbh = JointBranchHypothesis(
        branch_refs={"A": "uuid_a1", "B": "uuid_b1"},
        weight=0.137,
    )
    assert jbh.branch_refs["A"] == "uuid_a1"
    assert jbh.weight == pytest.approx(0.137)


# --------------------------------------------------------------
# JointWaveFunction — Cartesian product invariant
# --------------------------------------------------------------


def test_joint_cardinality_2_entities_3_each() -> None:
    """|JointWaveFunction| = product of per-entity |hypotheses|."""
    wf_a = _make_3hyp_wf("A")
    wf_b = _make_3hyp_wf("B")
    jwf = _cartesian_joint([wf_a, wf_b])
    assert len(jwf.hypotheses) == 9
    assert jwf.entity_ids == ("A", "B")


def test_joint_cardinality_3_entities_3_each() -> None:
    jwf = _cartesian_joint([
        _make_3hyp_wf("A"), _make_3hyp_wf("B"), _make_3hyp_wf("C"),
    ])
    assert len(jwf.hypotheses) == 27
    assert jwf.entity_ids == ("A", "B", "C")


def test_joint_branch_refs_have_correct_entity_keys() -> None:
    jwf = _cartesian_joint([_make_3hyp_wf("A"), _make_3hyp_wf("B")])
    for h in jwf.hypotheses:
        assert set(h.branch_refs.keys()) == {"A", "B"}


def test_joint_weights_sum_to_one_for_normalised_inputs() -> None:
    """If per-entity weights sum to 1.0, joint weights sum to 1.0 too."""
    jwf = _cartesian_joint([_make_3hyp_wf("A"), _make_3hyp_wf("B")])
    total = sum(h.weight for h in jwf.hypotheses)
    assert abs(total - 1.0) < 1e-9


# --------------------------------------------------------------
# OffDiagonalEntry
# --------------------------------------------------------------


def test_offdiag_entry_basic() -> None:
    e = OffDiagonalEntry(row=0, col=1, amplitude=complex(0.1, 0.0))
    assert e.row == 0
    assert e.col == 1
    assert e.amplitude == complex(0.1, 0.0)


def test_offdiag_hermitian_pair_via_conjugate() -> None:
    """A Hermitian off-diagonal pair has (i,j) and (j,i) with conjugate
    amplitudes."""
    upper = OffDiagonalEntry(row=0, col=1, amplitude=complex(0.10, 0.05))
    lower = OffDiagonalEntry(
        row=1, col=0, amplitude=upper.amplitude.conjugate(),
    )
    assert lower.amplitude == complex(0.10, -0.05)
    assert lower.amplitude == upper.amplitude.conjugate()


def test_joint_wavefunction_can_carry_off_diagonals() -> None:
    """JointWaveFunction accepts a tuple of OffDiagonalEntries at
    construction time."""
    jwf = _cartesian_joint([_make_3hyp_wf("A"), _make_3hyp_wf("B")])
    e_up = OffDiagonalEntry(row=0, col=1, amplitude=complex(0.10, 0.0))
    e_lo = OffDiagonalEntry(row=1, col=0, amplitude=complex(0.10, 0.0))
    jwf_with_coh = JointWaveFunction(
        entity_ids=jwf.entity_ids,
        hypotheses=jwf.hypotheses,
        off_diagonal_couplings=(e_up, e_lo),
    )
    assert len(jwf_with_coh.off_diagonal_couplings) == 2
