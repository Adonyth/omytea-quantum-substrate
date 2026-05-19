"""Tests for `omytea.dynamics.lindblad.LindbladOperator`.

Key invariant: under a dissipative Lindblad operator, off-diagonal
magnitudes in the joint wavefunction should monotonically decrease
(or stay equal) tick-to-tick.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import product

import pytest

from omytea.dynamics.lindblad import LindbladOperator
from omytea.dynamics.protocol import OperatorContext
from omytea.joint_belief import (
    JointBranchHypothesis,
    JointWaveFunction,
    OffDiagonalEntry,
)
from omytea.models import Position
from omytea.quantum import StateHypothesis, WaveFunction


_NOW = datetime(2026, 5, 19, tzinfo=timezone.utc)


def _make_wf(entity_id: str, n: int = 3) -> WaveFunction:
    hyps = tuple(
        StateHypothesis(
            object_id=entity_id, label=f"h{i}", stream_id="t",
            timestamp=_NOW, position=Position(x=0.1 * i, y=0.5),
            weight=1.0 / n, branch_label=f"h{i}",
        )
        for i in range(n)
    )
    return WaveFunction(
        object_id=entity_id, label=entity_id, stream_id="t",
        timestamp=_NOW, hypotheses=hyps, action_arm=None,
    )


def _make_joint_with_coherence(amp: float = 0.1) -> JointWaveFunction:
    """Build a 2-entity joint state with a single Hermitian off-diagonal pair."""
    wf_a = _make_wf("A", n=3)
    wf_b = _make_wf("B", n=3)
    joint_hyps = []
    for h_a, h_b in product(wf_a.hypotheses, wf_b.hypotheses):
        joint_hyps.append(JointBranchHypothesis(
            branch_refs={"A": h_a.hypothesis_id, "B": h_b.hypothesis_id},
            weight=h_a.weight * h_b.weight,
        ))
    # Hermitian off-diagonal pair between joint hypotheses 0 and 1
    offdiags = (
        OffDiagonalEntry(row=0, col=1, amplitude=complex(amp, 0.0)),
        OffDiagonalEntry(row=1, col=0, amplitude=complex(amp, 0.0)),
    )
    return JointWaveFunction(
        entity_ids=("A", "B"),
        hypotheses=tuple(joint_hyps),
        off_diagonal_couplings=offdiags,
    )


# --------------------------------------------------------------
# Construction + basic shape
# --------------------------------------------------------------


def test_lindblad_construction_with_default_rate() -> None:
    op = LindbladOperator()
    assert op.name == "lindblad"
    assert op.decoherence_rate == 0.1


def test_lindblad_with_custom_rate() -> None:
    op = LindbladOperator(decoherence_rate=0.25)
    assert op.decoherence_rate == 0.25


# --------------------------------------------------------------
# Core invariant: off-diagonal magnitudes monotonically decay
# --------------------------------------------------------------


def test_lindblad_monotonic_decay_one_tick() -> None:
    """After one tick at γ > 0, off-diagonal magnitude is strictly smaller."""
    jwf = _make_joint_with_coherence(amp=0.1)
    op = LindbladOperator(decoherence_rate=0.1)
    ctx = OperatorContext(scenario_name="test", tick=0)
    initial_mag = abs(jwf.off_diagonal_couplings[0].amplitude)
    evolved = op.evolve(jwf, dt=1.0, ctx=ctx)
    assert isinstance(evolved, JointWaveFunction)
    new_mag = abs(evolved.off_diagonal_couplings[0].amplitude)
    assert new_mag < initial_mag


def test_lindblad_monotonic_decay_over_many_ticks() -> None:
    """Over a longer horizon, every tick the off-diagonal magnitude
    decreases (or stays equal). This is the dissipative invariant."""
    jwf = _make_joint_with_coherence(amp=0.1)
    op = LindbladOperator(decoherence_rate=0.08)
    ctx = OperatorContext(scenario_name="test", tick=0)

    magnitudes = [abs(jwf.off_diagonal_couplings[0].amplitude)]
    current = jwf
    for _ in range(10):
        current = op.evolve(current, dt=1.0, ctx=ctx)
        magnitudes.append(abs(current.off_diagonal_couplings[0].amplitude))

    # Every step is ≤ the previous step.
    for i in range(1, len(magnitudes)):
        assert magnitudes[i] <= magnitudes[i - 1] + 1e-9, (
            f"Non-monotonic at tick {i}: {magnitudes[i-1]} → {magnitudes[i]}"
        )


def test_lindblad_decay_approaches_zero_as_horizon_grows() -> None:
    """At γ > 0, off-diagonal magnitudes should approach 0 asymptotically."""
    jwf = _make_joint_with_coherence(amp=0.1)
    op = LindbladOperator(decoherence_rate=0.3)
    ctx = OperatorContext(scenario_name="test", tick=0)
    current = jwf
    for _ in range(50):
        current = op.evolve(current, dt=1.0, ctx=ctx)
    final_mag = abs(current.off_diagonal_couplings[0].amplitude)
    # After 50 ticks at γ=0.3 ≈ e^{-15} ≈ 3e-7
    assert final_mag < 1e-3


def test_lindblad_zero_gamma_preserves_coherence() -> None:
    """At γ = 0, off-diagonals should not decay (within numerical tolerance)."""
    jwf = _make_joint_with_coherence(amp=0.1)
    op = LindbladOperator(decoherence_rate=0.0)
    ctx = OperatorContext(scenario_name="test", tick=0)
    current = jwf
    for _ in range(10):
        current = op.evolve(current, dt=1.0, ctx=ctx)
    final_mag = abs(current.off_diagonal_couplings[0].amplitude)
    assert abs(final_mag - 0.1) < 1e-6


# --------------------------------------------------------------
# Multiple off-diagonal pairs: each decays independently
# --------------------------------------------------------------


def test_lindblad_multiple_pairs_all_decay() -> None:
    """If the joint state carries multiple off-diagonal pairs, every one
    should monotonically decay."""
    wf_a = _make_wf("A", n=3)
    wf_b = _make_wf("B", n=3)
    joint_hyps = []
    for h_a, h_b in product(wf_a.hypotheses, wf_b.hypotheses):
        joint_hyps.append(JointBranchHypothesis(
            branch_refs={"A": h_a.hypothesis_id, "B": h_b.hypothesis_id},
            weight=h_a.weight * h_b.weight,
        ))
    # Three Hermitian pairs at different amplitudes
    offdiags = []
    for (i, j), amp in [((0, 1), 0.10), ((2, 3), 0.05), ((4, 5), 0.20)]:
        offdiags.append(OffDiagonalEntry(row=i, col=j, amplitude=complex(amp)))
        offdiags.append(OffDiagonalEntry(row=j, col=i, amplitude=complex(amp)))

    jwf = JointWaveFunction(
        entity_ids=("A", "B"),
        hypotheses=tuple(joint_hyps),
        off_diagonal_couplings=tuple(offdiags),
    )

    op = LindbladOperator(decoherence_rate=0.1)
    ctx = OperatorContext(scenario_name="test", tick=0)
    initial_mags = sorted(abs(e.amplitude) for e in jwf.off_diagonal_couplings)
    evolved = op.evolve(jwf, dt=1.0, ctx=ctx)
    final_mags = sorted(abs(e.amplitude) for e in evolved.off_diagonal_couplings)
    # Every magnitude should have decreased.
    for i, f in zip(initial_mags, final_mags):
        assert f < i + 1e-9


# --------------------------------------------------------------
# Joint structure preservation
# --------------------------------------------------------------


def test_lindblad_preserves_joint_cardinality() -> None:
    """Lindblad operates on off-diagonals; |hypotheses| should not change."""
    jwf = _make_joint_with_coherence()
    op = LindbladOperator(decoherence_rate=0.1)
    ctx = OperatorContext(scenario_name="test", tick=0)
    evolved = op.evolve(jwf, dt=1.0, ctx=ctx)
    assert len(evolved.hypotheses) == len(jwf.hypotheses)


def test_lindblad_preserves_entity_ids() -> None:
    jwf = _make_joint_with_coherence()
    op = LindbladOperator()
    ctx = OperatorContext(scenario_name="test", tick=0)
    evolved = op.evolve(jwf, dt=1.0, ctx=ctx)
    assert evolved.entity_ids == jwf.entity_ids
