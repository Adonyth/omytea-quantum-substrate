"""Tests for `omytea.quantum` — StateHypothesis + WaveFunction primitives."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from omytea.models import Position
from omytea.quantum import StateHypothesis, WaveFunction


_NOW = datetime(2026, 5, 19, tzinfo=timezone.utc)


def _hyp(label: str, weight: float, x: float = 0.5) -> StateHypothesis:
    return StateHypothesis(
        object_id="A",
        label=label,
        stream_id="test",
        timestamp=_NOW,
        position=Position(x=x, y=0.5),
        weight=weight,
        branch_label=label,
    )


# --------------------------------------------------------------
# StateHypothesis
# --------------------------------------------------------------


def test_state_hypothesis_basic_fields() -> None:
    h = _hyp("continue", weight=0.6)
    assert h.object_id == "A"
    assert h.label == "continue"
    assert h.weight == 0.6
    assert h.position.x == 0.5
    assert h.velocity == (0.0, 0.0, 0.0)


def test_state_hypothesis_id_is_unique() -> None:
    h1 = _hyp("continue", 0.5)
    h2 = _hyp("continue", 0.5)
    # Two hypotheses with the same shape still get distinct UUIDs.
    assert h1.hypothesis_id != h2.hypothesis_id


def test_state_hypothesis_with_velocity() -> None:
    h = StateHypothesis(
        object_id="A", label="moving", stream_id="t",
        timestamp=_NOW, position=Position(x=0.0, y=0.0),
        velocity=(0.5, 0.3, 0.0),
        weight=1.0,
    )
    assert h.velocity == (0.5, 0.3, 0.0)


def test_state_hypothesis_attributes_default_empty() -> None:
    assert _hyp("c", 1.0).attributes == {}


# --------------------------------------------------------------
# WaveFunction
# --------------------------------------------------------------


def test_wavefunction_construction() -> None:
    hyps = (_hyp("a", 0.5), _hyp("b", 0.5))
    wf = WaveFunction(
        object_id="A", label="entity_a", stream_id="t",
        timestamp=_NOW, hypotheses=hyps, action_arm=None,
    )
    assert len(wf.hypotheses) == 2
    assert wf.action_arm is None


def test_wavefunction_hypotheses_are_immutable_tuple() -> None:
    """``hypotheses`` must be a tuple, not a list (immutability invariant)."""
    hyps = (_hyp("a", 1.0),)
    wf = WaveFunction(
        object_id="A", label="x", stream_id="t",
        timestamp=_NOW, hypotheses=hyps, action_arm=None,
    )
    assert isinstance(wf.hypotheses, tuple)


def test_wavefunction_weights_can_sum_to_one() -> None:
    """We don't enforce normalization at construction, but a properly-built
    wavefunction's weights should sum to ~1.0."""
    hyps = (
        _hyp("continue",   0.55),
        _hyp("accelerate", 0.25),
        _hyp("decelerate", 0.20),
    )
    wf = WaveFunction(
        object_id="A", label="entity", stream_id="t",
        timestamp=_NOW, hypotheses=hyps, action_arm=None,
    )
    total = sum(h.weight for h in wf.hypotheses)
    assert abs(total - 1.0) < 1e-9


def test_wavefunction_with_action_arm() -> None:
    """`action_arm` marker tags the WaveFunction as Pearl-rung-2 (interventional)."""
    wf = WaveFunction(
        object_id="A", label="entity", stream_id="t",
        timestamp=_NOW,
        hypotheses=(_hyp("c", 1.0),),
        action_arm="do(T=apply_brake)",
    )
    assert wf.action_arm == "do(T=apply_brake)"
