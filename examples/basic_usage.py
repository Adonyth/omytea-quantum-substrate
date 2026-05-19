"""Basic end-to-end example: 2-entity joint state + Lindblad evolution.

Run:
    python examples/basic_usage.py

Output is a printed table of off-diagonal magnitudes per Lindblad tick,
demonstrating monotonic decay.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import product

from omytea import (
    JointBranchHypothesis,
    JointWaveFunction,
    OffDiagonalEntry,
    Position,
    StateHypothesis,
    WaveFunction,
)
from omytea.dynamics import LindbladOperator, OperatorContext


def main() -> None:
    now = datetime.now(tz=timezone.utc)

    # --- Step 1: build per-entity WaveFunctions ---
    # Each entity has 3 future-position hypotheses: continue / accelerate /
    # decelerate, with priors 55% / 25% / 20%.

    def make_wf(entity_id: str, label: str) -> WaveFunction:
        names_x_weights = [
            ("continue",   0.6, 0.55),
            ("accelerate", 0.9, 0.25),
            ("decelerate", 0.3, 0.20),
        ]
        hyps = tuple(
            StateHypothesis(
                object_id=entity_id, label=name, stream_id="demo",
                timestamp=now, position=Position(x=cx, y=0.5),
                weight=w, branch_label=name,
            )
            for name, cx, w in names_x_weights
        )
        return WaveFunction(
            object_id=entity_id, label=label, stream_id="demo",
            timestamp=now, hypotheses=hyps, action_arm=None,
        )

    wf_a = make_wf("A", "left_to_right_walker")
    wf_b = make_wf("B", "right_to_left_walker")

    # --- Step 2: build the Cartesian-product joint state (3×3 = 9) ---

    joint_hyps = [
        JointBranchHypothesis(
            branch_refs={"A": h_a.hypothesis_id, "B": h_b.hypothesis_id},
            weight=h_a.weight * h_b.weight,
        )
        for h_a, h_b in product(wf_a.hypotheses, wf_b.hypotheses)
    ]
    print(f"Joint cardinality: {len(joint_hyps)} (expected 9)")

    # --- Step 3: add off-diagonal correlations ---
    # Heuristic: pair joint cells 0 and 1 with a small positive coherence.
    # In a real model you'd derive this from scene-level coupling priors.

    offdiags = (
        OffDiagonalEntry(row=0, col=1, amplitude=complex(0.10, 0.0)),
        OffDiagonalEntry(row=1, col=0, amplitude=complex(0.10, 0.0)),
    )

    jwf = JointWaveFunction(
        entity_ids=("A", "B"),
        hypotheses=tuple(joint_hyps),
        off_diagonal_couplings=offdiags,
    )

    # --- Step 4: evolve under Lindblad at γ = 0.08 ---

    lindblad = LindbladOperator(decoherence_rate=0.08)
    ctx = OperatorContext(scenario_name="basic_usage_demo", tick=0)

    print("\nLindblad evolution (γ = 0.08):")
    print(f"  tick    off-diag magnitude")
    print(f"  ----    ------------------")
    current = jwf
    for tick in range(11):
        mag = abs(current.off_diagonal_couplings[0].amplitude)
        print(f"  {tick:>4d}    {mag:.5f}")
        if tick < 10:
            current = lindblad.evolve(current, dt=1.0, ctx=ctx)

    print(
        "\nThe magnitudes decrease monotonically — that's the dissipative "
        "Lindblad invariant.\n"
    )


if __name__ == "__main__":
    main()
