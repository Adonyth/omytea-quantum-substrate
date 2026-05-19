"""Omytea dynamics — open-system evolution operators.

Public API:

>>> from omytea.dynamics.lindblad import LindbladOperator
>>> from omytea.dynamics.protocol import (
...     BeliefState, JointBeliefState, OperatorContext,
... )

The Lindblad operator dissipates off-diagonal coherence on a
``JointWaveFunction`` at a configurable decoherence rate γ. See
``LindbladOperator.evolve`` for the per-tick update rule.
"""

from omytea.dynamics.lindblad import LindbladOperator
from omytea.dynamics.protocol import (
    BeliefState,
    JointBeliefState,
    OperatorContext,
)

__all__ = [
    "LindbladOperator",
    "BeliefState",
    "JointBeliefState",
    "OperatorContext",
]
