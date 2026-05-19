"""Omytea quantum-information substrate.

This package is the math core of the Omytea world-model system:

    BeliefState (API concept)
        ρ (density matrix) — open-system primary
            WaveFunction — per-entity sparse view (diagonal of ρ_i)
            JointWaveFunction — entity-tuple sparse view (diagonal of ρ_{AB})
            OffDiagonalEntry — sparse off-diagonal of ρ_{AB}

    LindbladOperator — open-system evolution of ρ over time

Stdlib + NumPy only. No SciPy / Torch / JAX. Pure CPython, no GPU. Apache 2.0.

Public API:

>>> from omytea.quantum import StateHypothesis, Velocity, WaveFunction
>>> from omytea.joint_belief import JointBranchHypothesis, JointWaveFunction, OffDiagonalEntry
>>> from omytea.density import DensityMatrix
>>> from omytea.models import Position, RectRegion, Observation
>>> from omytea.dynamics.lindblad import LindbladOperator
>>> from omytea.dynamics.protocol import OperatorContext

Companion product (the first application built on this substrate):
the Omytea Personal Future Console —
https://github.com/Adonyth/omytea-personal-console

For citation see CITATION.cff at the repo root.
"""

from omytea.density import DensityMatrix
from omytea.joint_belief import (
    JointBranchHypothesis,
    JointWaveFunction,
    OffDiagonalEntry,
)
from omytea.models import (
    Observation,
    Position,
    RectRegion,
)
from omytea.quantum import (
    StateHypothesis,
    Velocity,
    WaveFunction,
)

__version__ = "0.1.2"

__all__ = [
    "__version__",
    # quantum.py
    "StateHypothesis",
    "Velocity",
    "WaveFunction",
    # joint_belief.py
    "JointBranchHypothesis",
    "JointWaveFunction",
    "OffDiagonalEntry",
    # density.py
    "DensityMatrix",
    # models.py
    "Position",
    "RectRegion",
    "Observation",
]
