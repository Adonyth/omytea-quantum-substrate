"""Tests for `omytea.density` — DensityMatrix primary representation."""

from __future__ import annotations

import pytest

from omytea.density import DensityMatrix


# --------------------------------------------------------------
# DensityMatrix construction + invariants
# --------------------------------------------------------------


def test_density_matrix_2x2_identity_construction() -> None:
    """A 2×2 density matrix [[0.5, 0], [0, 0.5]] — maximally mixed state."""
    data = (
        (complex(0.5, 0), complex(0, 0)),
        (complex(0, 0), complex(0.5, 0)),
    )
    rho = DensityMatrix(data=data)
    assert rho.data[0][0] == complex(0.5, 0)
    assert rho.data[1][1] == complex(0.5, 0)


def test_density_matrix_trace_equals_one() -> None:
    """For a valid ρ, the diagonal sums to 1.0."""
    data = (
        (complex(0.7, 0), complex(0, 0)),
        (complex(0, 0), complex(0.3, 0)),
    )
    rho = DensityMatrix(data=data)
    trace = sum(rho.data[i][i] for i in range(len(rho.data)))
    assert abs(trace - complex(1.0, 0.0)) < 1e-9


def test_density_matrix_hermitian_off_diagonals() -> None:
    """Off-diagonal elements satisfy ρ_{ij} = ρ_{ji}^*."""
    data = (
        (complex(0.5, 0),    complex(0.1, 0.05)),
        (complex(0.1, -0.05), complex(0.5, 0)),
    )
    rho = DensityMatrix(data=data)
    upper = rho.data[0][1]
    lower = rho.data[1][0]
    assert lower == upper.conjugate()


def test_density_matrix_3x3_construction() -> None:
    """Larger dimensions work the same way."""
    n = 3
    data = tuple(
        tuple(complex(1.0 / n if i == j else 0.0, 0.0) for j in range(n))
        for i in range(n)
    )
    rho = DensityMatrix(data=data)
    assert len(rho.data) == 3
    assert len(rho.data[0]) == 3
    trace = sum(rho.data[i][i] for i in range(n))
    assert abs(trace - complex(1.0, 0.0)) < 1e-9


def test_density_matrix_immutable() -> None:
    """``data`` is a tuple of tuples — mutation should not be possible."""
    data = (
        (complex(0.5, 0), complex(0, 0)),
        (complex(0, 0), complex(0.5, 0)),
    )
    rho = DensityMatrix(data=data)
    # tuple is immutable
    with pytest.raises((TypeError, AttributeError)):
        rho.data[0] = (complex(0.9, 0), complex(0.1, 0))  # type: ignore[index]
