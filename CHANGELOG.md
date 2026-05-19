# Changelog

All notable changes to `omytea-quantum-substrate`.

The format is based on [Keep a Changelog](https://keepachangelog.com/) and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-05-19

### Added — initial public release

This is the first standalone release of the Omytea quantum-information substrate. The same code has previously shipped as a **vendored snapshot** inside the [Personal Future Console](https://github.com/Adonyth/omytea-personal-console) public repo since v0.2.0 (2026-05-18). The standalone package extracts those 6 modules into their own importable library so they can be:

- Installed via `pip install omytea-quantum-substrate` without cloning a product repo
- Cited independently via PyPI version number + Zenodo DOI (forthcoming)
- Versioned at a different cadence from any one product built on top

### Module surface

- `omytea.models` — `Position`, `RectRegion`, `Observation`
- `omytea.density` — `DensityMatrix`
- `omytea.quantum` — `StateHypothesis`, `Velocity`, `WaveFunction`
- `omytea.joint_belief` — `JointBranchHypothesis`, `JointWaveFunction`, `OffDiagonalEntry`
- `omytea.dynamics.protocol` — `BeliefState`, `JointBeliefState`, `OperatorContext`
- `omytea.dynamics.lindblad` — `LindbladOperator`

### Dependencies

- Python >= 3.11
- **NumPy >= 1.24** (used by the stochastic resample + roughening paths in `quantum.py` and by the default `OperatorContext` rng).
- **No other runtime dependencies.** No SciPy, no Torch, no JAX, no GPU. Substrate runs in any CPython 3.11+ environment with NumPy installed.

### Tests

Test suite covers public-API surface invariants:

- `WaveFunction` construction + weight normalization
- `JointWaveFunction` Cartesian-product invariant
- `OffDiagonalEntry` Hermitian-pair construction
- `LindbladOperator` monotonic-decay invariant
- `DensityMatrix` trace preservation

Run via `pytest -q`.
