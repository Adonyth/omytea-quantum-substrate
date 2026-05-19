# Changelog

All notable changes to `omytea-quantum-substrate`.

The format is based on [Keep a Changelog](https://keepachangelog.com/) and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.2] — 2026-05-19

### Added (CI / archival only — no API changes vs 0.1.1)

- **First Zenodo-archived release.** The repo's GitHub-Zenodo integration was enabled after v0.1.1 had already shipped, so v0.1.0 and v0.1.1 did not auto-archive. v0.1.2 is the first release Zenodo's webhook picks up; the minted DOI becomes the canonical citation handle from this release forward.
- README and CITATION.cff will be updated to surface the DOI badge + bibtex once the mint completes (separate commit, typically within a few minutes of this tag).

### Notes

This is a version-bump-only release. All public API, dependencies, tests, and behaviour are byte-equivalent to 0.1.1. Functionally identical; archivally distinct.

## [0.1.1] — 2026-05-19

### Added (CI only — no API changes vs 0.1.0)

- `.github/workflows/ci.yml` `publish` job with PyPI **Trusted Publishing** (OIDC). Future tag pushes auto-publish to PyPI without any API tokens.

### Notes

This is the **first version published to PyPI**. v0.1.0 was tagged on GitHub before the Trusted-Publishing workflow was committed, so it ships only via the GitHub release page (wheel + sdist attached there). v0.1.1 is functionally identical to v0.1.0 — same module surface, same tests, same dependencies. Use `pip install omytea-quantum-substrate` from now on.

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
