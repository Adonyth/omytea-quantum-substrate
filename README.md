# Omytea Quantum Substrate

[![PyPI](https://img.shields.io/pypi/v/omytea-quantum-substrate.svg)](https://pypi.org/project/omytea-quantum-substrate/)
[![Python](https://img.shields.io/pypi/pyversions/omytea-quantum-substrate.svg)](https://pypi.org/project/omytea-quantum-substrate/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/Adonyth/omytea-quantum-substrate/actions/workflows/ci.yml/badge.svg)](https://github.com/Adonyth/omytea-quantum-substrate/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20293619.svg)](https://doi.org/10.5281/zenodo.20293619)

Quantum-information substrate for the **Omytea world-model system**. Stdlib + NumPy only — no SciPy, no Torch, no JAX, no GPU. Apache 2.0.

```bash
pip install omytea-quantum-substrate
```

```python
from omytea import (
    StateHypothesis, WaveFunction, JointWaveFunction,
    OffDiagonalEntry, DensityMatrix, Position,
)
from omytea.dynamics import LindbladOperator, OperatorContext
```

## What this is

The math core of an **open-system probabilistic world model**:

- `WaveFunction` — per-entity sparse branch grid (diagonal of `ρ_i`)
- `JointWaveFunction` — entity-tuple sparse branch grid (diagonal of `ρ_{AB}`)
- `OffDiagonalEntry` — sparse off-diagonal of `ρ_{AB}` (classical correlation between joint hypotheses)
- `DensityMatrix` — open-system primary representation
- `LindbladOperator` — Gorini-Kossakowski-Sudarshan-Lindblad master-equation operator for monotonic decoherence of off-diagonal coherence

> **Honest framing**: this is *quantum-information formalism applied to classical inference*. We use `ρ` and Lindblad as a unified bookkeeping for joint distributions + their correlations + dissipation — not because the underlying system is literally quantum. The off-diagonal magnitudes carry classical-correlation information, not amplitudes of a true quantum state.

## What this is for

Building world models that:

1. Express **uncertainty over multiple entities jointly** (not just per-entity priors)
2. Model how **correlations between predicted entity futures decohere** as horizon extends
3. Stay **calibratable** via measurement updates (Brier / log-loss / reliability diagrams)
4. Run on consumer hardware — no GPU, no NumPy, no compiled extensions

The substrate is deliberately small (~2,400 lines, 6 modules) and depends only on Python's standard library + NumPy. Plug it under your perception layer and your decision UI.

## Quick example

```python
from datetime import datetime, timezone
from omytea import (
    StateHypothesis, WaveFunction, JointWaveFunction,
    JointBranchHypothesis, OffDiagonalEntry, Position,
)
from omytea.dynamics import LindbladOperator, OperatorContext

now = datetime.now(tz=timezone.utc)

# 1. Build per-entity WaveFunctions (each has 3 future-position hypotheses)
def make_wf(entity_id: str, label: str) -> WaveFunction:
    hyps = tuple(
        StateHypothesis(
            object_id=entity_id, label=name, stream_id="demo",
            timestamp=now, position=Position(x=cx, y=0.5, space="image_norm"),
            weight=w, branch_label=name,
        )
        for name, cx, w in [
            ("continue",    0.6, 0.55),
            ("accelerate",  0.9, 0.25),
            ("decelerate",  0.3, 0.20),
        ]
    )
    return WaveFunction(
        object_id=entity_id, label=label, stream_id="demo",
        timestamp=now, hypotheses=hyps, action_arm=None,
    )

wf_a = make_wf("A", "left_to_right")
wf_b = make_wf("B", "right_to_left")

# 2. Build JointWaveFunction (Cartesian product, 3×3 = 9 joint hypotheses)
joint_hyps = []
for h_a in wf_a.hypotheses:
    for h_b in wf_b.hypotheses:
        joint_hyps.append(JointBranchHypothesis(
            branch_refs={"A": h_a.hypothesis_id, "B": h_b.hypothesis_id},
            weight=h_a.weight * h_b.weight,
        ))

# 3. Add a correlation: matching-continue pairs have +0.1 off-diagonal coherence
offdiags = []
for i, hi in enumerate(joint_hyps):
    for j, hj in enumerate(joint_hyps):
        if i >= j: continue
        # ... pair (continue, continue) vs (continue, continue) — same hypothesis pairs (skip)
        # ... or actual pair logic per your model
        pass

jwf = JointWaveFunction(
    entity_ids=("A", "B"),
    hypotheses=tuple(joint_hyps),
    off_diagonal_couplings=tuple(offdiags),
)

# 4. Evolve under Lindblad at decoherence rate γ = 0.08 for 6 ticks
lindblad = LindbladOperator(decoherence_rate=0.08)
ctx = OperatorContext(scenario_name="demo", tick=0)
current = jwf
for tick in range(6):
    current = lindblad.evolve(current, dt=1.0, ctx=ctx)

# Off-diagonal magnitudes have decayed monotonically.
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for a runnable version.

## What this is *not*

- **Not a quantum-computing library.** No quantum gates, no qubits, no Pauli ops. The names borrow from quantum-information formalism; the implementation is classical.
- **Not a perception or vision library.** Substrate consumes detection bounding boxes; it doesn't produce them.
- **Not a vision-language interface.** That's downstream — see the [Personal Future Console](https://github.com/Adonyth/omytea-personal-console) for an end-to-end product built on this substrate.
- **Not a deep-learning library.** No PyTorch / JAX / TensorFlow dep. Substrate runs in CPython 3.11+ with zero compiled extensions.

## Companion product

The first end-to-end application built on this substrate:

**🚀 [Omytea Personal Future Console](https://github.com/Adonyth/omytea-personal-console)**  
Live demo: <https://omytea-personal-console.streamlit.app>

A Streamlit app that takes natural-language decision queries or short videos, runs the substrate's perception + joint-wavefunction + Lindblad evolution, and produces calibrated probability distributions over short-horizon scene futures.

## Cite this work

If you use the substrate in research, please cite it via the [`CITATION.cff`](CITATION.cff) file at the repo root, or:

```bibtex
@software{omytea_quantum_substrate_2026,
  author    = {Chen, Jiaxuan},
  title     = {Omytea Quantum Substrate: Open-System World-Model Math Core},
  year      = {2026},
  publisher = {Zenodo},
  version   = {0.1.2},
  doi       = {10.5281/zenodo.20293619},
  url       = {https://doi.org/10.5281/zenodo.20293619},
}
```

The DOI [`10.5281/zenodo.20293619`](https://doi.org/10.5281/zenodo.20293619) is the **concept DOI** — it resolves to the latest archived version (currently v0.1.2). Each individual version also gets its own DOI on the Zenodo page; pin to one if your work requires version-locked reproducibility.

The companion paper *Omytea Video World Console — Quantum-Operator Evolution over Streaming Belief States* (draft v0.1) lives in the Personal Future Console repo at [`docs/papers/OMYTEA_VIDEO_CONSOLE_DRAFT.md`](https://github.com/Adonyth/omytea-personal-console/blob/main/docs/papers/OMYTEA_VIDEO_CONSOLE_DRAFT.md).

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

## Contributing

Issues and pull requests welcome. The substrate is deliberately minimal — additions should preserve the **pure-stdlib** constraint and the **typed-dataclass** convention. See the consoles' [`CONTRIBUTING.md`](https://github.com/Adonyth/omytea-personal-console/blob/main/CONTRIBUTING.md) for the broader Omytea project's contribution guidance (same fits/doesn't-fit rules apply).

## Versioning

Semantic versioning. v0.x releases may introduce breaking changes between minor versions until the API surface stabilizes. From v1.0, breaking changes will be reserved for major versions.

See [`CHANGELOG.md`](CHANGELOG.md) for release notes.
