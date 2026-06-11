# cvqbench

**Comparative resource and noise-robustness benchmark of quantum algorithms
ported to photonic continuous-variable (CV) hardware.**

`cvqbench` ports four representative quantum algorithms to a common
continuous-variable (CV) model and compares them under a shared noise channel
(photon loss + Gaussian displacement) with Gottesman–Kitaev–Preskill (GKP)
error correction. It also includes a worked physics application: recovering the
vibrational spectra of diatomic molecules (H₂, HF, I₂) at spectroscopic
accuracy with control-free CV quantum phase estimation.

This is the software accompanying the manuscript *"Which quantum algorithms
suit photonic continuous-variable hardware? A comparative resource and
noise-robustness study"* (Wakaura & Tanimae). Every figure and table in the
paper is reproduced by the scripts here.

## Install

```bash
pip install cvqbench            # base (numpy, scipy, sympy)
pip install "cvqbench[ml]"      # + torch, scikit-learn (for QKAN modules)
pip install "cvqbench[viz]"     # + matplotlib (figure generation)
pip install "cvqbench[dev]"     # everything + pytest
```

From source:

```bash
git clone https://github.com/hwakaura/cvqbench
cd cvqbench
pip install -e ".[dev]"
```

## Usage

Each benchmark writes JSON results under `simulation_results/` in the current
directory. Run an individual target or all of them:

```bash
cvqbench qpe          # control-free QPE on CV
cvqbench noise        # cross-algorithm noise-robustness comparison
cvqbench h2-morse     # H2/HF/I2 vibrational spectra (CV-QPE)
cvqbench all          # run every benchmark
```

Equivalently as modules:

```bash
python -m cvqbench.qpe_cv
python -m cvqbench.noise_comparison
python -m cvqbench.physics_h2_morse
```

Or import the API directly:

```python
from cvqbench.cv_qec_port import GKPStabilizerCV, db_to_delta
from cvqbench.physics_h2_morse import build_morse_hamiltonian, cv_qpe_morse
```

## Benchmarks / targets

| target          | module                  | what it does |
|-----------------|-------------------------|--------------|
| `qec`           | `cv_qec_port`           | GKP / bosonic-code QEC primitives, Gaussian no-go demo |
| `qpe`           | `qpe_cv`                | control-free quantum phase estimation on CV |
| `qdrift`        | `qdrift_cv`             | randomized product formulas (qDRIFT) on CV |
| `qkan`          | `qkan_cv`               | quantum Kolmogorov–Arnold network via CV-QSP |
| `qkan-deep`     | `qkan_deep`             | multi-layer QKAN, non-separable target *(needs `[ml]`)* |
| `qpe-nongauss`  | `qpe_nongaussian`       | control-free QPE generalized to non-Gaussian H |
| `regev`         | `regev_cv`              | Regev factoring ported to CV (GKP comb + CV-QFT) |
| `noise`         | `noise_comparison`      | photon-loss + displacement cross comparison, GKP sweep |
| `normalized`    | `normalized_metric`     | unit-independent ranking (post-processing) |
| `gkp-validate`  | `gkp_validate`          | validation of the GKP correction surrogate vs MC |
| `dv-baseline`   | `dv_baseline`           | discrete-variable (surface-code) baseline |
| `phase`         | `noise_phase_diagram`   | 2D (σ_d × loss) phase diagram + ensembles |
| `h2-morse`      | `physics_h2_morse`      | diatomic Morse vibrational spectra (H₂/HF/I₂) |
| `h2-qkan`       | `physics_h2_qkan`       | QKAN classification of vibrational states *(needs `[ml]`)* |

## Repository layout

```
src/cvqbench/    installable package (canonical source)
tests/           pytest suite
paper/           manuscript (LaTeX) + make_figs.py
legacy/          original flat development scripts (superseded; see legacy/README.md)
submit_zenodo.py standalone Zenodo deposit helper
```

`legacy/` is kept for provenance only and is **not** shipped to PyPI; all
maintained code lives in `src/cvqbench/`.

## Methodology notes

- Photon loss and Gaussian displacement are simulated as **exact** Fock-space
  channels; the GKP residual-noise surrogate is **validated** against an
  explicit Glancy–Knill error-correction Monte Carlo (agreement ≤ 0.1%).
- Stochastic quantities use 10 seeds with 95% confidence intervals.
- Heavy ML dependencies (`torch`, `scikit-learn`) are required only by the two
  QKAN-learning modules and are isolated behind the `[ml]` extra.

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

## Citation

If you use this software, please cite the accompanying manuscript. See
`CITATION.cff`.

## License

MIT — see [LICENSE](LICENSE).
