# legacy/

These are the original **flat, standalone** scripts used during development.
They are kept for provenance and for reproducing the paper figures from a fresh
checkout, but they are **superseded by the installable `cvqbench` package**
(see `../src/cvqbench/`).

## Relationship to the package

Each file here has an identical-logic counterpart in `cvqbench`:

| legacy script            | packaged module               |
|--------------------------|-------------------------------|
| `cv_qec_port.py`         | `cvqbench.cv_qec_port`        |
| `qpe_cv.py`              | `cvqbench.qpe_cv`             |
| `qdrift_cv.py`           | `cvqbench.qdrift_cv`          |
| `qkan_cv.py`             | `cvqbench.qkan_cv`            |
| `qkan_deep.py`           | `cvqbench.qkan_deep`          |
| `qpe_nongaussian.py`     | `cvqbench.qpe_nongaussian`    |
| `regev_cv.py`            | `cvqbench.regev_cv`           |
| `noise_comparison.py`    | `cvqbench.noise_comparison`   |
| `normalized_metric.py`   | `cvqbench.normalized_metric`  |
| `gkp_validate.py`        | `cvqbench.gkp_validate`       |
| `dv_baseline.py`         | `cvqbench.dv_baseline`        |
| `noise_phase_diagram.py` | `cvqbench.noise_phase_diagram`|
| `physics_h2_morse.py`    | `cvqbench.physics_h2_morse`   |
| `physics_h2_qkan.py`     | `cvqbench.physics_h2_qkan`    |

The only code difference is import style: the legacy scripts use flat
`from qpe_cv import ...` (run as `python legacy/<file>.py` from this directory),
while the package uses relative imports (run as `cvqbench <target>` or
`python -m cvqbench.<module>`).

## Recommended usage

Prefer the package:

```bash
pip install -e ".[dev]"
cvqbench all            # regenerates simulation_results/*.json
python paper/make_figs.py
```

New development should go into `src/cvqbench/`, not here.
