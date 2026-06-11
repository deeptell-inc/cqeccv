"""Smoke + correctness tests for cvqbench (base deps only: numpy/scipy/sympy).

Tests requiring torch/scikit-learn are skipped automatically when those
optional ML extras are not installed.
"""
import importlib
import numpy as np
import pytest


# ---- imports of all base modules --------------------------------------

BASE_MODULES = [
    "cvqbench.cv_qec_port", "cvqbench.qpe_cv", "cvqbench.qdrift_cv",
    "cvqbench.qkan_cv", "cvqbench.regev_cv", "cvqbench.noise_comparison",
    "cvqbench.gkp_validate", "cvqbench.dv_baseline",
    "cvqbench.noise_phase_diagram", "cvqbench.physics_h2_morse",
    "cvqbench.qpe_nongaussian",
]


@pytest.mark.parametrize("mod", BASE_MODULES)
def test_import(mod):
    importlib.import_module(mod)


# ---- GKP finite-squeezing physics -------------------------------------

def test_gkp_floor_db_to_delta_monotone():
    from cvqbench.cv_qec_port import db_to_delta
    # higher squeezing (dB) -> smaller peak width Delta
    assert db_to_delta(15.0) < db_to_delta(9.0) < db_to_delta(3.0)


def test_gkp_logical_error_decreases_with_squeezing():
    from cvqbench.cv_qec_port import GKPStabilizerCV
    p9 = GKPStabilizerCV(9.0).logical_error_rate(0.3)[0]
    p15 = GKPStabilizerCV(15.0).logical_error_rate(0.3)[0]
    assert 0.0 <= p15 < p9 <= 1.0


# ---- Morse Hamiltonian matches analytic spectrum (CV-QPE physics) -----

def test_morse_eigenvalues_match_analytic_low_v():
    from cvqbench.physics_h2_morse import (
        build_morse_hamiltonian, morse_eigenenergies_analytic)
    H = build_morse_hamiltonian(cutoff=80)
    ev = np.sort(np.linalg.eigvalsh(H))
    ana = morse_eigenenergies_analytic(v_max=10)
    # low vibrational levels accurate to spectroscopic precision
    for v in range(6):
        err_cm = abs(ev[v] - ana[v]) * 4401.21
        assert err_cm < 1.0, f"v={v} error {err_cm:.3f} cm^-1"


# ---- Regev factoring correctness (number-theoretic core) --------------

def test_regev_factors_small_biprime():
    from cvqbench.regev_cv import RegevCVFactoring
    alg = RegevCVFactoring(143, sq_db=12.0, verbose=False)  # 11 x 13
    f = alg.factor(max_attempts=8, seed=0)
    assert f is not None and 1 < f < 143 and 143 % f == 0


# ---- GKP correction surrogate validation ------------------------------

def test_gkp_surrogate_matches_protocol():
    from cvqbench.gkp_validate import (
        p_round_analytic, gkp_ec_montecarlo)
    from cvqbench.cv_qec_port import db_to_delta
    rng = np.random.default_rng(0)
    delta = db_to_delta(12.0)
    pa = p_round_analytic(0.3, delta)
    pm, _ = gkp_ec_montecarlo(0.3, delta, 200000, rng)
    assert abs(pa - pm) < 3e-3
