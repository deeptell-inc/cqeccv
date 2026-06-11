"""cvqbench — Comparative resource & noise-robustness benchmark of quantum
algorithms ported to photonic continuous-variable (CV) hardware.

Modules (each is also runnable as ``python -m cvqbench.<module>``):
  cv_qec_port          GKP / bosonic-code QEC primitives (C-CV ... F-CV)
  qpe_cv               control-free quantum phase estimation on CV
  qdrift_cv            randomized product formulas (qDRIFT) on CV
  qkan_cv, qkan_deep   quantum Kolmogorov-Arnold networks via CV-QSP
  regev_cv             Regev factoring ported to CV (GKP comb + CV-QFT)
  noise_comparison     cross-algorithm noise-robustness comparison
  normalized_metric    unit-independent ranking (post-processing)
  gkp_validate         validation of the GKP correction surrogate
  dv_baseline          discrete-variable (surface-code) baseline
  noise_phase_diagram  2D (sigma_d x loss) phase diagram + ensembles
  physics_h2_morse     diatomic Morse vibrational spectra (H2/HF/I2)
  physics_h2_qkan      QKAN classification of vibrational states

Heavy ML deps (torch, scikit-learn) are required only by ``qkan_deep`` and
``physics_h2_qkan``; install them via the ``[ml]`` extra.
"""

__version__ = "0.1.0"

__all__ = [
    "cv_qec_port", "qpe_cv", "qdrift_cv", "qkan_cv",
    "regev_cv", "noise_comparison", "normalized_metric",
    "gkp_validate", "dv_baseline", "noise_phase_diagram",
    "physics_h2_morse",
]
