#!/usr/bin/env python3
r"""
residual_audit.py — 監査で残った弱点の追跡（解消ではなく定量的な開示）

(1) qDRIFT の Fock cutoff 依存性
    cutoff ごとに乱択 Hamiltonian を作り直すと誤差は 25% ばらつく。
    非ガウス項 x^3, n^2 は非有界なので lambda = sum ||h_l|| が
    16.3 (c=10) -> 91.1 (c=26) と増大するが、これは説明にならない:
    lambda を cutoff 10 の値に揃え直しても誤差は cutoff とともに
    約 10 分の 1 へ単調に減少する。つまり本物の切断依存性があり、
    lambda の増大がそれを見かけ上打ち消していただけである。
    ここでは (a) 作り直し系列、(b) lambda 規格化系列、(c) cutoff 固定で
    lambda だけを振った系列 (経験的指数) の 3 本を出し、収束していない
    ことをそのまま報告する。

(2) 制御なし QPE の参照オラクル
    E_rec = E_R + nu なので E_R の誤差は係数 1 で絶対エネルギーに乗る。
    参照状態の忠実度を 0.90 まで落としても劣化は小さい。

(3) 制御なし QPE の「頑健性」の n_t 依存性
    変位雑音は n_t 個の時間サンプルで平均化されるので、noise_comparison
    の onset sigma は n_t (既定 3072) の関数である。T を固定して n_t だけ
    動かし、頑健性がサンプリング予算で決まっていることを数値で示す。
"""

import json
import warnings
import numpy as np
from pathlib import Path

# macOS Accelerate BLAS は有限な結果に対して matmul の FP 例外警告を出す
# (結果は正しい)。それ以外の警告は握りつぶさない。
warnings.filterwarnings("ignore", message=".*encountered in matmul")

from .qdrift_cv import Bosonic, QDriftCV
from .qpe_cv import BosonicMode, ControlFreeQPE_CV
from .physics_h2_morse import (
    MOLECULES, build_morse_hamiltonian, morse_eigenenergies_analytic,
)
from .noise_comparison import ci95, qpe_error


# ═══════════════════════════════════════════════════════════════════
# (1) qDRIFT: cutoff 依存性は lambda では説明できない
# ═══════════════════════════════════════════════════════════════════

def _qdrift_err(terms, c, n_seeds, N, R):
    qd = QDriftCV(terms)
    psi = np.zeros(c, dtype=complex)
    psi[0], psi[1], psi[2] = 0.7, 0.6, 0.39
    psi /= np.linalg.norm(psi)
    rho0 = np.outer(psi, psi.conj())
    errs = [qd.channel(rho0, 1.0, N=N, R=R,
                       rng=np.random.default_rng(100 + s))["err_avg_channel"]
            for s in range(n_seeds)]
    return float(qd.lam), float(np.mean(errs)), ci95(errs)


def qdrift_cutoff_study(cutoffs=(10, 14, 18, 22, 26), n_seeds=5, N=48, R=20,
                        lam_scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """作り直し系列 / lambda 規格化系列 / lambda 直接スケール系列。"""
    def build(c):
        return Bosonic(cutoff=c).random_hamiltonian(
            np.random.default_rng(0), n_gauss=4, n_nongauss=2, scale=0.4)

    rebuilt, normalized = [], []
    lam_ref = None
    for c in cutoffs:
        terms = build(c)
        lam, m, ci = _qdrift_err(terms, c, n_seeds, N, R)
        if lam_ref is None:
            lam_ref = lam
        rebuilt.append({"cutoff": c, "lambda": lam, "err": m, "ci95": ci})
        scaled = [(tag, h * (lam_ref / lam)) for tag, h in terms]
        lam2, m2, ci2 = _qdrift_err(scaled, c, n_seeds, N, R)
        normalized.append({"cutoff": c, "lambda": lam2, "err": m2, "ci95": ci2})

    c0 = cutoffs[0]
    base = build(c0)
    lam_scan = []
    for s in lam_scales:
        lam, m, ci = _qdrift_err([(t, h * s) for t, h in base], c0, n_seeds, N, R)
        lam_scan.append({"scale": s, "lambda": lam, "err": m, "ci95": ci})
    xs = np.log([r["lambda"] for r in lam_scan])
    ys = np.log([r["err"] for r in lam_scan])
    exponent = float(np.polyfit(xs, ys, 1)[0])

    def spread(rows):
        v = [r["err"] for r in rows]
        return float((max(v) - min(v)) / np.mean(v))

    lam_ratio = rebuilt[-1]["lambda"] / rebuilt[0]["lambda"]
    return {
        "rebuilt_per_cutoff": rebuilt,
        "lambda_normalized": normalized,
        "lambda_scan_fixed_cutoff": lam_scan,
        "lambda_exponent_fit": exponent,
        "spread_rebuilt": spread(rebuilt),
        "spread_lambda_normalized": spread(normalized),
        "lambda_ratio_rebuilt": float(lam_ratio),
        "err_ratio_rebuilt": float(rebuilt[-1]["err"] / rebuilt[0]["err"]),
        "err_ratio_predicted_by_lambda": float(lam_ratio ** exponent),
        "err_ratio_normalized": float(normalized[-1]["err"] / normalized[0]["err"]),
        "diagnosis": (
            "Not converged in the Fock cutoff. The lambda growth of the rebuilt "
            "model does not explain the spread: the measured lambda exponent "
            "predicts a much larger error increase than observed, and with "
            "lambda held fixed the error falls monotonically with cutoff. "
            "Absolute qDRIFT magnitudes are properties of the chosen cutoff."),
    }


# ═══════════════════════════════════════════════════════════════════
# (2) QPE 参照オラクルのコストと感度
# ═══════════════════════════════════════════════════════════════════

def _morse_qpe(cutoff=100, molecule="H2", alpha=2.5, p_ref=0.85):
    p = MOLECULES[molecule]
    l2 = p["omega_x_cm"] / p["omega_e_cm"]
    H = build_morse_hamiltonian(cutoff=cutoff, lambda2=l2)
    bm = BosonicMode(cutoff=cutoff)
    phi = bm.coherent(alpha)
    return ControlFreeQPE_CV(H, phi, p_ref=p_ref), p, l2


def reference_energy_sensitivity(deltas_cm=(0.0, 0.1, 0.5, 1.0, 5.0),
                                 T_max=4800.0, n_t=32768, seed=0):
    """E_R に誤差を与えたときの復元誤差。E_rec = E_R + nu なので係数 1。"""
    qpe, p, l2 = _morse_qpe()
    w = np.asarray(qpe.weights)
    n_peaks = max(2, int(np.sum(w[1:] > 5e-3)))
    tg = np.linspace(0.0, T_max, n_t)
    g = qpe.acquire_series(tg, shots=0, seed=seed)
    ana = morse_eigenenergies_analytic(v_max=60, lambda2=l2)
    rows = []
    E_R_true = qpe.E_R
    for d_cm in deltas_cm:
        qpe.E_R = E_R_true + d_cm / p["omega_e_cm"]     # 単位: omega_e
        rec, _ = qpe.reconstruct_spectrum(tg, g, n_peaks)
        errs = []
        for r in rec:
            v = int(np.argmin([abs(r - e) for e in ana]))
            errs.append(abs(r - ana[v]) * p["omega_e_cm"])
        rows.append({"E_R_error_cm": float(d_cm),
                     "median_recovery_error_cm": float(np.median(errs))})
    qpe.E_R = E_R_true
    return rows


def imperfect_reference(fidelities=(1.0, 0.999, 0.99, 0.95, 0.90),
                        T_max=4800.0, n_t=32768, seed=0):
    """参照状態を忠実度 F の不完全な状態に置き換えたときの劣化。"""
    qpe, p, l2 = _morse_qpe()
    ana = morse_eigenenergies_analytic(v_max=60, lambda2=l2)
    tg = np.linspace(0.0, T_max, n_t)
    rng = np.random.default_rng(seed)
    R_true, phi = qpe.R.copy(), qpe.phi.copy()
    w = np.asarray(qpe.weights)
    n_peaks = max(2, int(np.sum(w[1:] > 5e-3)))
    rows = []
    for F in fidelities:
        noise = rng.normal(size=R_true.size) + 1j * rng.normal(size=R_true.size)
        noise -= (R_true.conj() @ noise) * R_true
        noise /= np.linalg.norm(noise)
        Rp = np.sqrt(F) * R_true + np.sqrt(1 - F) * noise
        Rp /= np.linalg.norm(Rp)
        psi = np.sqrt(qpe.p_ref) * Rp + np.sqrt(1 - qpe.p_ref) * phi
        qpe.psi = psi / np.linalg.norm(psi)
        qpe.weights = np.abs(qpe.evecs.conj().T @ qpe.psi) ** 2
        g = qpe.acquire_series(tg, shots=0, seed=seed)
        rec, _ = qpe.reconstruct_spectrum(tg, g, n_peaks)
        errs = []
        for r in rec:
            v = int(np.argmin([abs(r - e) for e in ana]))
            errs.append(abs(r - ana[v]) * p["omega_e_cm"])
        rows.append({"reference_fidelity": float(F),
                     "median_recovery_error_cm": float(np.median(errs)),
                     "n_peaks": len(rec)})
    return rows


# ═══════════════════════════════════════════════════════════════════
# (3) QPE の頑健性は時間サンプル数 n_t で決まる
# ═══════════════════════════════════════════════════════════════════

def qpe_nt_sensitivity(n_ts=(192, 768, 3072, 6144),
                       sigmas=(0.0, 0.05, 0.1, 0.2), n_seeds=10, eta=1.0):
    """T=150 固定、n_t だけを変えた noise_comparison.qpe_error (max|dE|)。"""
    rows = []
    for n_t in n_ts:
        for sd in sigmas:
            v = [qpe_error(sd, eta, s, n_t=n_t) for s in range(n_seeds)]
            rows.append({"n_t": n_t, "sigma_d": sd, "err": float(np.mean(v)),
                         "ci95": ci95(v)})
    ref = {r["sigma_d"]: r["err"] for r in rows if r["n_t"] == 3072}
    for r in rows:
        r["ratio_to_nt3072"] = float(r["err"] / ref[r["sigma_d"]])
    return {"T": 150.0, "eta": eta, "rows": rows,
            "note": ("Displacement noise is averaged over n_t samples; the "
                     "robustness onset of control-free QPE is therefore a "
                     "sampling-budget statement, not a circuit property.")}


def main():
    OUT = Path("simulation_results")
    OUT.mkdir(exist_ok=True)
    R = {}
    print("=" * 74)
    print(" residual_audit — 残存弱点の定量的開示")
    print("=" * 74)

    print("\n[1] qDRIFT の Fock cutoff 依存性")
    R["qdrift_cutoff"] = qdrift_cutoff_study()
    q = R["qdrift_cutoff"]
    print(f"  {'cutoff':>7}{'lambda':>10}{'err(rebuilt)':>14}"
          f"{'err(lambda fixed)':>19}")
    for a, b in zip(q["rebuilt_per_cutoff"], q["lambda_normalized"]):
        print(f"  {a['cutoff']:>7}{a['lambda']:>10.2f}{a['err']:>14.4f}"
              f"{b['err']:>19.4f}")
    print(f"  spread: rebuilt {q['spread_rebuilt']:.1%}, "
          f"lambda-normalized {q['spread_lambda_normalized']:.1%}")
    print(f"  lambda x{q['lambda_ratio_rebuilt']:.2f} -> err x"
          f"{q['err_ratio_rebuilt']:.2f} observed; lambda^"
          f"{q['lambda_exponent_fit']:.2f} law predicts x"
          f"{q['err_ratio_predicted_by_lambda']:.1f}")
    print(f"  lambda fixed: err ratio c26/c10 = {q['err_ratio_normalized']:.3f}")

    print("\n[2a] 参照固有値 E_R の誤差伝播")
    R["reference_energy_sensitivity"] = reference_energy_sensitivity()
    for r in R["reference_energy_sensitivity"]:
        print(f"  E_R 誤差 {r['E_R_error_cm']:>5.1f} cm^-1 "
              f"→ 復元誤差 {r['median_recovery_error_cm']:>7.3f} cm^-1")

    print("\n[2b] 参照状態が不完全な場合")
    R["imperfect_reference"] = imperfect_reference()
    for r in R["imperfect_reference"]:
        print(f"  F={r['reference_fidelity']:<6.3f} → "
              f"{r['median_recovery_error_cm']:>8.3f} cm^-1 "
              f"(peaks {r['n_peaks']})")

    print("\n[3] QPE 頑健性の n_t 依存性 (T=150, eta=1)")
    R["qpe_nt_sensitivity"] = qpe_nt_sensitivity()
    rows = R["qpe_nt_sensitivity"]["rows"]
    sig = sorted({r["sigma_d"] for r in rows})
    print(f"  {'n_t':>6}" + "".join(f"{f'sd={s}':>12}" for s in sig))
    for n_t in sorted({r["n_t"] for r in rows}):
        cells = "".join(f"{r['err']:>12.5f}" for r in rows if r["n_t"] == n_t)
        print(f"  {n_t:>6}{cells}")

    p = OUT / "residual_audit_results.json"
    p.write_text(json.dumps(R, indent=2, default=str))
    print(f"\n結果保存: {p}")
    return R


if __name__ == "__main__":
    main()
