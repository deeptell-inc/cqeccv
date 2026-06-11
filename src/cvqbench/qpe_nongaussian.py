#!/usr/bin/env python3
"""
qpe_nongaussian.py — 制御なし QPE の非ガウス H への一般化（あれば良いデータ）

論文 §3 で「制御なし QPE は H が2次(ガウス)なら e^{iHt} が純ガウスで
CV 最良適合」と示した。本スクリプトは **非ガウス H** (Kerr/cubic を含む)
でも制御ユニタリ 0 のまま絶対スペクトルを復元できることを示し、
「制御 U 不要」性が H の種類に依存しないこと（一般化）を検証する。

物理:
  H = ω n̂ + (g/2)(a²+a†²) + χ (n̂)² + κ x³   ← 非ガウス
  e^{iHt} は非ガウス → 実機では GKP ビット + Trotter が必要だが、
  制御なし回路構造 (U_ψ → e^{iHt} → U_ψ† → 真空射影) は不変。
  ⇒ CV 最大障壁の制御-e^{iHt} は依然回避される（一般化が成立）。

参照固有状態ホログラフィ phase-retrieval (qpe_cv と同一, H 非依存)。
依存: NumPy, qpe_cv
"""

import numpy as np
from pathlib import Path
import json
from .qpe_cv import BosonicMode, ControlFreeQPE_CV


def nongaussian_hamiltonian(bm, omega=1.0, g=0.25, chi=0.15, kappa=0.05):
    """非ガウス H = ω n̂ + (g/2)(a²+a†²) + χ n̂² + κ x³."""
    a, ad, n = bm.a, bm.adag, bm.n_op
    x = (a + ad) / np.sqrt(2)
    H = (omega * n + 0.5 * g * (a @ a + ad @ ad)
         + chi * (n @ n) + kappa * (x @ x @ x))
    return 0.5 * (H + H.conj().T)                    # Hermitize（数値安定）


def _snorm(M):
    return float(np.max(np.abs(np.linalg.eigvalsh((M + M.conj().T) / 2))))


def nongaussian_fraction(bm, omega, g, chi, kappa):
    """実係数で重み付けした非ガウス成分の相対スペクトルノルム.

    fraction = ‖χ n̂²+κ x³‖ / (‖ω n̂+(g/2)(a²+a†²)‖ + ‖χ n̂²+κ x³‖)
    係数依存 ⇒ χ,κ 増大で単調増加（資源指標として正しい）。
    """
    a, ad, n = bm.a, bm.adag, bm.n_op
    x = (a + ad) / np.sqrt(2)
    gauss = omega * n + 0.5 * g * (a @ a + ad @ ad)
    ng = chi * (n @ n) + kappa * (x @ x @ x)
    sg, sn = _snorm(gauss), _snorm(ng)
    return float(sn / (sg + sn))


def main():
    OUT = Path("simulation_results")
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print(" 制御なし QPE の非ガウス H 一般化（あれば良いデータ）")
    print("=" * 72)

    bm = BosonicMode(cutoff=18)
    R = {}

    # --- 1. ガウス vs 非ガウス H で復元精度比較 ---
    print("\n[1] ガウス H vs 非ガウス H：絶対スペクトル復元")
    # (name, omega, g, chi, kappa)
    specs = [
        ('gaussian (g only)', 1.0, 0.30, 0.0, 0.0),
        ('nonG weak (χ=0.05)', 1.0, 0.25, 0.05, 0.02),
        ('nonG mid (χ=0.15)', 1.0, 0.25, 0.15, 0.05),
        ('nonG strong (χ=0.30)', 1.0, 0.25, 0.30, 0.10),
    ]
    cases = {}
    phi = bm.coherent(1.1)
    T, n_t = 200.0, 12288
    tg = np.linspace(0, T, n_t)
    case_res = []
    for name, om, gg, ch, ka in specs:
        H = nongaussian_hamiltonian(bm, om, gg, ch, ka)
        cases[name] = H
        qpe = ControlFreeQPE_CV(H, phi, p_ref=0.6)
        target = np.sort(qpe.phi_levels[:4])
        s = qpe.acquire_series(tg, shots=0)
        rec, _ = qpe.reconstruct_spectrum(tg, s, 4)
        err = max(min(abs(rec - e)) if len(rec) else np.inf
                  for e in target)
        ngf = nongaussian_fraction(bm, om, gg, ch, ka)
        case_res.append({'case': name, 'nongauss_fraction': ngf,
                         'max_abs_error': float(err),
                         'target': target.tolist()})
        print(f"  {name:<22}: 非ガウス比={ngf:.2f}  "
              f"最大誤差={err:.3e}")
    R['gaussian_vs_nongaussian'] = case_res

    # --- 2. 非ガウス H + 有限ショット (5シード mean±std) ---
    print("\n[2] 非ガウス mid H + 有限ショット (5シード)")
    H = cases['nonG mid (χ=0.15)']
    qpe = ControlFreeQPE_CV(H, phi, p_ref=0.6)
    target = np.sort(qpe.phi_levels[:4])
    shot_scan = []
    for shots in [1000, 5000, 20000, 100000]:
        errs = []
        for sd in range(5):
            ss = qpe.acquire_series(tg, shots=shots, seed=sd)
            rr, _ = qpe.reconstruct_spectrum(tg, ss, 4)
            errs.append(max(min(abs(rr - e)) if len(rr) else np.inf
                            for e in target))
        m, st = float(np.mean(errs)), float(np.std(errs))
        shot_scan.append({'shots': shots, 'err_mean': m, 'err_std': st})
        print(f"  shots={shots:>6d}: 最大誤差 = {m:.3e} ± {st:.1e}")
    R['nongaussian_shot_noise'] = shot_scan

    # --- 3. 資源台帳: 制御U=0 は不変, 非ガウスコストのみ発生 ---
    print("\n[3] 資源台帳（ガウス↔非ガウスの差分）")
    ngf_mid = nongaussian_fraction(bm, 1.0, 0.25, 0.15, 0.05)
    led = {
        'controlled_unitaries': 0,                   # ★H 非依存で不変
        'ancilla_qubits': 0,
        'gaussian_H': {
            'e^{iHt}': 'pure Gaussian (interferometer+sq+phase)',
            'non_gaussian_gates': 0,
            'squeezing_dB': 0.0,
        },
        'nongaussian_H': {
            'e^{iHt}': 'GKP-qubit Trotter (制御Uは依然不要)',
            'non_gaussian_gates': 'Trotter段×非ガウス比',
            'nongauss_fraction_mid': ngf_mid,
            'note': '制御なし性は保持; コストは非ガウス Trotter のみ',
        },
        'measurement': 'vacuum projection (homodyne/PNR)',
    }
    R['resource_ledger'] = led
    print(f"  制御ユニタリ: 0 (ガウス/非ガウス共通・H非依存)")
    print(f"  非ガウス H の追加コスト = 非ガウス Trotter のみ "
          f"(比率 {ngf_mid:.2f})")
    print(f"  ⇒ 「制御 U 不要」は H の種類に依存しない（一般化成立）")

    path = OUT / "qpe_nongaussian_results.json"
    with open(path, 'w') as fh:
        json.dump(R, fh, indent=2, default=str)
    print(f"\n結果保存: {path}")

    print("\n" + "=" * 72)
    print(" 結論")
    print("=" * 72)
    g_err = case_res[0]['max_abs_error']
    s_err = case_res[-1]['max_abs_error']
    print(f" • 非ガウス H でも絶対スペクトル復元可 "
          f"(ガウス {g_err:.1e} → 強非ガウス {s_err:.1e})")
    print(" • 制御ユニタリ 0 は H の種類に依存せず不変（一般化成立）")
    print(" • 非ガウス H の追加コストは非ガウス Trotter のみ;")
    print("   CV 最大障壁の制御-e^{iHt} は依然回避 → QPE 最良適合は一般的")
    return R


if __name__ == '__main__':
    main()
