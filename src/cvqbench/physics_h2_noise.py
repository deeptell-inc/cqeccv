#!/usr/bin/env python3
r"""
physics_h2_noise.py — 分子振動分光を雑音下で行う (改訂 (b))

従来の physics_h2_morse は shots=0・損失なし・GKP なしで走っており、
本論文の主題（雑音下でどのアルゴリズムが CV に向くか）と切断されていた。
本モジュールはその欠陥を埋める。

物理:
  制御なし QPE の観測量は g(t) = |<psi| e^{iHt} |psi>|^2。
  光子損失チャネル Λ_η を挟むと
      g(t) = Tr[ |psi><psi| Λ_η( U_t |psi><psi| U_t^† ) ]
           = Σ_l | <psi| E_l U_t |psi> |^2 ,      E_l = Kraus 演算子
  U_t = V diag(e^{i E_k t}) V^† を使うと
      <psi| E_l U_t |psi> = Σ_k a_{lk} e^{i E_k t},
      a_{lk} = <psi|E_l|E_k> <E_k|psi>
  となるので、a を一度だけ作れば全時刻を行列積で得られる（厳密・高速）。
  有限ショットは g を成功確率とする二項サンプリング、変位雑音は加法。

出力:
  (i) ショット数 vs 分光精度
  (ii) (損失, 変位) 平面での 1 cm^-1 達成領域
  (iii) 同点での GKP 訂正の効果
"""

import json
import warnings
import numpy as np
from pathlib import Path

from .physics_h2_morse import (
    MOLECULES, build_morse_hamiltonian, morse_eigenenergies_analytic,
    assign_recovered_to_v,
)
from .qpe_cv import BosonicMode, ControlFreeQPE_CV
from .noise_comparison import loss_kraus, gkp_correct, ci95

# macOS Accelerate BLAS の偽 FP 例外警告のみ抑制 (結果は有限で正しい)
warnings.filterwarnings("ignore", message=".*encountered in matmul")


def noisy_series(qpe, t_grid, eta=1.0, sigma_d=0.0, shots=0, seed=0,
                 cutoff=None):
    """損失(厳密 Kraus)・変位・有限ショット下の g(t) を返す。

    損失は Kraus 演算子を通じて厳密に扱う（線形化しない）。
    """
    rng = np.random.default_rng(seed)
    psi = qpe.psi
    ev, V = qpe.evals, qpe.evecs
    c = V.conj().T @ psi                       # <E_k|psi>
    if eta >= 1.0 - 1e-12:
        amp = (np.abs(c) ** 2)[None, :]        # 単一 "Kraus" = 恒等
    else:
        Ks = loss_kraus(len(psi) if cutoff is None else cutoff, eta)
        rows = []
        for E in Ks:
            # a_lk = <psi|E|E_k><E_k|psi>
            rows.append((psi.conj() @ E @ V) * c)
        amp = np.array(rows)                    # (L, K) 複素
    phase = np.exp(1j * np.outer(t_grid, ev))   # (T, K)
    if eta >= 1.0 - 1e-12:
        g = np.abs(phase @ (np.abs(c) ** 2 + 0j)) ** 2
    else:
        g = np.sum(np.abs(phase @ amp.T) ** 2, axis=1)
    g = np.clip(np.real(g), 0.0, 1.0)
    if shots and shots > 0:
        g = rng.binomial(shots, g) / shots
    if sigma_d > 0:
        g = g + rng.normal(0.0, sigma_d, g.size)
    return g


def spectroscopy_error(molecule="H2", alpha=2.5, cutoff=100,
                       T_max=4800.0, n_t=32768, p_ref=0.85,
                       eta=1.0, sigma_d=0.0, shots=0, seed=0):
    """1 回の測定条件で復元誤差 (cm^-1) の中央値・最大値を返す。"""
    p = MOLECULES[molecule]
    l2 = p["omega_x_cm"] / p["omega_e_cm"]
    H = build_morse_hamiltonian(cutoff=cutoff, lambda2=l2)
    bm = BosonicMode(cutoff=cutoff)
    phi = bm.coherent(alpha)
    qpe = ControlFreeQPE_CV(H, phi, p_ref=p_ref)
    w = np.asarray(qpe.weights)
    n_peaks = max(2, int(np.sum(w[1:] > 5e-3)))
    tg = np.linspace(0.0, T_max, n_t)
    g = noisy_series(qpe, tg, eta=eta, sigma_d=sigma_d, shots=shots,
                     seed=seed, cutoff=cutoff)
    rec, _ = qpe.reconstruct_spectrum(tg, g, n_peaks)
    ana = morse_eigenenergies_analytic(v_max=60, lambda2=l2)
    rows = assign_recovered_to_v(rec, ana, weight_thresh=1e-4,
                                 weights=qpe.weights,
                                 omega_e_cm=p["omega_e_cm"])
    if not rows:
        return {"median_cm": float("inf"), "max_cm": float("inf"), "n": 0}
    e = [r["error_cm"] for r in rows]
    return {"median_cm": float(np.median(e)), "max_cm": float(max(e)),
            "n": len(e)}


def _stat(n_seeds=20, **kw):
    """n_seeds 本のシード統計。

    median_cm : シードごとの中央値誤差の、シード間中央値
    mean_cm   : 同、シード間平均 (外れシードに敏感)
    worst_cm  : 最悪シードの中央値誤差
    worst_level_cm : 全シード・全準位で最悪の単一準位誤差 (誤同定ピーク)
    frac_within_1cm : 中央値誤差が 1 cm^-1 以内に収まったシードの割合
    n_levels_mean   : 復元・同定できた準位数の平均 (消えた準位は誤差統計に
                      入らないので、必ず併記する)
    """
    med, nlev, mx = [], [], []
    for s in range(n_seeds):
        r = spectroscopy_error(seed=s, **kw)
        med.append(r["median_cm"])
        nlev.append(r["n"])
        mx.append(r["max_cm"])
    fin = [m for m in med if np.isfinite(m)]
    if not fin:
        return {"median_cm": float("inf"), "mean_cm": float("inf"),
                "worst_cm": float("inf"), "ci95": 0.0,
                "worst_level_cm": float("inf"),
                "frac_within_1cm": 0.0, "n_levels_mean": 0.0,
                "n_ok": 0, "n_seeds": n_seeds}
    return {"median_cm": float(np.median(fin)), "mean_cm": float(np.mean(fin)),
            "worst_cm": float(max(fin)), "ci95": ci95(fin),
            "worst_level_cm": float(max(m for m in mx if np.isfinite(m))),
            "frac_within_1cm": float(np.mean([m <= 1.0 for m in med])),
            "n_levels_mean": float(np.mean(nlev)),
            "n_ok": len(fin), "n_seeds": n_seeds}


def main():
    OUT = Path("simulation_results")
    OUT.mkdir(exist_ok=True)
    R = {}
    print("=" * 74)
    print(" 分子振動分光を雑音下で実施 (改訂 (b))")
    print("=" * 74)

    # --- (i) ショット雑音 ---
    print("\n[i] 有限ショット (H2, 損失なし・変位なし)")
    shots_rows = []
    for shots in [0, 10**2, 10**3, 10**4, 10**5]:
        r = _stat(molecule="H2", shots=shots, n_seeds=20)
        shots_rows.append({"shots": shots, **r})
        lbl = "exact" if shots == 0 else f"{shots:.0e}"
        print(f"  shots={lbl:>8}: median {r['median_cm']:.3f} "
              f"± {r['ci95']:.3f} worst {r['worst_cm']:.3f} cm^-1")
    R["shot_noise"] = shots_rows

    # --- (ii) 損失・変位平面（境界を跨ぐ範囲を掃引）---
    print("\n[ii] (損失, 変位) 平面 — 1 cm^-1 の許容領域 (H2, shots=1e4)")
    grid = []
    losses = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    sigmas = [0.0, 0.05, 0.1, 0.2]
    print(f"  {'loss / sigma_d':>16}" + "".join(f"{s:>10.2f}" for s in sigmas))
    for ls in losses:
        row = []
        for sd in sigmas:
            r = _stat(molecule="H2", eta=1.0 - ls, sigma_d=sd,
                      shots=10**4, n_seeds=20)
            row.append(r)
            grid.append({"loss": ls, "sigma_d": sd, **r})
        cells = "".join(f"{r['median_cm']:>6.2f}({r['frac_within_1cm']:.2f})"
                        if np.isfinite(r["median_cm"]) and r["median_cm"] < 1e4
                        else f"{'fail':>10}" for r in row)
        print(f"  {ls:>16.2f}{cells}")
    print("  (セル: シード間中央値 cm^-1 (1 cm^-1 以内のシード割合))")
    R["loss_displacement_grid"] = grid
    ok = [g for g in grid if g["median_cm"] <= 1.0]
    strict = [g for g in grid if g["frac_within_1cm"] >= 0.9]
    print(f"  → シード間中央値が 1 cm^-1 以内: {len(ok)}/{len(grid)} 点; "
          f"90% のシードが 1 cm^-1 以内: {len(strict)}/{len(grid)} 点 "
          f"(最大許容損失 {max((g['loss'] for g in strict), default=0):.2f})")

    # --- (iii) GKP 訂正の効果（訂正が要る領域で評価）---
    print("\n[iii] GKP 訂正の効果 (H2, 分光が破綻する領域で評価)")
    gkp_rows = []
    ls, sd = 0.40, 0.10
    base = _stat(molecule="H2", eta=1.0 - ls, sigma_d=sd, shots=10**4,
                 n_seeds=20)
    print(f"  uncorrected (loss={ls}, sd={sd}): median {base['median_cm']:.3f}"
          f" worst {base['worst_cm']:.2f} cm^-1, within 1 cm^-1: "
          f"{base['frac_within_1cm']:.2f}")
    gkp_rows.append({"dB": None, "loss": ls, "sigma_d": sd, **base})
    for db in [9.0, 12.0, 15.0, 18.0]:
        sdc, etac, _, _ = gkp_correct(sd, 1.0 - ls, db)
        r = _stat(molecule="H2", eta=etac, sigma_d=sdc, shots=10**4,
                  n_seeds=20)
        gkp_rows.append({"dB": db, **r})
        print(f"  GKP {db:>4.0f} dB: median {r['median_cm']:.3f} worst "
              f"{r['worst_cm']:.2f} cm^-1, within 1 cm^-1: "
              f"{r['frac_within_1cm']:.2f}")
    R["gkp_effect"] = gkp_rows

    # --- (iv) 3分子, 現実的条件 ---
    print("\n[iv] 3分子 (shots=1e4, loss=0.10, sigma_d=0.05)")
    mol_rows = []
    for mol, alpha, cut in [("H2", 2.5, 100), ("HF", 2.8, 110),
                            ("I2", 2.0, 100)]:
        r = _stat(molecule=mol, alpha=alpha, cutoff=cut, eta=0.90,
                  sigma_d=0.05, shots=10**4, n_seeds=20)
        mol_rows.append({"molecule": mol, **r})
        print(f"  {mol:<4}: median {r['median_cm']:.3f} ± {r['ci95']:.3f} "
              f"worst {r['worst_cm']:.3f} cm^-1 (levels {r['n_levels_mean']:.1f})")
    R["molecules_realistic"] = mol_rows

    p = OUT / "physics_h2_noise_results.json"
    p.write_text(json.dumps(R, indent=2, default=str))
    print(f"\n結果保存: {p}")
    return R


if __name__ == "__main__":
    main()
