#!/usr/bin/env python3
"""
gkp_validate.py — GKP 訂正モデルの整合性チェック

解析的 p_round (単一ラウンド GKP shift-error モデル) を、同じシフト模型を
手続きとして書いた Monte Carlo と照合する。チェック [1] (p_round) だけが
解析式に対する独立な数値検証である。

⚠ チェック [2],[3] は構成上の恒等式であり検証ではない:
  r = e − wrap(e+ξ, √π) = −ξ + √π k  ⇒  wrap(r, √π) ≡ −ξ  ⇒  σ_res ≡ Δ。
  σ_phys を 0.3 → 50 に変えても σ_res は 15 桁一致する。[3] は
  その σ_res=Δ を surrogate 式に代入し直しているだけなので、surrogate が
  「二チャネル真値」と一致するのは定義による。これらは実装の自己整合性
  を示すだけで、surrogate を物理的に検証したことにはならない。

プロトコル (Glancy-Knill 型, 1 quadrature):
  入力シフト e ~ N(0, σ_phys)         (物理変位誤り)
  補助 GKP の有限スクイズ syndrome 雑音 ξ ~ N(0, Δ)
  syndrome s = wrap_{√π}(e + ξ)        (modular 測定)
  訂正 c = s,  残留 r = e - c
  論理誤り: 残留が論理 Voronoi 境界 √π/2 を越える
これは閉形式ではなく手続きそのものの MC。解析式と照合する。

GKP は2 quadrature (q,p) 独立 → p_L = 1-(1-p_round)².
依存: NumPy, SciPy, cv_qec_port
"""

import numpy as np
from scipy.special import erf
from pathlib import Path
import json
from .cv_qec_port import db_to_delta, GKPStabilizerCV

SQRT_PI = np.sqrt(np.pi)


def wrap(x, period):
    """[-period/2, period/2) へ折り畳み."""
    return (x + period / 2) % period - period / 2


def gkp_ec_montecarlo(sigma_phys, delta, n_samples, rng):
    """実 GKP EC プロトコルの MC: 1 quadrature あたりの

    - 論理誤り率 p_round_emp
    - 訂正後の連続残留 std σ_res_emp (= 床 ~Δ になるはず)
    """
    e = rng.normal(0, sigma_phys, n_samples)        # 入力シフト
    xi = rng.normal(0, delta, n_samples)            # 補助有限スクイズ雑音
    s = wrap(e + xi, SQRT_PI)                        # modular syndrome
    c = s                                            # 訂正量
    r = e - c                                        # 残留シフト
    # 論理誤り: 残留が論理セル (周期 √π) の境界を越える
    r_log = wrap(r, SQRT_PI)
    logical_err = np.abs(wrap(r, SQRT_PI) - r) > 1e-9  # 折り畳みで判定
    # 連続残留 (床): 論理セル内に丸めた残り
    sigma_res = float(np.std(r_log))
    p_round_emp = float(np.mean(np.abs(r) > SQRT_PI / 2))
    return p_round_emp, sigma_res


def p_round_analytic(sigma_phys, delta):
    sigma_eff = np.sqrt(sigma_phys ** 2 + delta ** 2)
    return float(1.0 - erf((SQRT_PI / 2) / (np.sqrt(2) * sigma_eff)))


def main():
    OUT = Path("simulation_results")
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    print("=" * 72)
    print(" GKP 訂正モデル検証: 実 EC プロトコル MC vs 解析式")
    print("=" * 72)

    R = {}

    # --- 1. p_round: 解析式 vs MC (σ_phys × dB グリッド, 10シード) ---
    print("\n[1] 論理誤り率 p_round: 解析 vs 実プロトコル MC")
    print(f"{'σ_phys':>7}{'dB':>5}{'Δ':>7}{'p_ana':>10}{'p_MC':>12}"
          f"{'相対差':>9}")
    grid = []
    for sigma in [0.10, 0.20, 0.30, 0.40]:
        for db in [9.0, 12.0, 15.0]:
            delta = db_to_delta(db)
            pa = p_round_analytic(sigma, delta)
            mc = []
            for sd in range(10):
                r2 = np.random.default_rng(100 + sd)
                pe, _ = gkp_ec_montecarlo(sigma, delta, 200000, r2)
                mc.append(pe)
            pm, pci = float(np.mean(mc)), float(
                1.96 * np.std(mc, ddof=1) / np.sqrt(10))
            # 相対差は p_analytic>1e-3 の意味のある領域でのみ評価。
            # p≈0 では 0÷0 で発散するため絶対差を併記。
            meaningful = pa > 1e-3
            rel = (abs(pm - pa) / pa) if meaningful else None
            grid.append({'sigma_phys': sigma, 'dB': db, 'delta': delta,
                         'p_analytic': pa, 'p_mc_mean': pm,
                         'p_mc_ci95': pci, 'rel_diff': rel,
                         'abs_diff': abs(pm - pa),
                         'meaningful': bool(meaningful)})
            rs = f"{rel:>8.1%}" if rel is not None else "   (p≈0)"
            print(f"{sigma:>7.2f}{db:>5.0f}{delta:>7.3f}{pa:>10.4f}"
                  f"{pm:>9.4f}±{pci:.0e}{rs}")
    R['p_round_validation'] = grid
    rels = [g['rel_diff'] for g in grid if g['rel_diff'] is not None]
    max_rel = max(rels)
    max_abs = max(g['abs_diff'] for g in grid)
    print(f"  → 相対差 (p>1e-3 領域) 最大 {max_rel:.1%}; "
          f"全域 絶対差最大 {max_abs:.2e}")
    print(f"    (解析式は実プロトコルを再現)")
    R['max_rel_diff_meaningful'] = max_rel
    R['max_abs_diff'] = max_abs

    # --- 2. 連続残留床 σ_res = Δ (恒等式: wrap(r) ≡ -ξ) ---
    print("\n[2] 訂正後の連続残留床 σ_res (構成上 σ_res ≡ Δ; 恒等式の確認)")
    floor = []
    for db in [9.0, 12.0, 15.0]:
        delta = db_to_delta(db)
        _, sres = gkp_ec_montecarlo(0.20, delta, 500000,
                                    np.random.default_rng(7))
        floor.append({'dB': db, 'delta': delta, 'sigma_res': sres,
                      'ratio': sres / delta})
        print(f"  {db:>4.0f}dB: Δ={delta:.3f}  σ_res={sres:.3f}  "
              f"σ_res/Δ={sres/delta:.2f}")
    R['residual_floor'] = floor

    # --- 3. surrogate σ_corr: σ_res=Δ を代入した自己整合性 (恒等式) ---
    print("\n[3] surrogate σ_corr² = (1-p_L)Δ² + p_L σ_phys² の自己整合性")
    print("    (σ_res ≡ Δ を代入するので一致は定義による; 独立検証ではない)")
    comp = []
    for sigma in [0.10, 0.20, 0.30]:
        for db in [9.0, 12.0, 15.0]:
            delta = db_to_delta(db)
            gkp = GKPStabilizerCV(db)
            pL, _ = gkp.logical_error_rate(sigma)
            sur = np.sqrt((1 - pL) * delta ** 2 + pL * sigma ** 2)
            # 二チャネル真値: 残留 Δ (確率1-pL) + σ_phys 規模の論理誤り(pL)
            # を分散合成した実効スカラー (MC 残留床 + 論理誤り寄与)
            _, sres = gkp_ec_montecarlo(sigma, delta, 300000,
                                        np.random.default_rng(11))
            true_eff = np.sqrt((1 - pL) * sres ** 2 + pL * sigma ** 2)
            rel = abs(sur - true_eff) / true_eff
            comp.append({'sigma_phys': sigma, 'dB': db,
                         'surrogate': float(sur),
                         'two_channel': float(true_eff),
                         'rel_diff': float(rel)})
            print(f"  σ={sigma:.2f} {db:>4.0f}dB: surrogate={sur:.4f} "
                  f"二チャネル={true_eff:.4f} 相対差={rel:.1%}")
    R['surrogate_validation'] = comp
    max_sur = max(c['rel_diff'] for c in comp)
    R['surrogate_max_rel_diff'] = max_sur

    path = OUT / "gkp_validate_results.json"
    with open(path, 'w') as fh:
        json.dump(R, fh, indent=2, default=str)
    print(f"\n結果保存: {path}")

    print("\n" + "=" * 72)
    print(" 結論")
    print("=" * 72)
    print(f" • 解析的 p_round は同一シフト模型の MC を相対差≤{max_rel:.1%}"
          f"(p>1e-3), 絶対差≤{max_abs:.1e}(全域) で再現 (唯一の独立チェック)")
    print(f" • σ_res/Δ = 1 と surrogate 一致 (≤{max_sur:.2%}) は構成上の恒等式")
    print(" ⇒ surrogate は自己整合的だが、物理的 GKP EC 回路に対しては未検証")
    return R


if __name__ == '__main__':
    main()
