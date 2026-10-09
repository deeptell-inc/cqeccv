#!/usr/bin/env python3
r"""
panel_audit.py — 査読パネルが同定した欠陥に対するデータ補完

このモジュールは、独立した3名の査読者（異なるモデルファミリを含む）が
再現実験によって確定させた欠陥について、欠落していた成果物を生成する。
各関数は「原稿のどの主張を検証/反証するか」を docstring に明示する。

生成される成果物 (simulation_results/panel_audit_results.json):
  A. regev_ablation        : Regev の成功が量子コアに依存するかの対照実験
  B. aligned_comparison    : 両側に同一チャネルを与えた整合版 GKP 改善率
  C. correction_decomposition : 改善率を σ 補正 / η 補正に分解
  D. onset_table           : Table robust (onset σ) の生成 —— 従来は生成コード不在
  E. cutoff_convergence    : Fock cutoff 収束確認 —— 従来は成果物不在
  F. scale_sensitivity     : 正規化尺度を独立に振った順位安定性

依存: numpy, scipy, sympy (base extra のみ)
"""

import json
import numpy as np
from math import sqrt
from pathlib import Path

from .noise_comparison import (
    qpe_error, qdrift_error, qkan_error, regev_error,
    gkp_correct, SIGMA_D, LOSS_FIXED, ci95,
)
from .cv_qec_port import db_to_delta

ALGS = {"QPE": qpe_error, "qDRIFT": qdrift_error,
        "QKAN": qkan_error, "Regev": regev_error}


# ═══════════════════════════════════════════════════════════════════
# A. Regev 対照実験
# ═══════════════════════════════════════════════════════════════════

def regev_ablation(n_seeds=5, biprimes=(143, 187, 221, 247, 253, 391)):
    """原稿の主張「CV量子コアが因数分解を駆動し 100% 成功」を検証する。

    4条件を比較する:
      quantum          : 量子コアの出力をそのまま使う（原稿の設定）
      zeros            : 量子出力 w を全て 0 に置換
      uniform_random   : 量子出力 w を一様乱数に置換
      quantum_nofallback : 量子コアそのまま + 古典乱択フォールバックを無効化

    「量子コアが寄与している」なら quantum > zeros/uniform であるはず。
    """
    from . import regev_cv as R

    def run(mode, allow_fallback=True):
        ok = 0
        total = 0
        for N in biprimes:
            for sd in range(n_seeds):
                alg = R.RegevCVFactoring(N, sq_db=12.0, verbose=False)
                orig_run = R.RegevCVProcedure.run
                orig_post = alg._postprocess

                if mode != "quantum":
                    def patched(self, Rv, rng, _m=mode):
                        d = self.d
                        if _m == "zeros":
                            return np.zeros(d)
                        return rng.random(d)
                    R.RegevCVProcedure.run = patched

                if not allow_fallback:
                    # 乱択整数結合探索を止め、LLL 候補のみで判定する
                    def post_nofb(samples, b_list, d, Rv, _a=alg):
                        import numpy as _np
                        N_ = _a.N
                        m = len(samples)
                        delta = sqrt(d) / (sqrt(2) * Rv) if Rv > 0 else 1.0
                        S = 1.0 / delta if delta > 1e-15 else Rv
                        dim = d + m
                        B = _np.zeros((dim, dim))
                        for i in range(d):
                            B[i, i] = 1.0
                        for j in range(m):
                            B[d + j, :d] = S * samples[j]
                            B[d + j, d + j] = S
                        try:
                            Br = R.lll_reduce(B, delta=0.99)
                        except Exception:
                            return None
                        cands = []
                        for i in range(dim):
                            v = Br[i]
                            head = _np.round(v[:d]).astype(int)
                            if (_np.linalg.norm(head) > 0.5
                                    and _np.linalg.norm(v[d:]) < S * 0.5):
                                cands.append(head)
                        if not cands:
                            norms = [_np.linalg.norm(Br[i]) for i in range(dim)]
                            for idx in _np.argsort(norms):
                                head = _np.round(Br[idx, :d]).astype(int)
                                if _np.any(head != 0):
                                    cands.append(head)
                                if len(cands) >= d:
                                    break
                        for z in cands:
                            f = _a._try_factor(z, b_list)
                            if f is not None:
                                return f
                        return None            # 乱択フォールバックを行わない
                    alg._postprocess = post_nofb

                try:
                    f = alg.factor(max_attempts=8, seed=sd)
                    if f is not None and 1 < f < N and N % f == 0:
                        ok += 1
                finally:
                    R.RegevCVProcedure.run = orig_run
                    alg._postprocess = orig_post
                total += 1
        return ok, total

    out = {}
    for mode, fb in [("quantum", True), ("zeros", True),
                     ("uniform_random", True), ("quantum", False)]:
        key = mode if fb else "quantum_nofallback"
        ok, total = run(mode, allow_fallback=fb)
        out[key] = {"successes": ok, "trials": total,
                    "success_rate": ok / total}
    out["interpretation"] = (
        "If success_rate(quantum) is not greater than success_rate(zeros) "
        "and success_rate(uniform_random), the reported factoring success "
        "is not attributable to the CV quantum core.")
    return out


# ═══════════════════════════════════════════════════════════════════
# B. 整合版比較 / C. 補正の分解
# ═══════════════════════════════════════════════════════════════════

def aligned_comparison(db_list=(6, 7, 7.6955, 8, 9, 12, 15, 18),
                       n_seeds=10, sigma_d=None, loss=None):
    """原稿の「必要スクイズ量はアルゴリズム依存」という主張を検証する。

    原稿版: 未訂正側 (sigma_d, eta),  訂正側 (sigma_corr, eta_corr≈1)
            → 雑音の大きさと種類が同時に変わる（交絡）
    整合版: 未訂正側 (sigma_phys, 1), 訂正側 (sigma_corr, 1)
            → 同一チャネル上の単調比較。Prop.1 は sigma_phys=Delta で
              交差する（=一律の閾値）ことを予言する。
    """
    sd = SIGMA_D if sigma_d is None else sigma_d
    ls = LOSS_FIXED if loss is None else loss
    eta = 1.0 - ls
    sigma_phys = sqrt(sd ** 2 + 0.5 * ls)
    delta_boundary_db = -10.0 * np.log10(2.0 * sigma_phys ** 2)

    res = {"sigma_d": sd, "loss": ls, "sigma_phys": sigma_phys,
           "analytic_boundary_dB": float(delta_boundary_db),
           "by_alg": {}}
    for name, fn in ALGS.items():
        u_paper = [fn(sd, eta, s) for s in range(n_seeds)]
        u_align = [fn(sigma_phys, 1.0, s) for s in range(n_seeds)]
        rows = []
        for db in db_list:
            sdc, etac, sc, p_L = gkp_correct(sd, eta, db)
            c_paper = [fn(sdc, etac, s) for s in range(n_seeds)]
            c_align = [fn(sc, 1.0, s) for s in range(n_seeds)]
            rows.append({
                "dB": float(db), "delta": float(db_to_delta(db)),
                "p_L": float(p_L),
                "paper_ratio": float(np.mean(u_paper) / np.mean(c_paper)),
                "aligned_ratio": float(np.mean(u_align) / np.mean(c_align)),
                "aligned_ratio_ci95": float(
                    ci95([a / b for a, b in zip(u_align, c_align)])),
            })
        res["by_alg"][name] = rows
    return res


def correction_decomposition(db=18.0, n_seeds=10):
    """改善率が「スクイズ床への圧縮」と「損失除去」のどちらに由来するかを分離する。

    原稿は Abstract で「up to ~2.7x at 18 dB」を GKP 訂正の効果として提示するが、
    gkp_correct は sigma を下げると同時に eta を ~1 に戻している。
    sigma のみ / eta のみ に分けると寄与が分かる。

    加えて論理誤り判定窓の感度を報告する: 既定 √π/2 (理想格子の Voronoi
    境界) と Glancy–Knill の耐故障窓 √π/6。改善率は全て p_L 経由で
    eta_corr = 1 - p_L(1-eta) に依存するので、窓の選択は結論を左右する。
    """
    sd, ls = SIGMA_D, LOSS_FIXED
    eta = 1.0 - ls
    out = {"dB": db, "sigma_d": sd, "loss": ls, "by_alg": {},
           "window_sensitivity": {}}
    u = {name: np.mean([fn(sd, eta, s) for s in range(n_seeds)])
         for name, fn in ALGS.items()}
    sdc, etac, _, pL = gkp_correct(sd, eta, db)
    for name, fn in ALGS.items():
        full = np.mean([fn(sdc, etac, s) for s in range(n_seeds)])
        eta_only = np.mean([fn(sd, etac, s) for s in range(n_seeds)])
        sig_only = np.mean([fn(sdc, eta, s) for s in range(n_seeds)])
        out["by_alg"][name] = {
            "full_correction": float(u[name] / full),
            "eta_only": float(u[name] / eta_only),
            "sigma_only": float(u[name] / sig_only),
        }
    for label, w in [("sqrt_pi_over_2", np.sqrt(np.pi) / 2),
                     ("sqrt_pi_over_6", np.sqrt(np.pi) / 6)]:
        sdw, etaw, _, pLw = gkp_correct(sd, eta, db, window=w)
        row = {"window": float(w), "p_L": float(pLw), "eta_corr": float(etaw),
               "full_correction": {}}
        for name, fn in ALGS.items():
            full = np.mean([fn(sdw, etaw, s) for s in range(n_seeds)])
            row["full_correction"][name] = float(u[name] / full)
        out["window_sensitivity"][label] = row
    return out


# ═══════════════════════════════════════════════════════════════════
# D. onset テーブル（従来は生成コードが存在しなかった）
# ═══════════════════════════════════════════════════════════════════

def onset_table(sigma_grid=None, multipliers=(3.0, 5.0, 10.0), n_seeds=10,
                loss=0.0):
    """Table robust (onset sigma) を生成する。

    onset(k) := 誤差が clean baseline の k 倍を最初に超える sigma_d。
    原稿の Table robust には生成コード・JSON が存在しなかったため、
    ここで定義を明示して作り直す。
    """
    grid = (np.arange(0.0, 0.41, 0.02) if sigma_grid is None
            else np.asarray(sigma_grid))
    eta = 1.0 - loss
    out = {"sigma_grid": grid.tolist(), "loss": loss,
           "multipliers": list(multipliers), "by_alg": {}}
    for name, fn in ALGS.items():
        curve, cis = [], []
        for s in grid:
            vals = [fn(float(s), eta, sd) for sd in range(n_seeds)]
            curve.append(float(np.mean(vals)))
            cis.append(ci95(vals))
        base = curve[0]
        onsets = {}
        for k in multipliers:
            thr = k * base if base > 0 else float("inf")
            hit = next((float(grid[i]) for i, v in enumerate(curve)
                        if v > thr), None)
            onsets[f"x{k:g}"] = hit
        out["by_alg"][name] = {"clean_baseline": base,
                               "curve_mean": curve, "curve_ci95": cis,
                               "onset": onsets}
    return out


# ═══════════════════════════════════════════════════════════════════
# E. Fock cutoff 収束（従来は成果物が存在しなかった）
# ═══════════════════════════════════════════════════════════════════

def cutoff_convergence(cutoffs=(10, 14, 18, 22), n_seeds=5):
    """Limitations の「cutoff {10,14,18} で収束確認」に対応する成果物を作る。

    qDRIFT のチャネル誤差を cutoff を変えて測り、相対変動を報告する。
    """
    from .qdrift_cv import Bosonic, QDriftCV, trace_distance
    from scipy.linalg import expm
    out = {"cutoffs": list(cutoffs), "qdrift": {}}
    for c in cutoffs:
        bo = Bosonic(cutoff=c)
        terms = bo.random_hamiltonian(np.random.default_rng(0),
                                      n_gauss=4, n_nongauss=2, scale=0.4)
        qd = QDriftCV(terms)
        psi = np.zeros(bo.N, dtype=complex)
        psi[0], psi[1], psi[2] = 0.7, 0.6, 0.39
        psi /= np.linalg.norm(psi)
        rho0 = np.outer(psi, psi.conj())
        errs = []
        for sd in range(n_seeds):
            r = qd.channel(rho0, 1.0, N=48, R=20,
                           rng=np.random.default_rng(100 + sd))
            errs.append(r["err_avg_channel"])
        out["qdrift"][str(c)] = {"mean": float(np.mean(errs)),
                                 "ci95": ci95(errs)}
    vals = [v["mean"] for v in out["qdrift"].values()]
    out["max_relative_spread"] = float((max(vals) - min(vals)) / np.mean(vals))
    return out


# ═══════════════════════════════════════════════════════════════════
# F. 正規化尺度の独立変動に対する順位安定性
# ═══════════════════════════════════════════════════════════════════

def scale_sensitivity(n_grid=9, factor_range=(0.5, 2.0)):
    """原稿の「尺度 s を [0.5,2]x で振っても順位不変」を検証する。

    原稿の記述は独立変動と読めるため、各アルゴリズムの s を独立に振り、
    順位が反転する組合せの割合を数える。
    """
    from .normalized_metric import SCALE, soft
    path = Path("simulation_results/noise_comparison_results.json")
    if not path.exists():
        return {"error": "run `cvqbench noise` first"}
    d = json.loads(path.read_text())["squeeze_sweep"]
    names = list(d.keys())
    errs = {n: d[n]["uncorrected"]["mean"] for n in names}
    base = {}
    for n in names:
        key = next((k for k in SCALE if k in n or n in k), None)
        base[n] = SCALE.get(key, 1.0)
    facs = np.linspace(factor_range[0], factor_range[1], n_grid)
    nominal = sorted(names, key=lambda n: soft(errs[n], base[n]))
    total = flips = 0
    for f0 in facs:
        for f1 in facs:
            for f2 in facs:
                for f3 in facs:
                    fs = dict(zip(names, [f0, f1, f2, f3]))
                    order = sorted(
                        names, key=lambda n: soft(errs[n], base[n] * fs[n]))
                    total += 1
                    if order != nominal:
                        flips += 1
    return {"nominal_order": nominal, "combinations": total,
            "order_changes": flips, "flip_fraction": flips / total,
            "note": ("Common-factor scaling preserves order trivially; "
                     "independent variation is the informative test.")}


# ═══════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════

def main():
    OUT = Path("simulation_results")
    OUT.mkdir(exist_ok=True)
    R = {}
    print("=" * 74)
    print(" panel_audit — 査読パネル指摘に対するデータ補完")
    print("=" * 74)

    print("\n[A] Regev 対照実験（量子コアの寄与）")
    R["regev_ablation"] = regev_ablation()
    for k, v in R["regev_ablation"].items():
        if isinstance(v, dict):
            print(f"  {k:<20}: {v['successes']:>2d}/{v['trials']:<3d} "
                  f"= {v['success_rate']:.0%}")

    print("\n[B] 整合版比較（両側に同一チャネル）")
    R["aligned_comparison"] = aligned_comparison()
    ac = R["aligned_comparison"]
    print(f"  解析境界 = {ac['analytic_boundary_dB']:.4f} dB "
          f"(sigma_phys={ac['sigma_phys']:.4f})")
    print(f"  {'dB':>8} " + " ".join(f"{n:>18}" for n in ac["by_alg"]))
    for i, db in enumerate([r["dB"] for r in next(iter(ac["by_alg"].values()))]):
        cells = []
        for n in ac["by_alg"]:
            r = ac["by_alg"][n][i]
            cells.append(f"{r['paper_ratio']:>8.3f}/{r['aligned_ratio']:<9.3f}")
        print(f"  {db:>8.2f} " + " ".join(cells))
    print("  （各セル: 原稿版 / 整合版）")

    print("\n[C] 改善率の分解（18 dB）")
    R["correction_decomposition"] = correction_decomposition()
    print(f"  {'alg':<10}{'full':>9}{'eta_only':>10}{'sigma_only':>12}")
    for n, v in R["correction_decomposition"]["by_alg"].items():
        print(f"  {n:<10}{v['full_correction']:>9.3f}"
              f"{v['eta_only']:>10.3f}{v['sigma_only']:>12.3f}")
    print("  判定窓感度 (full_correction):")
    for lbl, w in R["correction_decomposition"]["window_sensitivity"].items():
        cells = " ".join(f"{n}={v:.2f}" for n, v in w["full_correction"].items())
        print(f"    {lbl:<16} p_L={w['p_L']:.4f}  {cells}")

    print("\n[D] onset テーブル（生成コードを新設）")
    R["onset_table"] = onset_table()
    print(f"  {'alg':<10}{'clean':>12}{'x3':>8}{'x5':>8}{'x10':>8}")
    for n, v in R["onset_table"]["by_alg"].items():
        o = v["onset"]
        f = lambda x: f"{x:.2f}" if x is not None else "  -"
        print(f"  {n:<10}{v['clean_baseline']:>12.3e}"
              f"{f(o['x3']):>8}{f(o['x5']):>8}{f(o['x10']):>8}")

    print("\n[E] Fock cutoff 収束")
    R["cutoff_convergence"] = cutoff_convergence()
    for c, v in R["cutoff_convergence"]["qdrift"].items():
        print(f"  cutoff={c:<4}: {v['mean']:.4e} ± {v['ci95']:.1e}")
    print(f"  最大相対変動 = "
          f"{R['cutoff_convergence']['max_relative_spread']:.2%}")

    print("\n[F] 正規化尺度の独立変動")
    R["scale_sensitivity"] = scale_sensitivity()
    ss = R["scale_sensitivity"]
    if "error" not in ss:
        print(f"  順位反転 {ss['order_changes']}/{ss['combinations']} "
              f"= {ss['flip_fraction']:.1%}")

    p = OUT / "panel_audit_results.json"
    p.write_text(json.dumps(R, indent=2, default=str))
    print(f"\n結果保存: {p}")
    return R


if __name__ == "__main__":
    main()
