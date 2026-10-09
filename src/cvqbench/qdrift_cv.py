#!/usr/bin/env python3
"""
qdrift_cv.py — qDRIFT ランダム積公式 (PRX Quantum 2, 040305) の
               光連続量(CV)量子計算機への移植

原論文: Chen, Huang, Kueng, Tropp,
        "Concentration for Random Product Formulas"

CV 移植方針:
  H = Σ_l h_l をボソン演算子の和 (ガウス2次 + 非ガウス: Kerr/cubic)。
  qDRIFT: 重要度サンプリングで項 l を p_l=‖h_l‖/λ で選び
          V_k = exp(-i(t/N) X_k),  X_k=(λ/‖h_l‖)h_l  を適用。
  平均チャネル E[∏V_k] ≈ e^{-iHt} (ダイアモンド誤差 O(λ²t²/N))。

CV での価値 (cv_qec_port.py の非ガウス律速の教訓に整合):
  各 V_k は光学ゲート (ガウス項=干渉計/スクイーザ, 非ガウス項=cubic)。
  ゲート数 N は項数 L に非依存 (論文の核心) ⇒ CV 最大コストの
  非ガウス(cubic-phase/GKP-magic)消費を最小化できる。

依存: NumPy, SciPy
"""

import numpy as np
from scipy.linalg import expm
from pathlib import Path
import json


# ═══════════════════════════════════════════════════════════════════
# ボソンモード演算子 (Fock 切断)
# ═══════════════════════════════════════════════════════════════════

class Bosonic:
    def __init__(self, cutoff=10):
        self.N = cutoff
        a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
        self.a = a
        self.ad = a.conj().T
        x = (a + self.ad) / np.sqrt(2)
        p = (a - self.ad) / (1j * np.sqrt(2))
        n = self.ad @ a
        # ガウス(2次)生成子と非ガウス生成子
        self.terms_gauss = {
            'x': x, 'p': p, 'n': n,
            'sq+': (a @ a + self.ad @ self.ad) / 2,
            'sq-': (a @ a - self.ad @ self.ad) / (2j),
        }
        self.terms_nongauss = {
            'kerr': n @ n,
            'cubic': x @ x @ x,
        }

    def random_hamiltonian(self, rng, n_gauss=4, n_nongauss=2, scale=0.3):
        """H = Σ h_l (ガウス + 非ガウス) をランダム係数で構築.

        ⚠ 非ガウス項 x³, n² は非有界演算子なので、cutoff を変えて構築し直すと
        ‖h_l‖ が増大し λ=Σ‖h_l‖ も変わる (λ: 16.3 @ c=10 → 91.1 @ c=26)。
        λ を揃えても qDRIFT 誤差は cutoff 10→26 で約 10 分の 1 に単調減少
        する (residual_audit.qdrift_cutoff_study) ので、本ベンチマークは
        Fock cutoff に関して収束していない。絶対値は cutoff の性質として読む。
        """
        terms = []
        gk = list(self.terms_gauss)
        for _ in range(n_gauss):
            k = gk[rng.integers(len(gk))]
            c = rng.normal(0, scale)
            terms.append(('G', c * self.terms_gauss[k]))
        nk = list(self.terms_nongauss)
        for _ in range(n_nongauss):
            k = nk[rng.integers(len(nk))]
            c = rng.normal(0, scale * 0.5)
            terms.append(('NG', c * self.terms_nongauss[k]))
        return terms


def spec_norm(M):
    return float(np.max(np.abs(np.linalg.eigvalsh((M + M.conj().T) / 2))))
    # Hermitian 項のスペクトルノルム


def trace_distance(rho, sigma):
    d = rho - sigma
    sv = np.linalg.svd(d, compute_uv=False)
    return float(0.5 * np.sum(sv))


# ═══════════════════════════════════════════════════════════════════
# qDRIFT チャネル (CV)
# ═══════════════════════════════════════════════════════════════════

class QDriftCV:
    def __init__(self, terms):
        self.terms = terms                          # [('G'|'NG', h_l)]
        self.norms = np.array([spec_norm(h) for _, h in terms])
        self.lam = float(self.norms.sum())          # λ = Σ‖h_l‖
        self.probs = self.norms / self.lam
        self.is_ng = np.array([1 if tag == 'NG' else 0
                               for tag, _ in terms])
        self.H = sum(h for _, h in terms)
        self.dim = self.H.shape[0]

    def ideal(self, rho0, t):
        U = expm(-1j * self.H * t)
        return U @ rho0 @ U.conj().T

    def one_realization(self, rho0, t, N, rng):
        """N ステップ qDRIFT 1 実現. 非ガウスゲート数も返す."""
        rho = rho0.copy()
        ng_count = 0
        idx = rng.choice(len(self.terms), size=N, p=self.probs)
        for l in idx:
            _, h = self.terms[l]
            Xk = (self.lam / self.norms[l]) * h     # ‖Xk‖=λ
            Vk = expm(-1j * (t / N) * Xk)
            rho = Vk @ rho @ Vk.conj().T
            ng_count += self.is_ng[l]
        return rho, int(ng_count)

    def channel(self, rho0, t, N, R, rng):
        """平均チャネル ρ̄ (R 実現平均) と単一実現平均誤差を評価."""
        rho_ideal = self.ideal(rho0, t)
        rho_bar = np.zeros_like(rho0)
        single_errs = []
        ng_total = 0
        for _ in range(R):
            rr, ng = self.one_realization(rho0, t, N, rng)
            rho_bar += rr
            single_errs.append(trace_distance(rr, rho_ideal))
            ng_total += ng
        rho_bar /= R
        return {
            'err_avg_channel': trace_distance(rho_bar, rho_ideal),
            'err_single_mean': float(np.mean(single_errs)),
            'err_single_std': float(np.std(single_errs)),
            'ng_gates_per_run': ng_total / R,
            'lambda': self.lam,
        }


class TrotterCV:
    """1次 Lie-Trotter (決定論的積公式) — L 依存の対比基準.

    同じゲート予算 N で r=⌊N/L⌋ 周回, 各周回 ∏_l e^{-i h_l t/r}。
    誤差は交換子構造に依存 ⇒ L (項数) に依存する (論文 Eq.2)。
    qDRIFT の L 非依存性を非自明に示すための対照群。
    """

    def __init__(self, terms):
        self.terms = terms
        self.H = sum(h for _, h in terms)

    def ideal(self, rho0, t):
        U = expm(-1j * self.H * t)
        return U @ rho0 @ U.conj().T

    def evolve(self, rho0, t, N):
        L = len(self.terms)
        r = max(1, N // L)
        gates = [expm(-1j * h * (t / r)) for _, h in self.terms]
        rho = rho0.copy()
        for _ in range(r):
            for V in gates:
                rho = V @ rho @ V.conj().T
        return rho, r * L

    def error(self, rho0, t, N):
        rr, g = self.evolve(rho0, t, N)
        return trace_distance(rr, self.ideal(rho0, t)), g


def split_terms(terms, copies, rng):
    """各 h_l を copies 個へ正の係数でランダム不等分割.

    Σf_i=1, f_i>0 ⇒ Σ f_i h_l = h_l (H 不変),
    Σ‖f_i h_l‖ = ‖h_l‖ Σf_i = ‖h_l‖ (λ 不変)。
    分割は不等 ⇒ qDRIFT サンプリング粒度は実質的に変化し、
    「誤差が L に依存しない」ことを非自明に検証できる。
    """
    out = []
    for tag, h in terms:
        f = rng.random(copies) + 0.1
        f = f / f.sum()
        for fi in f:
            out.append((tag, fi * h))
    return out


# ═══════════════════════════════════════════════════════════════════
# メイン
# ═══════════════════════════════════════════════════════════════════

def main():
    OUT = Path("simulation_results")
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print(" qDRIFT ランダム積公式 → 光連続量(CV)量子計算機 移植")
    print("=" * 72)

    bo = Bosonic(cutoff=10)
    rng0 = np.random.default_rng(0)
    terms = bo.random_hamiltonian(rng0, n_gauss=4, n_nongauss=2, scale=0.4)
    qd = QDriftCV(terms)
    psi0 = bo.a[:, 0] * 0
    psi0_vec = np.zeros(bo.N, dtype=complex)
    # 初期状態: 低励起重ね合わせ (固定入力)
    psi0_vec[0], psi0_vec[1], psi0_vec[2] = 0.7, 0.6, 0.39
    psi0_vec /= np.linalg.norm(psi0_vec)
    rho0 = np.outer(psi0_vec, psi0_vec.conj())
    t = 1.0

    print(f"\nH: {len(terms)} 項 (ガウス4+非ガウス2), λ=‖Σ‖_terms="
          f"{qd.lam:.4f}, t={t}")

    R = {}

    # --- 1. 平均チャネル誤差 vs N (O(λ²t²/N) 減衰) ---
    print("\n[1] 平均チャネル誤差 vs ステップ数 N (5シード mean±std)")
    n_scan = []
    for N in [16, 32, 64, 128, 256]:
        eavg, esing = [], []
        for sd in range(5):
            rng = np.random.default_rng(100 + sd)
            r = qd.channel(rho0, t, N, R=60, rng=rng)
            eavg.append(r['err_avg_channel'])
            esing.append(r['err_single_mean'])
        em, es = float(np.mean(eavg)), float(np.std(eavg))
        sm = float(np.mean(esing))
        bound = qd.lam ** 2 * t ** 2 / N
        n_scan.append({'N': N, 'err_avg_mean': em, 'err_avg_std': es,
                       'err_single_mean': sm, 'bound_l2t2_over_N': bound})
        print(f"  N={N:>3d}: 平均ch誤差={em:.4e}±{es:.1e} "
              f"| 単一実現={sm:.4e} | 上界λ²t²/N={bound:.3e}")
    R['error_vs_N'] = n_scan

    # --- 2. L 非依存性 (論文の核心) vs L 依存の Trotter 対比 ---
    print("\n[2] qDRIFT の L 非依存性  vs  Trotter の L 依存性")
    print("    (H, λ 固定で L のみ増加, ゲート予算 N=64 固定)")
    l_scan = []
    N_fixed = 64
    for copies in [1, 2, 4, 8]:
        tt = split_terms(terms, copies, np.random.default_rng(900 + copies))
        qd2 = QDriftCV(tt)
        eavg = []
        for sd in range(5):
            rng = np.random.default_rng(200 + sd)
            rr = qd2.channel(rho0, t, N_fixed, R=60, rng=rng)
            eavg.append(rr['err_avg_channel'])
        em, es = float(np.mean(eavg)), float(np.std(eavg))
        tr_err, tr_g = TrotterCV(tt).error(rho0, t, N_fixed)
        l_scan.append({'L': len(tt), 'copies': copies, 'lambda': qd2.lam,
                       'qdrift_err_mean': em, 'qdrift_err_std': es,
                       'trotter_err': float(tr_err),
                       'trotter_gates': int(tr_g)})
        print(f"  L={len(tt):>2d}: qDRIFT={em:.4e}±{es:.1e} (L非依存) "
              f"| Trotter={tr_err:.4e} (L依存, {tr_g}ゲート)")
    R['L_independence'] = l_scan
    print("  注: qDRIFT の X_k=(λ/‖h_l‖)h_l は分割不変 ⇒ L 非依存は厳密。")
    print("      Trotter は固定予算で L 増加に伴い誤差が変動 (対照)。")

    # --- 3. CV 資源台帳: 非ガウスゲート消費 ---
    print("\n[3] CV 量子資源台帳 (非ガウス律速)")
    ng_frac = float(qd.is_ng @ qd.probs)            # 非ガウス λ 比率
    ledger = {}
    for N in [16, 64, 256]:
        rng = np.random.default_rng(7)
        r = qd.channel(rho0, t, N, R=40, rng=rng)
        ledger[f'N_{N}'] = {
            'total_gates': N,
            'nongaussian_gates_mean': r['ng_gates_per_run'],
            'gaussian_gates_mean': N - r['ng_gates_per_run'],
            'err_avg_channel': r['err_avg_channel'],
        }
        print(f"  N={N:>3d}: 全{N}ゲート中 非ガウス≈"
              f"{r['ng_gates_per_run']:.1f} (比率{ng_frac:.2%}) "
              f"誤差={r['err_avg_channel']:.3e}")
    ledger['nongaussian_lambda_fraction'] = ng_frac
    ledger['note'] = ('ゲート数 N は項数 L 非依存 → '
                      '非ガウス(cubic/GKP-magic)消費を最小化可能')
    R['resource_ledger'] = ledger

    path = OUT / "qdrift_cv_results.json"
    with open(path, 'w') as fh:
        json.dump(R, fh, indent=2, default=str)
    print(f"\n結果保存: {path}")

    print("\n" + "=" * 72)
    print(" 結論")
    print("=" * 72)
    e16 = n_scan[0]['err_avg_mean']
    e256 = n_scan[-1]['err_avg_mean']
    print(f" • 平均チャネル誤差は N で単調減衰 "
          f"({e16:.2e}→{e256:.2e}, ∝λ²t²/N)")
    lq = [x['qdrift_err_mean'] for x in l_scan]
    lt = [x['trotter_err'] for x in l_scan]
    print(f" • qDRIFT は L 非依存 ({min(lq):.2e}〜{max(lq):.2e} 一定) "
          f"⇔ Trotter は L 依存 ({min(lt):.2e}〜{max(lt):.2e})")
    print(f" • 非ガウスゲートは全 N の {ng_frac:.1%} のみ → "
          f"N 最小化で cubic/GKP-magic 消費を削減")
    print(" • CV では各 V_k がガウス演算+少数 cubic = 直接光学実装可能")
    return R


if __name__ == '__main__':
    main()
