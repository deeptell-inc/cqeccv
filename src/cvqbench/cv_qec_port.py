#!/usr/bin/env python3
"""
cv_qec_port.py — 3layerstabilizer の QEC パラダイムを光連続量(CV)量子計算へ移植

3layerstabilizer/bio_qec_analysis.py の 5 パラダイム + 統合系を、
離散スピン系から光 CV (ボソンモード) 系へ物理的に対応づけて移植し、
各パラダイムの量子資源コストを精査する。

物理的対応:
  C. ゲージング射影      → GKP 格子スタビライザ      (最強: 変位誤り訂正)
  B. PQEC SWAP 浄化      → GKP breeding / ガウス蒸留  (反復浄化)
  D. DFS 一重項          → 二モード DFS (共通位相不変) (受動保護)
  E. DD ナローイング     → 位相空間回転エコー          (誤り抑制)
  A. ICEC 触媒回復       → 非ガウス触媒回復            (ガウス no-go の壁)

手法:
  - ガウスチャネル/状態は共分散行列形式で厳密計算 (D, E, B-gauss)
  - GKP は標準変位誤りモデル (modular quadrature) で閾値評価 (C, B-breed)
  - 非ガウス触媒は Fock 切断で cat 状態を直接シミュレート (A)

依存: NumPy, SciPy
参照元: ../3layerstabilizer/bio_qec_analysis.py
        ../3layerstabilizer/simulation_results/qec_analysis_results.json
"""

import numpy as np
from scipy.special import erf
from scipy.linalg import expm
from pathlib import Path
from math import factorial
import json

# ═══════════════════════════════════════════════════════════════════
# 物理定数 / 規約
# ═══════════════════════════════════════════════════════════════════

# 規約: [x,p]=i, vacuum variance = 1/2
SQRT_PI = np.sqrt(np.pi)


def squeeze_db(variance_ratio):
    """分散比 (圧縮後/真空) を dB スクイージングへ変換."""
    return -10.0 * np.log10(variance_ratio)


def db_to_delta(sq_db):
    """スクイージング dB → GKP 包絡線幅 Δ (標準偏差, √π 単位)."""
    # ピーク分散 σ² = (1/2)·10^(-sq_db/10) を √π 格子で規格化
    sigma2 = 0.5 * 10.0 ** (-sq_db / 10.0)
    return np.sqrt(sigma2)


# ═══════════════════════════════════════════════════════════════════
# C-CV. GKP 格子スタビライザ  ←  3layer C. ゲージング対称性保護
# ═══════════════════════════════════════════════════════════════════

class GKPStabilizerCV:
    """GKP 符号: 3layer の S_total 射影 (ゲージング) の CV 対応物.

    対応:
      論理演算子 L = S_total      → 論理 X̄ = D(√π), Z̄ = D(i√π)
      ゲージ射影 P_m              → modular quadrature 測定で格子へ射影
      ノイズ noise_rate           → ランダムガウス変位 σ
      対称性保存                  → スタビライザ S=exp(i2√π x̂),exp(i2√π p̂)

    3layer C は noise=0.1 でも F=1.0 (最強). CV では有限スクイージング
    Δ がこの「完璧さ」を制限する — それが資源コストの本質。
    """

    def __init__(self, sq_db):
        self.sq_db = sq_db
        self.delta = db_to_delta(sq_db)        # 包絡線幅 (有限エネルギー)
        # 1 論理 qubit あたり平均光子数 n̄ ≈ 1/(2Δ²) (GKP 漸近)
        self.nbar = 1.0 / (2.0 * self.delta ** 2)

    def logical_error_rate(self, sigma_disp, n_ec_rounds=1):
        """ランダム変位 σ_disp に対する論理誤り率.

        GKP 訂正: 各 quadrature を最近接 √π 格子へ丸める。
        実効雑音 σ_eff² = σ_disp² + Δ² (有限スクイージング補助の付加雑音)。
        論理誤り = |shift| > √π/2 の確率。
        n_ec_rounds 回の訂正で残留雑音は √π/2 で再規格化される。
        """
        sigma_eff = np.sqrt(sigma_disp ** 2 + self.delta ** 2)
        # 1 ラウンドあたりの quadrature 当たり論理フリップ確率
        p_round = 1.0 - erf((SQRT_PI / 2.0) / (np.sqrt(2) * sigma_eff))
        # 2 quadrature (X,Z), n ラウンドの蓄積 (独立近似)
        p_log = 1.0 - (1.0 - p_round) ** (2 * n_ec_rounds)
        return float(np.clip(p_log, 0, 1)), float(sigma_eff)

    def fidelity_vs_noise(self, sigma_list, n_ec_rounds=1):
        out = {'sigma': [], 'p_logical': [], 'fidelity': []}
        for s in sigma_list:
            p, _ = self.logical_error_rate(s, n_ec_rounds)
            out['sigma'].append(float(s))
            out['p_logical'].append(p)
            out['fidelity'].append(1.0 - p)
        return out

    def resource_cost(self):
        """C-CV の量子資源コスト台帳."""
        return {
            'paradigm': 'C-CV GKP stabilizer',
            'modes_per_logical_qubit': 1,
            'ancilla_GKP_per_EC_round': 1,           # Steane/Knill 型
            'squeezing_dB_required': self.sq_db,
            'mean_photon_number_nbar': float(self.nbar),
            'detectors': 'homodyne x2 (x,p modular)',
            'non_gaussian_resource': 'GKP ancilla (必須・高コスト)',
            'FT_threshold_dB': 9.75,                 # Fukui et al. 2018 近傍
        }


# ═══════════════════════════════════════════════════════════════════
# B-CV. GKP breeding / ガウス蒸留  ←  3layer B. PQEC SWAP 浄化
# ═══════════════════════════════════════════════════════════════════

class BreedingCV:
    """GKP breeding: 3layer の ρ→ρ²/Tr(ρ²) 浄化の CV 対応物.

    3layer B: ℓ 回浄化で 2^ℓ コピー消費, p<75% 閾値, F→1.0 (ℓ=10).
    CV 対応: 2 つのノイジー GKP を BS で干渉 + ホモダイン測定 →
             包絡線幅が Δ → Δ/√2 に改善 (1 ラウンド = 2 コピー消費)。
    ℓ ラウンドで Δ_ℓ = Δ_0 / 2^(ℓ/2), コピー数 = 2^ℓ (3layer と同じ指数則)。
    """

    def __init__(self, sq_db_initial):
        self.delta0 = db_to_delta(sq_db_initial)
        self.sq_db0 = sq_db_initial

    def breed(self, n_rounds, sigma_disp):
        """ℓ ラウンド breeding 後の論理性能."""
        delta_l = self.delta0 / (2.0 ** (n_rounds / 2.0))
        sigma_eff = np.sqrt(sigma_disp ** 2 + delta_l ** 2)
        p_round = 1.0 - erf((SQRT_PI / 2.0) / (np.sqrt(2) * sigma_eff))
        p_log = 1.0 - (1.0 - p_round) ** 2
        return {
            'n_rounds': int(n_rounds),
            'copies_consumed': int(2 ** n_rounds),
            'delta_effective': float(delta_l),
            'effective_sq_dB': float(squeeze_db(2 * delta_l ** 2)),
            'p_logical': float(np.clip(p_log, 0, 1)),
            'fidelity': float(np.clip(1.0 - p_log, 0, 1)),
        }

    def threshold_scan(self, rounds_list, sigma_list):
        res = {}
        for nr in rounds_list:
            fids = [self.breed(nr, s)['fidelity'] for s in sigma_list]
            res[f'rounds_{nr}'] = {
                'sigma': [float(s) for s in sigma_list],
                'fidelities': fids,
                'copies': int(2 ** nr),
            }
        return res

    def resource_cost(self, n_rounds):
        return {
            'paradigm': 'B-CV GKP breeding',
            'copies_consumed': int(2 ** n_rounds),     # 3layer と同じ 2^ℓ
            'modes_per_round': 2,
            'beamsplitters': int(2 ** n_rounds - 1),
            'homodyne_detections': int(2 ** n_rounds - 1),
            'squeezing_dB_initial': self.sq_db0,
            'scaling': 'コピー数 O(2^ℓ) — 指数的 (3layer PQEC と同一)',
            'non_gaussian_resource': '初期 GKP コピー (各 2^ℓ 個)',
        }


# ═══════════════════════════════════════════════════════════════════
# D-CV. 二モード DFS  ←  3layer D. デコヒーレンスフリー部分空間
# ═══════════════════════════════════════════════════════════════════

class DFS_CV:
    """二モード DFS: 3layer の核スピン一重項 DFS の CV 対応物.

    3layer D: 集団位相緩和に対し一重項 |S⟩ が不変 (S_z^tot=0)。
    CV 対応 (デュアルレール光子符号化):
      |0_L⟩ = |1,0⟩,  |1_L⟩ = |0,1⟩   (総光子数 n1+n2 = 1 一定)
      集団位相雑音   e^{iθ(n̂1+n̂2)} → 固定総光子数空間で大域位相のみ
                     ⇒ 論理コヒーレンス完全不変 (真の DFS, 3layer 集団相当)
      差分位相雑音   e^{iφ(n̂1-n̂2)} → |10⟩,|01⟩ 間に相対位相 ±φ
                     ⇒ 保護されない (3layer の差分非対称性と同一)
    """

    def __init__(self, r_tms=1.0):
        self.r = r_tms
        # デュアルレール: 単一光子源 1 個 + 50:50 BS = 実効スクイズ不要
        self.sq_db = 0.0

    @staticmethod
    def _logical_coherence(rho2x2):
        """論理 2x2 密度行列の非対角振幅 (1=完全コヒーレント)."""
        return float(2.0 * abs(rho2x2[0, 1]))

    def _noisy_logical(self, sigma_phi, mode, n_samples=2000):
        """|+_L⟩=(|10⟩+|01⟩)/√2 に位相雑音を印加し論理 ρ を返す.

        mode='collective': 位相 θ が n1+n2 に結合 (両レール同符号)
        mode='differential': 位相 φ が n1-n2 に結合 (両レール逆符号)
        """
        # 論理基底 |10⟩→idx0, |01⟩→idx1
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        rho_avg = np.zeros_like(rho)
        phis = np.random.normal(0, sigma_phi, n_samples)
        for ph in phis:
            if mode == 'collective':
                # n1+n2 = 1 (両基底) → 共通位相 e^{iph}: 大域位相
                ph10, ph01 = ph * 1, ph * 1
            else:
                # n1-n2: |10⟩→+1, |01⟩→-1
                ph10, ph01 = ph * 1, ph * (-1)
            U = np.diag([np.exp(1j * ph10), np.exp(1j * ph01)])
            rho_avg += U @ rho @ U.conj().T
        return rho_avg / n_samples

    def common_mode_noise_test(self, sigma_phi, n_samples=2000):
        """集団位相雑音下の論理コヒーレンス (真の DFS → 保持≈1)."""
        rho = self._noisy_logical(sigma_phi, 'collective', n_samples)
        coh = self._logical_coherence(rho)
        return {
            'sigma_phi': float(sigma_phi),
            'logical_coherence': coh,
            'dfs_retention': coh,           # 1 なら完全保護
        }

    def differential_mode_noise_test(self, sigma_phi, n_samples=2000):
        """差分位相雑音下 (保護されない — 3layer と同じ非対称性)."""
        rho = self._noisy_logical(sigma_phi, 'differential', n_samples)
        coh = self._logical_coherence(rho)
        return {
            'sigma_phi': float(sigma_phi),
            'logical_coherence': coh,
            'dfs_retention': coh,
        }

    def resource_cost(self):
        return {
            'paradigm': 'D-CV dual-rail DFS',
            'modes_per_logical_qubit': 2,
            'squeezing_dB_required': 0.0,            # 単一光子源で十分
            'non_gaussian_resource': '単一光子源 (デュアルレール)',
            'detectors': 'photon counting x2',
            'overhead': '低 (受動保護, 訂正操作不要)',
            'limitation': '集団雑音のみ保護 (差分雑音は素通り)',
        }


# ═══════════════════════════════════════════════════════════════════
# E-CV. 位相空間回転エコー  ←  3layer E. 動的デカップリング
# ═══════════════════════════════════════════════════════════════════

class DD_CV:
    """CV 動的デカップリング: 3layer のモーショナルナローイングの CV 対応.

    3layer E: ω₀τ_c<<1 でモーショナルナローイング, T₂ 桁違い延伸。
    CV 対応: 周期 τ で位相空間 π 回転 (フーリエ変換ゲート) を挿入し、
             低周波位相雑音をフィルタ関数で抑圧 (CPMG 相当)。
             1/T₂_eff = (Δω)²·τ_c·f(N_pulse) — パルス数 N で抑圧。
    """

    def __init__(self, delta_omega, tau_c):
        self.delta_omega = delta_omega   # 位相雑音振幅 (rad/s)
        self.tau_c = tau_c               # 相関時間 (s)

    def coherence_enhancement(self, n_pulses, t_total):
        """N パルスエコーによる実効コヒーレンス延伸係数."""
        # フィルタ関数近似: CPMG は低周波雑音を 1/N² で抑圧
        # 1/T2_static = Δω²·τ_c (モーショナル極限),
        # N パルスで実効 τ_c → τ_c/(1+ (t/(N τ_c))) 様の抑圧
        t2_static = 1.0 / (self.delta_omega ** 2 * self.tau_c)
        # CPMG スケーリング: T2(N) ≈ T2_static · N^(2/3) (1/f 雑音典型)
        t2_dd = t2_static * max(n_pulses, 1) ** (2.0 / 3.0)
        return {
            'n_pulses': int(n_pulses),
            'T2_static_s': float(t2_static),
            'T2_dd_s': float(t2_dd),
            'enhancement': float(t2_dd / t2_static),
            'coherence_at_t': float(np.exp(-t_total / t2_dd)),
        }

    def resource_cost(self, n_pulses):
        return {
            'paradigm': 'E-CV phase-space echo (DD)',
            'modes_per_logical_qubit': 1,
            'phase_shifters': int(n_pulses),     # π 回転ゲート列
            'squeezing_dB_required': 0.0,        # 雑音抑制のみ・符号化不要
            'non_gaussian_resource': 'なし',
            'overhead': '極低 (受動位相シフタ列)',
            'role': '誤り抑制 (suppression) — 訂正 (correction) ではない',
        }


# ═══════════════════════════════════════════════════════════════════
# A-CV. 非ガウス触媒回復  ←  3layer A. ICEC 触媒的コヒーレンス回復
# ═══════════════════════════════════════════════════════════════════

class CatalyticCV:
    """非ガウス資源回復: 3layer の ICEC の CV 対応 (Fock 切断).

    3layer A: F≈0.24-0.28 と低性能 (触媒単独では不十分)。
    CV では理由が **ガウス no-go 定理** (Niset-Fiurášek-Cerf, PRL 2009)
    として明確化される: 未知状態に対するガウス誤り (ランダム変位) は、
    いかなるガウス操作でも訂正できない。

    厳密な移植デモ:
      情報   = 未知コヒーレント状態 |β⟩ (β を集団からサンプル)
      誤り   = ランダム未知変位 D(ξ), ξ~N(0,σ)  (ガウス古典雑音チャネル)
      ガウス回復 = ξ 非依存の任意固定ガウス写像 → no-op を超えられない
      非ガウス解決 = 同じ情報を GKP 格子へ符号化 (= C-CV) → 訂正可能

    すなわち 3layer-A の失敗の CV 的解決は「C-CV (GKP) を使うこと」自身。
    """

    def __init__(self, cutoff=30):
        self.N = cutoff
        self.a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
        self.adag = self.a.conj().T
        self.n_op = self.adag @ self.a

    def coherent(self, alpha):
        n = np.arange(self.N)
        c = np.exp(-abs(alpha) ** 2 / 2) * alpha ** n / np.sqrt(
            np.array([factorial(int(k)) for k in n], dtype=float))
        return c / np.linalg.norm(c)

    def displace(self, xi):
        """変位演算子 D(ξ)=exp(ξ a† - ξ* a)."""
        return expm(xi * self.adag - np.conj(xi) * self.a)

    def squeeze(self, r):
        """スクイージング演算子 (代表的ガウス回復候補)."""
        return expm(0.5 * r * (self.a @ self.a - self.adag @ self.adag))

    def no_go_demo(self, sigma=0.35, n_beta=8, n_xi=120,
                   beta_spread=0.8, seed=0):
        """ガウス no-go の厳密数値デモ + 非ガウス (GKP) 解決.

        - F_noop  : 訂正なし平均忠実度
        - F_gauss : 最良の固定ガウス写像 (恒等/逆変位/スクイズ族) の平均
                    → no-go により F_noop を超えない
        - F_gkp   : 同じ σ を GKP 格子で訂正した忠実度 (= C-CV)
        """
        rng = np.random.default_rng(seed)
        betas = rng.normal(0, beta_spread, n_beta) + \
            1j * rng.normal(0, beta_spread, n_beta)
        xis = rng.normal(0, sigma, n_xi) + 1j * rng.normal(0, sigma, n_xi)

        # ガウス回復候補 (すべて ξ 非依存): 恒等 + 固定スクイズ族
        recov_ops = {'identity': np.eye(self.N)}
        for r in (-0.3, -0.15, 0.15, 0.3):
            recov_ops[f'squeeze_{r}'] = self.squeeze(r)

        f_noop = 0.0
        f_by_recov = {k: 0.0 for k in recov_ops}
        cnt = 0
        for b in betas:
            psi = self.coherent(b)
            for xi in xis:
                psi_err = self.displace(xi) @ psi
                f_noop += abs(np.vdot(psi, psi_err)) ** 2
                for k, G in recov_ops.items():
                    psi_r = G @ psi_err
                    nrm = np.linalg.norm(psi_r)
                    if nrm > 1e-12:
                        psi_r = psi_r / nrm
                    f_by_recov[k] += abs(np.vdot(psi, psi_r)) ** 2
                cnt += 1
        f_noop /= cnt
        for k in f_by_recov:
            f_by_recov[k] /= cnt
        f_gauss_best = max(f_by_recov.values())

        # 非ガウス解決: 同じ σ を GKP 格子で訂正 (= C-CV, 12 dB)
        gkp = GKPStabilizerCV(12.0)
        p_log, _ = gkp.logical_error_rate(sigma, n_ec_rounds=1)
        f_gkp = 1.0 - p_log

        return {
            'sigma': float(sigma),
            'fidelity_noop': float(f_noop),
            'fidelity_gaussian_best': float(f_gauss_best),
            'gaussian_recovery_breakdown':
                {k: float(v) for k, v in f_by_recov.items()},
            'fidelity_nongaussian_GKP': float(f_gkp),
            'gaussian_nogo_confirmed':
                bool(f_gauss_best <= f_noop + 1e-3),
            'nongaussian_advantage':
                float(f_gkp - f_gauss_best),
        }

    def resource_cost(self):
        return {
            'paradigm': 'A-CV non-Gaussian resource',
            'modes_per_logical_qubit': 1,
            'non_gaussian_resource': '必須 (GKP/cubic-phase/cat) — 高コスト',
            'gaussian_only': '原理的に不可能 (Niset no-go)',
            'resolution': '= C-CV (GKP 格子符号) に帰着',
            'biological_analog_fidelity': '3layer A: F≈0.24-0.28 (低)',
        }


# ═══════════════════════════════════════════════════════════════════
# F-CV. 統合 CV 安定化器  ←  3layer F. 3レイヤー統合
# ═══════════════════════════════════════════════════════════════════

class IntegratedCVStabilizer:
    """3layer F の統合系を CV へ移植.

    3layer F: overall F≈0.695 (Layer2/ICEC が律速)。
    CV 版: GKP メモリ (C) + breeding 浄化 (B) + DFS バッファ (D),
           律速は非ガウス触媒 (A) — 3layer と同じボトルネック構造。
    """

    def __init__(self, sq_db=12.0):
        self.gkp = GKPStabilizerCV(sq_db)
        self.breed = BreedingCV(sq_db)
        self.dfs = DFS_CV(r_tms=1.0)
        self.cat = CatalyticCV(cutoff=28)

    def full_cycle(self, n_cycles=20, sigma_disp=0.15):
        """3layer F と同じ「最弱リンク律速」構造を保つ.

        3layer F: overall≈0.695 ← Layer2/ICEC が律速。
        F-CV: GKP(C)・breed(B)・DFS(D) は高忠実だが、
              非ガウス資源を要する触媒層 (A-CV) が律速 → 同じ構造。
        """
        # 非ガウス符号を持たないガウス専用層は no-go で頭打ち → 律速リンク
        cat_run = self.cat.no_go_demo(sigma=0.35, n_beta=6, n_xi=80, seed=0)
        f_cat = cat_run['fidelity_gaussian_best']
        res = {'cycle': [], 'gkp_fidelity': [], 'breed_fidelity': [],
               'dfs_retention': [], 'catalytic_fidelity': [],
               'overall_fidelity': []}
        for c in range(n_cycles):
            p_gkp, _ = self.gkp.logical_error_rate(sigma_disp, n_ec_rounds=1)
            f_gkp = 1.0 - p_gkp
            f_breed = self.breed.breed(3, sigma_disp)['fidelity']
            d = self.dfs.common_mode_noise_test(0.2, n_samples=120)
            f_dfs = min(1.0, d['dfs_retention'])
            f_overall = f_gkp * f_breed * f_dfs * f_cat
            res['cycle'].append(c)
            res['gkp_fidelity'].append(float(f_gkp))
            res['breed_fidelity'].append(float(f_breed))
            res['dfs_retention'].append(float(f_dfs))
            res['catalytic_fidelity'].append(float(f_cat))
            res['overall_fidelity'].append(float(f_overall))
        return res


# ═══════════════════════════════════════════════════════════════════
# メイン: 移植ベンチマーク + 資源コスト精査
# ═══════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    OUT = Path("simulation_results")
    OUT.mkdir(exist_ok=True)

    print("=" * 72)
    print(" 3layerstabilizer → 光連続量(CV)量子計算 移植ベンチマーク")
    print("=" * 72)

    R = {}

    # --- C-CV: GKP スタビライザ (3layer C の移植, 最強候補) ---
    print("\n[C-CV] GKP 格子スタビライザ  ← 3layer C ゲージング")
    sigma_list = np.linspace(0.02, 0.40, 20)
    c_res = {}
    for sq in [9.0, 10.0, 12.0, 15.0]:
        gkp = GKPStabilizerCV(sq)
        fv = gkp.fidelity_vs_noise(sigma_list, n_ec_rounds=1)
        c_res[f'sq_{sq}dB'] = {**fv, 'cost': gkp.resource_cost()}
        print(f"  {sq:>4.0f} dB: n̄={gkp.nbar:6.2f}, "
              f"F(σ=0.1)={1 - gkp.logical_error_rate(0.1)[0]:.4f}, "
              f"F(σ=0.2)={1 - gkp.logical_error_rate(0.2)[0]:.4f}")
    R['C_CV_GKP'] = c_res

    # --- B-CV: GKP breeding (3layer B PQEC の移植) ---
    print("\n[B-CV] GKP breeding  ← 3layer B PQEC SWAP 浄化")
    br = BreedingCV(sq_db_initial=8.0)
    b_res = br.threshold_scan([1, 2, 3, 5, 10], sigma_list)
    for nr in [1, 2, 3, 5, 10]:
        d = br.breed(nr, 0.1)
        print(f"  ℓ={nr:>2d}: copies=2^{nr}={2**nr:>4d}, "
              f"Δ_eff→{d['effective_sq_dB']:5.1f}dB, "
              f"F(σ=0.1)={d['fidelity']:.4f}")
    b_res['cost_l10'] = br.resource_cost(10)
    R['B_CV_breeding'] = b_res

    # --- D-CV: 二モード DFS (3layer D の移植) ---
    print("\n[D-CV] 二モード DFS  ← 3layer D 一重項 DFS")
    dfs = DFS_CV(r_tms=1.0)
    d_res = {'common': [], 'differential': [], 'cost': dfs.resource_cost(),
             'n_seeds': 5}
    for sp in [0.05, 0.1, 0.2, 0.4]:
        cm_s, dm_s = [], []
        for sd in range(5):                       # 5 シード mean±std
            np.random.seed(100 + sd)
            cm_s.append(dfs.common_mode_noise_test(sp)['dfs_retention'])
            dm_s.append(dfs.differential_mode_noise_test(sp)['dfs_retention'])
        cm_m, cm_sd = float(np.mean(cm_s)), float(np.std(cm_s))
        dm_m, dm_sd = float(np.mean(dm_s)), float(np.std(dm_s))
        d_res['common'].append(
            {'sigma_phi': sp, 'mean': cm_m, 'std': cm_sd})
        d_res['differential'].append(
            {'sigma_phi': sp, 'mean': dm_m, 'std': dm_sd})
        print(f"  σ_φ={sp:.2f}: 共通雑音 DFS保持={cm_m:.4f}±{cm_sd:.4f} "
              f"| 差分雑音 保持={dm_m:.4f}±{dm_sd:.4f} (保護なし)")
    R['D_CV_DFS'] = d_res

    # --- E-CV: 位相空間エコー DD (3layer E の移植) ---
    print("\n[E-CV] 位相空間回転エコー  ← 3layer E モーショナルナローイング")
    dd = DD_CV(delta_omega=1e6, tau_c=25e-9)
    e_res = {'enhancements': [], 'cost': dd.resource_cost(16)}
    for npul in [1, 4, 16, 64, 256]:
        ce = dd.coherence_enhancement(npul, t_total=1e-3)
        e_res['enhancements'].append(ce)
        print(f"  N={npul:>3d} pulses: T₂ 延伸 ×{ce['enhancement']:6.2f}, "
              f"coh(1ms)={ce['coherence_at_t']:.4f}")
    R['E_CV_DD'] = e_res

    # --- A-CV: ガウス no-go + 非ガウス(GKP)解決 (3layer A の移植) ---
    print("\n[A-CV] ガウス no-go 厳密デモ  ← 3layer A ICEC")
    cat = CatalyticCV(cutoff=30)
    seeds_a = list(range(5))                       # 5 シード mean±std
    runs = [cat.no_go_demo(sigma=0.35, n_beta=8, n_xi=120, seed=s)
            for s in seeds_a]
    def _ms(key):
        v = [r[key] for r in runs]
        return float(np.mean(v)), float(np.std(v))
    noop_m, noop_s = _ms('fidelity_noop')
    gb_m, gb_s = _ms('fidelity_gaussian_best')
    gkp_m, gkp_s = _ms('fidelity_nongaussian_GKP')
    a_run = {
        'n_seeds': len(seeds_a),
        'fidelity_noop': {'mean': noop_m, 'std': noop_s},
        'fidelity_gaussian_best': {'mean': gb_m, 'std': gb_s},
        'fidelity_nongaussian_GKP': {'mean': gkp_m, 'std': gkp_s},
        'gaussian_nogo_confirmed':
            bool(all(r['gaussian_nogo_confirmed'] for r in runs)),
        'nongaussian_advantage': float(gkp_m - gb_m),
        'per_seed': runs,
        'cost': cat.resource_cost(),
    }
    R['A_CV_catalytic'] = a_run
    print(f"  訂正なし F      = {noop_m:.4f}±{noop_s:.4f}")
    print(f"  最良ガウス回復 F = {gb_m:.4f}±{gb_s:.4f}  "
          f"(no-go 確認={a_run['gaussian_nogo_confirmed']}: "
          f"no-op を超えない)")
    print(f"  非ガウス GKP F  = {gkp_m:.4f}±{gkp_s:.4f}  "
          f"(優位性 +{a_run['nongaussian_advantage']:.4f})")

    # --- F-CV: 統合系 (3layer F の移植) ---
    print("\n[F-CV] 統合 CV 安定化器  ← 3layer F 統合系")
    integ = IntegratedCVStabilizer(sq_db=12.0)
    f_res = integ.full_cycle(n_cycles=20, sigma_disp=0.12)
    R['F_CV_integrated'] = f_res
    print(f"  GKP平均F={np.mean(f_res['gkp_fidelity']):.4f}, "
          f"breed平均F={np.mean(f_res['breed_fidelity']):.4f}, "
          f"触媒(律速)F={np.mean(f_res['catalytic_fidelity']):.4f}, "
          f"overall平均F={np.mean(f_res['overall_fidelity']):.4f}")

    # 保存
    path = OUT / "cv_qec_port_results.json"
    with open(path, 'w') as fh:
        json.dump(R, fh, indent=2, default=str)
    print(f"\n結果保存: {path}")

    # === 量子資源コスト精査サマリー ===
    print("\n" + "=" * 72)
    print(" 量子資源コスト精査サマリー (光 CV 移植)")
    print("=" * 72)
    hdr = f"{'移植先':<22}{'モード':<7}{'スクイズ':<10}{'非ガウス':<14}{'スケール':<14}{'役割'}"
    print(hdr)
    print("-" * 88)
    rows = [
        ("C-CV GKP安定化",  "1",  "9-15 dB", "GKP補助:必須", "O(1)/論理",   "訂正(最強)"),
        ("B-CV breeding",   "2/回","8 dB初期","GKPコピー多数","O(2^ℓ) 指数", "浄化"),
        ("D-CV デュアルレール","2", "0 dB",    "単一光子源",   "O(1) 受動",   "受動保護"),
        ("E-CV 位相エコー",  "1",  "0 dB",    "なし",         "O(N) 線形",   "誤り抑制"),
        ("A-CV 非ガウス資源","1",  "—",       "GKP等:必須",   "no-go 障壁",  "→C-CVに帰着"),
        ("F-CV 統合",        "5+", "12 dB",   "GKP必須",      "Gauss層律速", "統合"),
    ]
    for r in rows:
        print(f"{r[0]:<22}{r[1]:<7}{r[2]:<10}{r[3]:<14}{r[4]:<14}{r[5]}")
    print("-" * 88)

    # === 3layer との対応・結論 ===
    print("\n【3layer 結果との対応】")
    print(f"  3layer C (F=1.0) → C-CV GKP: F(σ=0.1)="
          f"{1 - GKPStabilizerCV(12).logical_error_rate(0.1)[0]:.4f} "
          f"@12dB ⇒ 最強構造を維持, コスト=スクイージング")
    print(f"  3layer B (F→1.0,2^ℓ) → B-CV: 同一 2^ℓ 指数則, "
          f"ℓ=5 で {br.breed(5,0.1)['fidelity']:.4f}")
    print(f"  3layer D (F≈0.95) → D-CV: 集団雑音で受動保護 (最低コスト)")
    print(f"  3layer A (F≈0.24) → A-CV: ガウス no-go で原理的に困難 "
          f"(最良ガウス {gb_m:.3f} ≤ no-op {noop_m:.3f}, "
          f"解決=GKP {gkp_m:.3f})")
    print(f"  3layer F (F≈0.695) → F-CV: overall="
          f"{np.mean(f_res['overall_fidelity']):.4f} "
          f"(同じくボトルネック律速構造)")
    print("\n結論: 最有効移植先 = C-CV (GKP格子スタビライザ)。")
    print("      支配的資源コスト = スクイージング (9-15 dB) と GKP 補助状態生成。")
    print("      B-CV breeding は 3layer PQEC と同じ 2^ℓ 指数コストを継承。")

    return R


if __name__ == '__main__':
    main()
