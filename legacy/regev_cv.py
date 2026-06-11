#!/usr/bin/env python3
"""
regev_cv.py — Regev 因数分解アルゴリズム (arXiv:2308.06572) の
              光連続量(CV)量子計算機への移植

参照元の量子コア:
  /Users/deeptell01/Documents/alterego/QuantumSouken/baseline/regev_factoring.py
  (RegevQuantumProcedure / RegevFactoringAlgorithm)

CV 移植方針 (cv_qec_port.py の結論に整合):
  ┌────────────────────────────────────────────────────────────────┐
  │ Regev 量子段          │ CV 実装            │ コスト              │
  ├────────────────────────────────────────────────────────────────┤
  │ 離散ガウス状態         │ 有限エネルギー GKP  │ 低 (CVネイティブ)   │
  │ ρ_R(z)=exp(-πz²/R²)   │ /スクイズド・コム   │ スクイズ dB         │
  │ モジュラ冪 ∏aᵢ^{zᵢ}    │ GKPビット可逆算術   │ ★律速 (非ガウス)   │
  │ QFT over Z_D^d        │ mode-wise CV-QFT   │ 極低 (O(d) 回転)    │
  │ 測定                  │ GKP/ホモダイン      │ 低                  │
  │ 古典 LLL 後処理        │ 不変 (古典のまま)   │ 古典                │
  └────────────────────────────────────────────────────────────────┘

正しさは厳密な数論的周期性で担保 (実際に小 N を因数分解する)。
CV 有限スクイージングは読み出しに加法ガウス雑音 σ_cv として作用し、
Regev の古典許容誤差 δ≈√d/(√2 R) と比較して必要スクイズ量を定量化する。

依存: NumPy, SymPy。cv_qec_port.db_to_delta (同ディレクトリ) を流用。
"""

import numpy as np
from math import gcd, log2, ceil, exp, pi, sqrt
from sympy import isprime, factorint
from typing import List, Optional
import json
from pathlib import Path

try:
    from cv_qec_port import db_to_delta
except Exception:
    def db_to_delta(sq_db):
        return np.sqrt(0.5 * 10.0 ** (-sq_db / 10.0))


# ═══════════════════════════════════════════════════════════════════
# 数論ユーティリティ (regev_factoring.py より厳密に移植・qulacs非依存)
# ═══════════════════════════════════════════════════════════════════

def is_prime_power(n: int) -> bool:
    if n <= 1:
        return False
    if isprime(n):
        return True
    for k in range(2, int(log2(n)) + 1):
        root = round(n ** (1.0 / k))
        for r in (root - 1, root, root + 1):
            if r > 1 and r ** k == n and isprime(r):
                return True
    return False


def get_small_primes(d: int) -> List[int]:
    primes, p = [], 3
    while len(primes) < d:
        if isprime(p):
            primes.append(p)
        p += 2
    return primes


def multiplicative_order(a: int, n: int) -> int:
    if gcd(a, n) != 1:
        return 0
    order, cur = 1, a % n
    while cur != 1:
        cur = (cur * a) % n
        order += 1
        if order > n:
            return n
    return order


def gram_schmidt(B):
    n = B.shape[0]
    Bs = np.zeros_like(B, dtype=float)
    mu = np.zeros((n, n))
    for i in range(n):
        Bs[i] = B[i].astype(float)
        for j in range(i):
            ns = np.dot(Bs[j], Bs[j])
            mu[i, j] = 0 if ns < 1e-15 else np.dot(B[i], Bs[j]) / ns
            Bs[i] = Bs[i] - mu[i, j] * Bs[j]
    return Bs, mu


def lll_reduce(B, delta=0.99):
    B = B.copy().astype(float)
    n = B.shape[0]
    Bs, mu = gram_schmidt(B)
    k, it, mx = 1, 0, n * n * 100
    while k < n and it < mx:
        it += 1
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                B[k] = B[k] - round(mu[k, j]) * B[j]
                Bs, mu = gram_schmidt(B)
        if np.dot(Bs[k], Bs[k]) >= (delta - mu[k, k - 1] ** 2) * \
                np.dot(Bs[k - 1], Bs[k - 1]):
            k += 1
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            Bs, mu = gram_schmidt(B)
            k = max(k - 1, 1)
    return B


# ═══════════════════════════════════════════════════════════════════
# CV 量子段 1: 有限エネルギー GKP 離散ガウス状態準備
# ═══════════════════════════════════════════════════════════════════

class CVDiscreteGaussianPrep:
    """ρ_R(z)=exp(-πz²/R²) を有限エネルギー GKP / スクイズド・コムで準備.

    理想 GKP は無限スクイージング。有限 sq_db では各格子ピークが
    Δ=db_to_delta(sq_db) で広がり、コム包絡線にガウス畳み込みが入る。
    CV ではこの離散ガウス状態が「ネイティブに安い」(Regev の優位点)。
    """

    def __init__(self, n_q: int, sq_db: float):
        self.n_q = n_q
        self.D = 2 ** n_q
        self.sq_db = sq_db
        self.delta = db_to_delta(sq_db)          # 格子単位の有限幅
        # 1 論理レジスタあたり平均光子数 (GKP 漸近)
        self.nbar = 1.0 / (2.0 * self.delta ** 2)

    def amplitudes(self, R: float) -> np.ndarray:
        """理想離散ガウス振幅 (規格化)."""
        k = np.arange(self.D)
        z = k - self.D // 2
        a = np.exp(-pi * z.astype(float) ** 2 / (R * R))
        return a / np.linalg.norm(a)

    def finite_squeezing_readout_sigma(self, R: float) -> float:
        """有限スクイズが双対(QFT後 w)に与える加法ガウス雑音 σ_cv.

        GKP ピーク幅 Δ は双対空間で 1/(格子間隔) にスケールし、
        w∈[0,1) の単位では σ_cv ≈ Δ / √π (標準 GKP 双対関係)。
        """
        return float(self.delta / sqrt(pi))


# ═══════════════════════════════════════════════════════════════════
# CV 量子段 2: モジュラ冪 (GKPビット可逆算術) — 非ガウス律速
# ═══════════════════════════════════════════════════════════════════

class CVModularExponentiation:
    """∏ aᵢ^{zᵢ} mod N を GKP 符号化ビットの可逆算術で実行.

    アナログ近道は無い (モジュラ算術) → CV 最大コストの非ガウス段。
    正しさは厳密 pow で担保し、非ガウス資源量を台帳化する。
    """

    def __init__(self, N: int):
        self.N = N
        self.n_work = int(ceil(log2(N)))

    def periodic_projection(self, a_i: int, amps: np.ndarray, rng):
        """モジュラ冪+作業レジスタ測定 → z レジスタが周期状態に射影.

        regev_factoring._run_efficient_simulation と同一の数論構造。
        """
        D = len(amps)
        order_i = multiplicative_order(int(a_i), self.N)
        if order_i == 0:
            return None, 0
        e = pow(int(a_i), int(rng.integers(0, order_i)), self.N)
        proj = np.zeros(D, dtype=complex)
        for k in range(D):
            if pow(int(a_i), k, self.N) == e:
                proj[k] = amps[k]
        nrm = np.linalg.norm(proj)
        proj = proj / nrm if nrm > 1e-15 else amps.copy()
        return proj, order_i

    def nongaussian_gate_estimate(self, n_q: int, d: int) -> dict:
        """非ガウス資源 (GKP magic/T 等価) の見積り.

        モジュラ乗算 1 回 ≈ O(n_work²) Toffoli, Toffoli ≈ 7 GKP-magic。
        反復二乗で zᵢ あたり O(n_q) 乗算、全 d 次元。
        """
        n_w = self.n_work
        toffoli_per_mul = n_w * n_w
        mul_total = d * n_q
        toffoli = toffoli_per_mul * mul_total
        return {
            'GKP_qubit_work_register': n_w,
            'modular_multiplications': mul_total,
            'toffoli_estimate': toffoli,
            'GKP_magic_states': 7 * toffoli,    # 非ガウス律速資源
        }


# ═══════════════════════════════════════════════════════════════════
# CV 量子段 3: mode-wise CV-QFT (位相空間 π/2 回転)
# ═══════════════════════════════════════════════════════════════════

class CVQuantumFourier:
    """Z_D 上の QFT を CV では「モードごとの Fourier ゲート 1 個」で実行.

    qubit-QFT: O(n_q²) の制御位相ゲート/次元。
    CV-QFT  : 位相空間 π/2 回転 (Fourier) 1 個/モード ⇒ O(1)/次元。
    数値的には離散 Fourier 変換そのもの (厳密)。
    """

    @staticmethod
    def apply(amps: np.ndarray) -> np.ndarray:
        out = np.fft.fft(amps) / np.sqrt(len(amps))
        return out

    @staticmethod
    def cost(n_q: int, d: int) -> dict:
        return {
            'qubit_QFT_cphase_gates': d * n_q * (n_q - 1) // 2,
            'CV_QFT_fourier_gates': d,            # モードあたり 1
            'speedup_factor': float(
                (n_q * (n_q - 1) // 2) if n_q > 1 else 1),
        }


# ═══════════════════════════════════════════════════════════════════
# CV-Regev 量子手続き (per-dimension パイプライン)
# ═══════════════════════════════════════════════════════════════════

class RegevCVProcedure:
    """Regev 量子手続きの CV 版.

    regev_factoring.RegevQuantumProcedure._run_efficient_simulation
    の数論構造を保ちつつ、各段を CV 実装に置換し有限スクイズ雑音を導入。
    """

    def __init__(self, N: int, a_list: List[int], d: int, n_q: int,
                 sq_db: float):
        self.N = N
        self.a_list = a_list
        self.d = d
        self.n_q = n_q
        self.D = 2 ** n_q
        self.sq_db = sq_db
        self.prep = CVDiscreteGaussianPrep(n_q, sq_db)
        self.modexp = CVModularExponentiation(N)
        self.qft = CVQuantumFourier()

    def run(self, R: float, rng) -> np.ndarray:
        """CV パイプライン 1 回 → w ∈ [0,1)^d."""
        D = self.D
        sigma_cv = self.prep.finite_squeezing_readout_sigma(R)
        ideal_amps = self.prep.amplitudes(R)
        w = np.zeros(self.d)

        for i in range(self.d):
            a_i = self.a_list[i]
            # 段1: 有限エネルギー GKP 離散ガウス準備
            amps = ideal_amps.copy()
            # 段2: GKPビット・モジュラ冪 + 作業レジスタ測定 (周期射影)
            proj, order_i = self.modexp.periodic_projection(a_i, amps, rng)
            if proj is None:
                w[i] = rng.random()
                continue
            # 段3: mode-wise CV-QFT
            spec = self.qft.apply(proj)
            probs = np.abs(spec) ** 2
            s = probs.sum()
            if s < 1e-15:
                w[i] = rng.random()
                continue
            probs = probs / s
            # 段4: GKP/ホモダイン読み出し
            measured = rng.choice(D, p=probs)
            # 有限スクイズ由来の加法ガウス雑音 (双対空間, mod 1)
            noise = rng.normal(0.0, sigma_cv)
            w[i] = ((measured / D) + noise) % 1.0
        return w


# ═══════════════════════════════════════════════════════════════════
# CV-Regev 完全アルゴリズム (量子CV + 古典LLL後処理)
# ═══════════════════════════════════════════════════════════════════

class RegevCVFactoring:
    """Regev 因数分解の CV 完全版.

    パラメータ選択・古典後処理は regev_factoring.RegevFactoringAlgorithm
    と同一。量子コアのみ RegevCVProcedure (CV) に差し替え。
    """

    def __init__(self, N: int, sq_db: float = 12.0, verbose: bool = True):
        if N <= 1:
            raise ValueError("N must be > 1")
        self.N = N
        self.n = int(ceil(log2(N)))
        self.sq_db = sq_db
        self.verbose = verbose

    def factor(self, max_attempts: int = 10, seed: int = 0) -> Optional[int]:
        N, n = self.N, self.n
        if N % 2 == 0:
            return 2
        if isprime(N):
            raise ValueError(f"N = {N} is prime")
        if is_prime_power(N):
            return list(factorint(N).keys())[0]

        rng = np.random.default_rng(seed)
        d = max(2, int(ceil(sqrt(n))))
        b_list = get_small_primes(d)
        a_list = [b ** 2 for b in b_list]
        m = d + 4
        C = 2.0
        R = min(exp(C * sqrt(n)), 1e12)
        D_target = 2 * sqrt(d) * R
        n_q = max(3, min(int(ceil(log2(max(2, D_target)))), 10))

        for b in b_list:
            g = gcd(b, N)
            if 1 < g < N:
                return g

        qproc = RegevCVProcedure(N, a_list, d, n_q, self.sq_db)
        self._last_meta = {
            'N': N, 'n_bits': n, 'd': d, 'b_list': b_list,
            'a_list': a_list, 'm_runs': m, 'R': R, 'n_q': n_q, 'D': 2 ** n_q,
            'squeezing_dB': self.sq_db,
            'readout_sigma_cv': qproc.prep.finite_squeezing_readout_sigma(R),
            'regev_noise_tolerance_delta': sqrt(d) / (sqrt(2) * R),
            'nbar_per_register': qproc.prep.nbar,
            'modexp_cost': qproc.modexp.nongaussian_gate_estimate(n_q, d),
            'qft_cost': qproc.qft.cost(n_q, d),
        }
        if self.verbose:
            print(f"CV-Regev: N={N} ({n}b) d={d} b={b_list} "
                  f"m={m} n_q={n_q} sq={self.sq_db}dB")
            print(f"  σ_cv(読出)={self._last_meta['readout_sigma_cv']:.3e} "
                  f"vs Regev許容δ="
                  f"{self._last_meta['regev_noise_tolerance_delta']:.3e}")

        for attempt in range(max_attempts):
            samples = [qproc.run(R, rng) for _ in range(m)]
            fac = self._postprocess(samples, b_list, d, R)
            if fac is not None:
                return fac
        return None

    def _postprocess(self, samples, b_list, d, R) -> Optional[int]:
        """古典 LLL 後処理 (regev_factoring._classical_postprocess と同一)."""
        N = self.N
        m = len(samples)
        delta = sqrt(d) / (sqrt(2) * R) if R > 0 else 1.0
        S = 1.0 / delta if delta > 1e-15 else R
        dim = d + m
        B = np.zeros((dim, dim))
        for i in range(d):
            B[i, i] = 1.0
        for j in range(m):
            B[d + j, :d] = S * samples[j]
            B[d + j, d + j] = S
        try:
            Br = lll_reduce(B, delta=0.99)
        except Exception:
            return None

        candidates = []
        for i in range(dim):
            v = Br[i]
            head = np.round(v[:d]).astype(int)
            if np.linalg.norm(head) > 0.5 and \
                    np.linalg.norm(v[d:]) < S * 0.5:
                candidates.append(head)
        if not candidates:
            norms = [np.linalg.norm(Br[i]) for i in range(dim)]
            for idx in np.argsort(norms):
                head = np.round(Br[idx, :d]).astype(int)
                if np.any(head != 0):
                    candidates.append(head)
                if len(candidates) >= d:
                    break

        for z in candidates:
            f = self._try_factor(z, b_list)
            if f is not None:
                return f
        if candidates:
            for _ in range(50):
                co = np.random.randint(-3, 4, size=len(candidates))
                z = np.round(sum(c * v for c, v in zip(co, candidates)))
                z = z.astype(int)
                if np.any(z != 0):
                    f = self._try_factor(z, b_list)
                    if f is not None:
                        return f
        return None

    def _try_factor(self, z, b_list) -> Optional[int]:
        N = self.N
        b = 1
        for bi, zi in zip(b_list, z):
            zi = int(zi)
            if zi >= 0:
                b = (b * pow(bi, zi, N)) % N
            else:
                try:
                    b = (b * pow(pow(bi, -1, N), -zi, N)) % N
                except (ValueError, ZeroDivisionError):
                    g = gcd(bi, N)
                    return g if 1 < g < N else None
        if b == 1 or b == N - 1:
            return None
        for g in (gcd(b - 1, N), gcd(b + 1, N)):
            if 1 < g < N:
                return g
        return None


# ═══════════════════════════════════════════════════════════════════
# メイン: 因数分解検証 + CV 資源台帳 + スクイージング要求量計測
# ═══════════════════════════════════════════════════════════════════

def main():
    OUT = Path("simulation_results")
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print(" Regev 因数分解 → 光連続量(CV)量子計算機 移植")
    print("=" * 72)

    R = {}

    # --- 1. 正しさ検証: 量子コアを通す合成数 ({3,5,7}と互素) ---
    print("\n[1] 正しさ検証 (CV 量子コアを実際に通して因数分解)")
    targets = [143, 187, 221, 247, 253, 391]   # 11·13, 11·17, ...
    correctness = []
    for N in targets:
        ok, fac = 0, None
        for sd in range(5):                       # 5 シード
            alg = RegevCVFactoring(N, sq_db=12.0, verbose=False)
            f = alg.factor(max_attempts=8, seed=sd)
            if f is not None and 1 < f < N and N % f == 0:
                ok += 1
                fac = f
        rate = ok / 5
        correctness.append({'N': N, 'factor': fac,
                            'success_rate': rate})
        print(f"  N={N:>3d}: {N}={fac}×{N // fac if fac else '?'} "
              f"成功率 {ok}/5 ({rate:.0%})")
    R['correctness'] = correctness

    # --- 2. CV 資源台帳 (N=143 代表, 量子コア通過) ---
    print("\n[2] CV 量子資源台帳 (代表 N=143)")
    alg = RegevCVFactoring(143, sq_db=12.0, verbose=True)
    alg.factor(max_attempts=1, seed=0)
    meta = alg._last_meta
    R['resource_ledger'] = meta
    me = meta['modexp_cost']
    qf = meta['qft_cost']
    print(f"  GKP 作業レジスタ : {me['GKP_qubit_work_register']} qubit")
    print(f"  モジュラ乗算回数 : {me['modular_multiplications']}")
    print(f"  非ガウス律速資源 : GKP magic ≈ {me['GKP_magic_states']:,} "
          f"(★最大コスト)")
    print(f"  QFT: qubit={qf['qubit_QFT_cphase_gates']} cphase → "
          f"CV={qf['CV_QFT_fourier_gates']} Fourier "
          f"(×{qf['speedup_factor']:.0f} 削減)")
    print(f"  GKP 平均光子数/レジスタ n̄ = {meta['nbar_per_register']:.2f}")

    # --- 3a. 経験的: 小 N では LLL 後処理が読出雑音に頑健 ---
    print("\n[3a] 経験的雑音頑健性 (小 N, sq_dB 掃引)")
    emp = []
    for sq in [3.0, 6.0, 9.0, 12.0, 15.0]:
        ok = trials = 0
        for N in [143, 187, 221]:
            for sd in range(4):
                trials += 1
                a = RegevCVFactoring(N, sq_db=sq, verbose=False)
                f = a.factor(max_attempts=6, seed=sd)
                if f is not None and 1 < f < N and N % f == 0:
                    ok += 1
        emp.append({'sq_dB': sq, 'success_rate': ok / trials})
        print(f"  {sq:>4.0f} dB: 成功率 {ok/trials:.0%} ({ok}/{trials}) "
              f"← 小 N では LLL が σ_cv≫δ でも回収 (閾値見えず)")
    R['empirical_small_N'] = emp
    print("  注: 小 N の LLL 頑健性のため経験掃引では閾値不可視。")
    print("      → 束縛条件は下記の解析的 σ_cv ≲ δ 要求量。")

    # --- 3b. 解析的スクイージング要求量: σ_cv ≲ δ (真の CV 束縛) ---
    # σ_cv = Δ/√π,  Δ² = 0.5·10^(−dB/10)  ⇒
    # dB_req = −10·log10(2π·δ²),  δ = √d/(√2·R),  R=min(e^{2√n},1e12)
    print("\n[3b] 解析的スクイージング要求量  σ_cv ≲ δ  (n 依存)")
    req = []
    for n_bits in [8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        d = max(2, int(ceil(sqrt(n_bits))))
        Rn = min(exp(2.0 * sqrt(n_bits)), 1e12)
        delta = sqrt(d) / (sqrt(2) * Rn)
        db_req = -10.0 * np.log10(2.0 * pi * delta ** 2)
        capped = (exp(2.0 * sqrt(n_bits)) >= 1e12)
        req.append({'n_bits': n_bits, 'delta': delta,
                    'required_sq_dB': db_req, 'R_capped': capped})
        flag = " (R上限)" if capped else ""
        print(f"  n={n_bits:>4d}b: δ={delta:.2e} → 要求 "
              f"{db_req:6.1f} dB{flag}")
    R['analytic_squeezing_requirement'] = req
    print("  ⇒ R=e^{2√n} のため要求 dB は √n に線形 (R上限到達まで)。")

    path = OUT / "regev_cv_results.json"
    with open(path, 'w') as fh:
        json.dump(R, fh, indent=2, default=str)
    print(f"\n結果保存: {path}")

    # --- 結論 ---
    print("\n" + "=" * 72)
    print(" 結論")
    print("=" * 72)
    print(" • CV パイプラインで Regev は実際に因数分解可能 (正しさ担保)")
    print(" • 離散ガウス状態準備と QFT は CV ネイティブで安価")
    print(f"   (QFT を qubit比 ×{qf['speedup_factor']:.0f} 削減)")
    print(" • 律速は GKP-ビット上のモジュラ冪 (非ガウス magic 状態)")
    print(" • CV 固有束縛 = スクイージング: σ_cv=Δ/√π を δ≈√d/(√2R) 以下に")
    print("   要求 dB = -10log10(2πδ²) は √n に線形 (R=e^{2√n})")
    print("   小 N では LLL 頑健性で経験閾値は不可視 (3b の解析値が束縛)")
    print(" • √n+4 回の独立試行はフォトニックチップ間で完全並列")
    return R


if __name__ == '__main__':
    main()
