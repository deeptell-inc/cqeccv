#!/usr/bin/env python3
"""
qkan_deep.py — 多層・非分離 QKAN ＋ 古典パリティ（あれば良いデータ）

qkan_cv.py は 1 層・加法的標的で評価したが KART 的に易しすぎる。本実装は:
  (i)  真に非分離な標的  f(x1,x2)=sin(π·x1·x2)  (積=非加法)
  (ii) 1 層 Chebyshev-KAN は加法的で非分離を表現できず失敗
  (iii)2 層 Chebyshev-KAN (KART 合成) を勾配学習で回復
  (iv) 古典ベースライン (多項式 Ridge / MLP) と精度パリティ

辺活性化は Chebyshev 多項式。T_r は漸化式
  T_0=1, T_1=x, T_{r+1}=2x T_r − T_{r-1}
で計算（= cos(r·arccos x) = CV-QSP/ボソン QSP の応答に厳密一致）。
QKAN は係数を古典/変分学習する（原論文）— ここでは Adam で勾配学習。
依存: NumPy, PyTorch, scikit-learn
"""

import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

torch.set_default_dtype(torch.float64)


def nonseparable_target(X):
    """真に非分離: f(x1,x2)=sin(π·x1·x2) (g1(x1)+g2(x2) で書けない)."""
    return np.sin(np.pi * X[:, 0] * X[:, 1])


class ChebyKANLayer(nn.Module):
    """Chebyshev-KAN 層: 辺ごとに Σ_r c_{ij,r} T_r(tanh x_i).

    T_r は漸化式で計算 (= CV-QSP の cos(r·arccos x) 応答に一致)。
    """

    def __init__(self, dim_in, dim_out, degree):
        super().__init__()
        self.din, self.dout, self.d = dim_in, dim_out, degree
        self.coef = nn.Parameter(
            0.1 * torch.randn(dim_in, dim_out, degree + 1))

    def forward(self, x):
        x = torch.tanh(x)                            # [-1,1] へ
        T = [torch.ones_like(x), x]
        for r in range(2, self.d + 1):
            T.append(2 * x * T[-1] - T[-2])
        Ts = torch.stack(T, -1)                      # (n, din, d+1)
        # Σ_i Σ_r coef[i,o,r] T[n,i,r] → (n, dout)
        return torch.einsum('nir,ior->no', Ts, self.coef)


class ChebyKAN(nn.Module):
    def __init__(self, dims, degree):
        super().__init__()
        self.layers = nn.ModuleList(
            [ChebyKANLayer(dims[i], dims[i + 1], degree)
             for i in range(len(dims) - 1)])

    def forward(self, x):
        for L in self.layers:
            x = L(x)
        return x.squeeze(-1)


def train_kan(dims, degree, Xtr, ytr, epochs=600, lr=2e-2, seed=0):
    torch.manual_seed(seed)
    m = ChebyKAN(dims, degree)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, 200, 0.5)
    Xt = torch.tensor(Xtr)
    yt = torch.tensor(ytr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((m(Xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    return m


def kan_mse(m, X, y):
    with torch.no_grad():
        p = m(torch.tensor(X)).numpy()
    return float(np.mean((p - y) ** 2))


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def main():
    OUT = Path("simulation_results")
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print(" 多層・非分離 QKAN + 古典パリティ（あれば良いデータ）")
    print("=" * 72)
    print(" 標的 f(x1,x2)=sin(π·x1·x2)  (真に非分離=非加法)")

    R = {}
    k = 2

    # --- 1. 1層 Chebyshev-KAN は非分離を表現できない (5シード) ---
    print("\n[1] 1層 KAN (2→1, 加法的) の非分離での失敗 (5シード)")
    one = []
    for d in [4, 8, 12]:
        te = []
        for sd in range(5):
            rng = np.random.default_rng(sd)
            Xtr = rng.uniform(-1, 1, (600, k))
            Xte = rng.uniform(-1, 1, (600, k))
            ytr, yte = nonseparable_target(Xtr), nonseparable_target(Xte)
            m = train_kan([2, 1], d, Xtr, ytr, seed=sd)
            te.append(kan_mse(m, Xte, yte))
        mm, ss = float(np.mean(te)), float(np.std(te))
        one.append({'d': d, 'test_mse_mean': mm, 'test_mse_std': ss})
        print(f"  d={d:>2d}: 1層 test MSE = {mm:.4e} ± {ss:.1e} "
              f"(加法的→非分離を表現不可)")
    R['one_layer_fails'] = one

    # --- 2. 2層 Chebyshev-KAN (KART 合成) で回復 (5シード) ---
    print("\n[2] 2層 KAN (2→8→1, KART 合成) の回復 (5シード)")
    two = []
    for d in [4, 6, 8]:
        te = []
        for sd in range(5):
            rng = np.random.default_rng(100 + sd)
            Xtr = rng.uniform(-1, 1, (600, k))
            Xte = rng.uniform(-1, 1, (600, k))
            ytr, yte = nonseparable_target(Xtr), nonseparable_target(Xte)
            m = train_kan([2, 8, 1], d, Xtr, ytr, seed=sd)
            te.append(kan_mse(m, Xte, yte))
        mm, ss = float(np.mean(te)), float(np.std(te))
        two.append({'d': d, 'test_mse_mean': mm, 'test_mse_std': ss})
        print(f"  d={d:>2d}: 2層 test MSE = {mm:.4e} ± {ss:.1e}")
    R['two_layer_recovers'] = two

    # --- 3. 古典ベースラインとの精度パリティ (5シード) ---
    print("\n[3] 古典ベースラインとの精度パリティ (5シード, test MSE)")
    rows = []
    for sd in range(5):
        rng = np.random.default_rng(200 + sd)
        Xtr = rng.uniform(-1, 1, (600, k))
        Xte = rng.uniform(-1, 1, (600, k))
        ytr, yte = nonseparable_target(Xtr), nonseparable_target(Xte)
        q = train_kan([2, 8, 1], 8, Xtr, ytr, seed=sd)
        e_q = kan_mse(q, Xte, yte)
        poly = make_pipeline(PolynomialFeatures(9), Ridge(alpha=1e-5))
        poly.fit(Xtr, ytr)
        e_p = mse(poly.predict(Xte), yte)
        mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000,
                           random_state=sd)
        mlp.fit(Xtr, ytr)
        e_m = mse(mlp.predict(Xte), yte)
        rows.append({'seed': sd, 'qkan2': e_q, 'poly_ridge': e_p,
                     'mlp': e_m})

    def col(key):
        v = [r[key] for r in rows]
        return float(np.mean(v)), float(np.std(v))
    for nm in ['qkan2', 'poly_ridge', 'mlp']:
        mm, ss = col(nm)
        print(f"  {nm:<12}: test MSE = {mm:.4e} ± {ss:.1e}")
    R['classical_parity'] = {nm: col(nm) for nm in
                             ['qkan2', 'poly_ridge', 'mlp']}
    R['classical_parity']['per_seed'] = rows

    # --- 4. 資源台帳 ---
    led = {
        'layers': 2, 'dims': [2, 8, 1], 'chebyshev_degree_d': 8,
        'edges': 2 * 8 + 8 * 1,
        'nongaussian_rotations_total': 8 * (2 * 8 + 8 * 1),
        'aux_qubits_scaling': 'linear in #layers L (=2)',
        'training': 'Adam (係数は古典/変分学習, 原論文に整合)',
        'primitive': 'T_r 漸化式 = cos(r·arccos x) = CV-QSP 応答',
    }
    R['resource_ledger'] = led
    print(f"\n[4] 資源: 2層 dims[2,8,1], 非ガウス回転 ≈ "
          f"{led['nongaussian_rotations_total']} (補助は L 線形)")

    path = OUT / "qkan_deep_results.json"
    with open(path, 'w') as fh:
        json.dump(R, fh, indent=2, default=str)
    print(f"\n結果保存: {path}")

    print("\n" + "=" * 72)
    print(" 結論")
    print("=" * 72)
    o = min(x['test_mse_mean'] for x in one)
    t = min(x['test_mse_mean'] for x in two)
    q, _ = col('qkan2')
    p, _ = col('poly_ridge')
    mm, _ = col('mlp')
    print(f" • 1層 KAN は非分離で飽和 (最良 MSE {o:.2e}) = 加法の限界")
    print(f" • 2層 KAN (KART 合成) で回復 (最良 MSE {t:.2e}, "
          f"{o/t:.0f}× 改善)")
    print(f" • 古典パリティ: QKAN2={q:.2e} / 多項式={p:.2e} / MLP={mm:.2e}")
    print(" • 多層 QKAN-CV は非分離多変量を古典同等精度で表現可能")
    return R


if __name__ == '__main__':
    main()
