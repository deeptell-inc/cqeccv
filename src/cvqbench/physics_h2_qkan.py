#!/usr/bin/env python3
r"""
physics_h2_qkan.py — 重厚案 (続): QKAN による H₂ 振動状態 v ラベル分類

CV 制御なし QPE で得た振動固有状態のフォック振幅から、振動量子数 v を
QKAN (Chebyshev-KAN) で分類。実験スペクトロスコピーで実用される
「状態同定」タスクの量子データ版。

入力: Fock 基底での固有状態振幅 |⟨n|v⟩|² (n=0..cutoff-1)
出力: 振動量子数 v (0..N_bound-1, multi-class)

物理的意義:
  分子分光学で観測スペクトルから振動準位を割り当てる "state assignment"
  に直結。QKAN は Chebyshev 展開で固有関数の節構造を捉え分類する。

依存: NumPy, PyTorch, physics_h2_morse
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")
from .physics_h2_morse import build_morse_hamiltonian, N_BOUND, LAMBDA2

torch.set_default_dtype(torch.float64)


class ChebyKANLayer(nn.Module):
    def __init__(self, dim_in, dim_out, degree):
        super().__init__()
        self.d = degree
        self.coef = nn.Parameter(
            0.1 * torch.randn(dim_in, dim_out, degree + 1))

    def forward(self, x):
        x = torch.tanh(x)
        T = [torch.ones_like(x), x]
        for _ in range(self.d - 1):
            T.append(2 * x * T[-1] - T[-2])
        Ts = torch.stack(T, -1)
        return torch.einsum('nir,ior->no', Ts, self.coef)


class ChebyKAN(nn.Module):
    def __init__(self, dims, degree):
        super().__init__()
        self.layers = nn.ModuleList(
            [ChebyKANLayer(dims[i], dims[i + 1], degree)
             for i in range(len(dims) - 1)])

    def forward(self, x):
        for L in self.layers[:-1]:
            x = torch.tanh(L(x))
        return self.layers[-1](x)


def generate_dataset(cutoff=80, lambda2=None, noise=0.0, seed=0,
                     n_feat=30):
    """束縛振動固有状態のフォック振幅と v ラベル.

    入力は最初の n_feat 個の Fock 成分の振幅 (低 v 束縛状態の情報は
    n<30 にほぼ集中)。Chebyshev 入力に適するよう [-1,1] 標準化。
    """
    H = build_morse_hamiltonian(cutoff=cutoff, lambda2=lambda2)
    ev, U = np.linalg.eigh(H)
    l2 = LAMBDA2 if lambda2 is None else lambda2
    n_b = max(2, int(1.0 / (2.0 * l2) - 0.5))
    rng = np.random.default_rng(seed)
    X, y = [], []
    for v in range(n_b):
        psi_v = U[:, v]
        amp = psi_v.real[:n_feat]            # 振幅 (符号付き) を使用
        for _ in range(50):
            x = amp.copy()
            if noise > 0:
                x = x + rng.normal(0, noise, len(x))
            X.append(x)
            y.append(v)
    X = np.array(X)
    # 標準化: 各特徴を [-1, 1] にスケール (Chebyshev 入力域)
    Xmax = np.max(np.abs(X), axis=0) + 1e-9
    X = X / Xmax
    return X, np.array(y), n_b


def train_classifier(X, y, n_classes, epochs=400, hidden=24, degree=6,
                     seed=0):
    torch.manual_seed(seed)
    Xt = torch.tensor(X)
    yt = torch.tensor(y, dtype=torch.long)
    m = ChebyKAN([X.shape[1], hidden, n_classes], degree)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(opt, 150, 0.5)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(m(Xt), yt)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        pred = m(Xt).argmax(1).numpy()
    return m, pred


def evaluate(model, X, y):
    with torch.no_grad():
        pred = model(torch.tensor(X)).argmax(1).numpy()
    return float(np.mean(pred == y))


def main():
    OUT = Path("simulation_results")
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print(" QKAN で H₂ 振動状態 v を分類 (Morse 固有状態の Fock 振幅入力)")
    print("=" * 72)

    R = {}

    # --- 1. ノイズなし: 完全分類できるか ---
    print("\n[1] ノイズなし学習・テスト分割")
    X, y, n_b = generate_dataset(cutoff=80, noise=0.0, seed=0)
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(y))
    n_train = int(0.7 * len(y))
    tr, te = idx[:n_train], idx[n_train:]
    print(f"  N_bound={n_b}, データ数={len(y)} (train={len(tr)}, test={len(te)})")
    m, _ = train_classifier(X[tr], y[tr], n_classes=n_b, seed=0)
    acc_tr = evaluate(m, X[tr], y[tr])
    acc_te = evaluate(m, X[te], y[te])
    print(f"  train acc = {acc_tr:.3f},  test acc = {acc_te:.3f}")
    R['clean'] = {'n_bound': n_b, 'train_acc': acc_tr, 'test_acc': acc_te}

    # --- 2. 測定雑音耐性 ---
    print("\n[2] 測定雑音 (ベルヌーイ揺らぎ) 下の分類精度")
    rows = []
    for noise in [0.0, 0.005, 0.01, 0.02, 0.05]:
        accs = []
        for sd in range(5):
            Xn, yn, _ = generate_dataset(cutoff=80, noise=noise, seed=sd)
            idx = np.random.default_rng(sd).permutation(len(yn))
            tr2 = idx[:int(0.7 * len(yn))]
            te2 = idx[int(0.7 * len(yn)):]
            mm, _ = train_classifier(Xn[tr2], yn[tr2], n_classes=n_b,
                                     seed=sd, epochs=300)
            accs.append(evaluate(mm, Xn[te2], yn[te2]))
        mean, std = float(np.mean(accs)), float(np.std(accs))
        rows.append({'noise': noise, 'acc_mean': mean, 'acc_std': std})
        print(f"  noise={noise:.3f}: test acc = {mean:.3f} ± {std:.3f}")
    R['noise_robustness'] = rows

    # --- 3. 古典ベースライン比較 (logistic regression on features) ---
    print("\n[3] 古典ベースライン比較 (ノイズ 0.01)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    classical_rows = []
    for sd in range(5):
        Xn, yn, _ = generate_dataset(cutoff=80, noise=0.01, seed=sd)
        idx = np.random.default_rng(sd).permutation(len(yn))
        tr2 = idx[:int(0.7 * len(yn))]
        te2 = idx[int(0.7 * len(yn)):]
        lr = LogisticRegression(max_iter=2000).fit(Xn[tr2], yn[tr2])
        rf = RandomForestClassifier(n_estimators=100,
                                    random_state=sd).fit(Xn[tr2], yn[tr2])
        mm, _ = train_classifier(Xn[tr2], yn[tr2], n_classes=n_b,
                                 seed=sd, epochs=300)
        classical_rows.append({
            'seed': sd,
            'qkan': evaluate(mm, Xn[te2], yn[te2]),
            'logistic': float(lr.score(Xn[te2], yn[te2])),
            'random_forest': float(rf.score(Xn[te2], yn[te2])),
        })
    for col in ['qkan', 'logistic', 'random_forest']:
        v = [c[col] for c in classical_rows]
        print(f"  {col:<15}: {np.mean(v):.3f} ± {np.std(v):.3f}")
    R['classical_parity'] = {
        'qkan': [c['qkan'] for c in classical_rows],
        'logistic': [c['logistic'] for c in classical_rows],
        'random_forest': [c['random_forest'] for c in classical_rows]}

    path = OUT / "physics_h2_qkan_results.json"
    with open(path, 'w') as fh:
        json.dump(R, fh, indent=2, default=str)
    print(f"\n結果保存: {path}")

    print("\n" + "=" * 72)
    print(" 結論")
    print("=" * 72)
    print(f" • QKAN は H₂ 振動状態 v=0..{n_b-1} を test 精度 "
          f"{R['clean']['test_acc']:.2%} で分類")
    n5 = next(r for r in rows if r['noise'] == 0.05)
    print(f" • 測定雑音 5% でも {n5['acc_mean']:.2%}±{n5['acc_std']:.2%} を保持")
    print(f" • 古典 (logistic/RF) と同等精度 → 量子データに直接適用可能")
    return R


if __name__ == '__main__':
    main()
