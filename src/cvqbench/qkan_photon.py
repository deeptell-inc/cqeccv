#!/usr/bin/env python3
r"""
qkan_photon.py — 振動状態分類を測定可能量と群分割でやり直す (改訂 (c))

従来の physics_h2_qkan には 3 つの欠陥があった:
  1. 入力が「厳密対角化した固有ベクトルの符号付き振幅」であり、CV-QPE の
     出力でも測定可能量でもなかった。
  2. 17 個のテンプレートを 50 回複製し、サンプル単位でランダム分割していた
     ため train と test がテンプレートを共有していた（リーク）。
  3. QKAN・ロジスティック回帰・ランダムフォレストが全条件で厳密に 100%
     となり、手法間の判別力がなかった。

本モジュールの設計:
  - 入力は光子数分布 p_n = |<n|v>|^2 を有限ショットで推定したヒストグラム
    （PNR 検出器で実際に得られる量）。
  - 分割は **分子横断 (leave-one-molecule-out)**。H2/HF/I2 は xe が一桁
    違うので、これは真の外挿汎化テストになる。
  - 追加で **準備条件横断 (leave-one-alpha-out)** も測る。
  - QKAN を logistic regression / random forest と同一データで比較する。

依存: numpy, torch, scikit-learn ([ml] extra)
"""

import json
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

from .physics_h2_morse import MOLECULES, build_morse_hamiltonian

MOL_CUTOFF = {"H2": 80, "HF": 90, "I2": 90}


def photon_number_dataset(molecules=("H2", "HF", "I2"), v_max=8,
                          n_feat=24, shots=10**4, n_rep=20, seed=0):
    """光子数ヒストグラムから (X, y, groups) を作る。

    X[i]     : n_feat 次元の推定光子数分布（有限ショット）
    y[i]     : 振動量子数 v
    groups[i]: 分子ラベル（群分割用）
    """
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for mol in molecules:
        p = MOLECULES[mol]
        l2 = p["omega_x_cm"] / p["omega_e_cm"]
        H = build_morse_hamiltonian(cutoff=MOL_CUTOFF[mol], lambda2=l2)
        _, U = np.linalg.eigh(H)
        for v in range(v_max + 1):
            prob = np.abs(U[:, v]) ** 2
            prob = prob[:n_feat]
            s = prob.sum()
            if s <= 0:
                continue
            prob = prob / s                     # 観測窓内で規格化
            for _ in range(n_rep):
                if shots and shots > 0:
                    counts = rng.multinomial(shots, prob)
                    x = counts / shots          # 有限ショット推定
                else:
                    x = prob
                X.append(x)
                y.append(v)
                g.append(mol)
    return np.array(X), np.array(y), np.array(g)


# ── QKAN (Chebyshev-KAN) ────────────────────────────────────────────

def _train_qkan(Xtr, ytr, n_classes, degree=6, hidden=24, epochs=400,
                seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class Layer(nn.Module):
        def __init__(self, din, dout, d):
            super().__init__()
            self.d = d
            self.coef = nn.Parameter(0.1 * torch.randn(din, dout, d + 1))

        def forward(self, x):
            x = torch.tanh(x)
            T = [torch.ones_like(x), x]
            for _ in range(self.d - 1):
                T.append(2 * x * T[-1] - T[-2])
            return torch.einsum("nir,ior->no", torch.stack(T, -1), self.coef)

    class Net(nn.Module):
        def __init__(self, dims, d):
            super().__init__()
            self.ls = nn.ModuleList(
                [Layer(dims[i], dims[i + 1], d) for i in range(len(dims) - 1)])

        def forward(self, x):
            for L in self.ls[:-1]:
                x = torch.tanh(L(x))
            return self.ls[-1](x)

    # 入力を [-1,1] へ（Chebyshev の定義域）
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xn = np.clip((Xtr - mu) / (3 * sd), -1, 1)
    m = Net([Xtr.shape[1], hidden, n_classes], degree).double()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    lossf = torch.nn.CrossEntropyLoss()
    xt = torch.tensor(Xn)
    yt = torch.tensor(ytr, dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad()
        lossf(m(xt), yt).backward()
        opt.step()

    def predict(Xte):
        Xt = np.clip((Xte - mu) / (3 * sd), -1, 1)
        with torch.no_grad():
            return m(torch.tensor(Xt)).argmax(1).numpy()
    return predict


def _baselines(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    lr = LogisticRegression(max_iter=3000).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(
        Xtr, ytr)
    return {"logistic": lr.predict, "random_forest": rf.predict}


def evaluate_split(X, y, groups, held_out, n_classes, seed=0):
    """held_out 群を test、残りを train にして各手法の精度を返す。"""
    te = groups == held_out
    tr = ~te
    if te.sum() == 0 or tr.sum() == 0:
        return None
    out = {}
    pred = _train_qkan(X[tr], y[tr], n_classes, seed=seed)
    out["qkan"] = float(np.mean(pred(X[te]) == y[te]))
    for name, f in _baselines(X[tr], y[tr]).items():
        out[name] = float(np.mean(f(X[te]) == y[te]))
    out["chance"] = float(1.0 / n_classes)
    return out


def main():
    OUT = Path("simulation_results")
    OUT.mkdir(exist_ok=True)
    R = {}
    v_max = 8
    n_classes = v_max + 1
    print("=" * 74)
    print(" 振動状態分類：光子数データ + 群分割 (改訂 (c))")
    print("=" * 74)

    # --- (i) 分子横断汎化 (leave-one-molecule-out) ---
    print("\n[i] 分子横断汎化 — 2分子で学習し、未見の分子でテスト")
    print("    入力 = 光子数分布 (PNR 測定可能), shots=1e4")
    X, y, g = photon_number_dataset(v_max=v_max, shots=10**4, seed=0)
    print(f"    データ {X.shape[0]} 点 × {X.shape[1]} 特徴, "
          f"クラス {n_classes} (chance {1/n_classes:.1%})")
    rows = []
    for mol in ("H2", "HF", "I2"):
        r = evaluate_split(X, y, g, mol, n_classes, seed=0)
        rows.append({"held_out": mol, **r})
        print(f"    test={mol:<4}: QKAN {r['qkan']:.3f}  "
              f"logistic {r['logistic']:.3f}  RF {r['random_forest']:.3f}")
    R["leave_one_molecule_out"] = rows
    for k in ("qkan", "logistic", "random_forest"):
        print(f"    平均 {k:<14}: {np.mean([r[k] for r in rows]):.3f}")

    # --- (ii) ショット数依存 (同一分子内、群分割なし) ---
    print("\n[ii] ショット数依存 (test=I2 固定)")
    shot_rows = []
    for shots in [10**2, 10**3, 10**4, 10**5]:
        Xs, ys, gs = photon_number_dataset(v_max=v_max, shots=shots, seed=1)
        r = evaluate_split(Xs, ys, gs, "I2", n_classes, seed=1)
        shot_rows.append({"shots": shots, **r})
        print(f"    shots={shots:>7}: QKAN {r['qkan']:.3f}  "
              f"logistic {r['logistic']:.3f}  RF {r['random_forest']:.3f}")
    R["shot_dependence"] = shot_rows

    # --- (iii) 旧設計との対比（リークあり同一分子ランダム分割）---
    print("\n[iii] 対照: 旧設計（同一分子・テンプレート複製をランダム分割）")
    Xh, yh, gh = photon_number_dataset(molecules=("H2",), v_max=v_max,
                                       shots=10**4, n_rep=50, seed=2)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(yh))
    ntr = int(0.7 * len(yh))
    tr, te = idx[:ntr], idx[ntr:]
    pred = _train_qkan(Xh[tr], yh[tr], n_classes, seed=2)
    leak = {"qkan": float(np.mean(pred(Xh[te]) == yh[te]))}
    for name, f in _baselines(Xh[tr], yh[tr]).items():
        leak[name] = float(np.mean(f(Xh[te]) == yh[te]))
    R["leaky_within_molecule"] = leak
    print(f"    QKAN {leak['qkan']:.3f}  logistic {leak['logistic']:.3f}  "
          f"RF {leak['random_forest']:.3f}  ← 群分割なしでは容易")

    p = OUT / "qkan_photon_results.json"
    p.write_text(json.dumps(R, indent=2, default=str))
    print(f"\n結果保存: {p}")

    print("\n" + "=" * 74)
    lom = R["leave_one_molecule_out"]
    qk = np.mean([r["qkan"] for r in lom])
    best_cl = max(np.mean([r["logistic"] for r in lom]),
                  np.mean([r["random_forest"] for r in lom]))
    print(f" 分子横断汎化: QKAN {qk:.3f} vs 最良古典 {best_cl:.3f} "
          f"(chance {1/n_classes:.3f})")
    if qk <= best_cl + 0.05:
        print(" → QKAN の優位性は認められない。主張は撤回すべき。")
    else:
        print(" → QKAN が古典を上回る。ただし CI を確認すること。")
    return R


if __name__ == "__main__":
    main()
