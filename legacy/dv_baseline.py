#!/usr/bin/env python3
"""
dv_baseline.py — 離散変数(DV, 量子ビット+表面符号)基準（Nature補完②）

「CV が良い」という主張に DV 比較対象を与える。2 つの軸で比較:

  (A) 誤り抑制曲線: CV(GKP, vs スクイズ dB) と DV(表面符号, vs 符号距離 d)。
      ノイズ種が異なるため疑似等価は作らず、各々ネイティブ量で並置し
      「FT オーバーヘッドを何で払うか」を対比 (CV=スクイズ, DV=量子ビット数)。
  (B) 構造的 T/非ガウス資源: 制御なし QPE は CV で制御U=0・非ガウス=0
      (2次H)。DV の標準 QPE は制御-e^{iHt} (Trotter) を要し T カウント大。
      これはノイズモデル非依存の構造的 CV 優位。

依存: NumPy, cv_qec_port
"""

import numpy as np
from pathlib import Path
import json
from cv_qec_port import GKPStabilizerCV, db_to_delta

# 表面符号パラメータ (Fowler et al. 2012, 回路レベル近似)
P_TH = 1.0e-2          # 閾値 (illustrative)
A_SURF = 0.03          # 接頭係数


def surface_logical_error(p_phys, d):
    """回転表面符号の論理誤り率 p_L ≈ A (p/p_th)^((d+1)/2)."""
    return A_SURF * (p_phys / P_TH) ** ((d + 1) / 2.0)


def surface_qubits(d):
    """符号距離 d の物理量子ビット数 (回転表面符号 ≈ 2d²-1)."""
    return 2 * d * d - 1


def gkp_logical_error(sigma_phys, sq_db):
    return GKPStabilizerCV(sq_db).logical_error_rate(sigma_phys)[0]


def main():
    OUT = Path("simulation_results")
    OUT.mkdir(exist_ok=True)
    print("=" * 72)
    print(" DV (量子ビット+表面符号) 基準  vs  CV (GKP)")
    print("=" * 72)
    R = {}

    # --- (A1) DV 表面符号: 論理誤り vs 距離 d (物理誤り p) ---
    print("\n[A1] DV 表面符号: 論理誤り率 vs 符号距離 d")
    dv = []
    for p in [3e-3, 5e-3]:
        for d in [3, 5, 7, 9, 11]:
            pL = surface_logical_error(p, d)
            dv.append({'p_phys': p, 'd': d, 'qubits': surface_qubits(d),
                       'p_logical': pL})
            print(f"  p={p:.0e} d={d:>2d} ({surface_qubits(d):>3d}qubit): "
                  f"p_L={pL:.2e}")
    R['dv_surface'] = dv

    # --- (A2) CV GKP: 論理誤り vs スクイズ dB ---
    print("\n[A2] CV GKP: 論理誤り率 vs スクイズ dB")
    cv = []
    for sigma in [0.20, 0.30]:
        for db in [9, 12, 15, 18, 21]:
            pL = gkp_logical_error(sigma, db)
            cv.append({'sigma_phys': sigma, 'dB': db,
                       'delta': db_to_delta(db), 'p_logical': pL})
            print(f"  σ={sigma:.2f} {db:>2d}dB (Δ={db_to_delta(db):.3f}): "
                  f"p_L={pL:.2e}")
    R['cv_gkp'] = cv

    # --- (B) 資源到達: 目標論理誤り 1e-6 へ ---
    print("\n[B] 目標論理誤り 1e-6 への FT オーバーヘッド")
    target = 1e-6
    # DV: 最小距離 d (p=3e-3)
    p_phys = 3e-3
    d_req = next(d for d in range(3, 51, 2)
                 if surface_logical_error(p_phys, d) <= target)
    print(f"  DV (p={p_phys:.0e}): 距離 d={d_req}, "
          f"物理量子ビット {surface_qubits(d_req)}/論理")
    # CV: 最小 dB を σ ごとに探索 (単一ラウンド GKP は床で頭打ちしうる)
    cv_over = []
    for sigma in [0.10, 0.20]:
        db_req = next((float(db) for db in np.arange(6, 45, 0.5)
                       if gkp_logical_error(sigma, db) <= target), None)
        floor = gkp_logical_error(sigma, 45.0)        # 高dB極限 ≈ 床
        if db_req is not None:
            print(f"  CV (σ={sigma:.2f}): 単一ラウンド {db_req:.1f} dB/モード")
        else:
            print(f"  CV (σ={sigma:.2f}): 単一ラウンドでは到達不可 "
                  f"(床 p_L≈{floor:.1e}) → GKP+表面符号の連接が必要")
        cv_over.append({'sigma': sigma, 'db_req_single_round': db_req,
                        'single_round_floor': float(floor)})
    print("  注: ノイズ種が異なるため直接等価ではなく、FT を払う通貨"
          "(量子ビット数 vs スクイズ量) の対比。単一ラウンド GKP の床は")
    print("      DV の距離スケーリングと同様、連接 (GKP→表面符号) で克服。")
    R['overhead_to_1e-6'] = {
        'target': target, 'dv_distance': d_req,
        'dv_qubits_per_logical': surface_qubits(d_req),
        'dv_p_phys': p_phys, 'cv': cv_over,
    }

    # --- (C) 構造的資源: 制御なし QPE (CV) vs 標準 QPE (DV) ---
    print("\n[C] 構造的資源: 分光タスク (m=10 bit, n_qubit=8, Trotter r=50)")
    m_bits, n_sys, r_trot = 10, 8, 50
    # DV 標準 QPE: 制御-U^{2^k}, k=0..m-1。各制御-e^{iHt} を Trotter 化、
    # 各 Trotter ステップ ~ n_sys 個の制御回転、各回転 ~ 50 T ゲート。
    ctrl_eiHt = sum(2 ** k for k in range(m_bits))      # 総制御発展回数
    t_per_rotation = 50
    dv_T = ctrl_eiHt * r_trot * n_sys * t_per_rotation
    dv_ctrlU = ctrl_eiHt
    # CV 制御なし QPE (2次H): 制御U=0, 非ガウス=0; 時系列点数のみ
    cv_ctrlU, cv_nongauss = 0, 0
    print(f"  DV 標準QPE : 制御-U {dv_ctrlU} 回, T カウント ≈ {dv_T:,}")
    print(f"  CV 制御なし: 制御-U {cv_ctrlU}, 非ガウスゲート {cv_nongauss} "
          f"(2次H, ガウスのみ)")
    print(f"  ⇒ 制御U/マジック状態の構造的削減 (ノイズモデル非依存)")
    R['structural'] = {
        'task': 'spectroscopy m=10,n=8,r=50',
        'dv_controlled_U': dv_ctrlU, 'dv_T_count': dv_T,
        'cv_controlled_U': cv_ctrlU, 'cv_nongaussian_gates': cv_nongauss,
    }

    path = OUT / "dv_baseline_results.json"
    with open(path, 'w') as fh:
        json.dump(R, fh, indent=2, default=str)
    print(f"\n結果保存: {path}")

    print("\n" + "=" * 72)
    print(" 結論")
    print("=" * 72)
    cv10 = cv_over[0]['db_req_single_round']
    print(f" • FT 到達(1e-6): DV={surface_qubits(d_req)}量子ビット/論理 "
          f"(d={d_req}) ⇔ CV(σ=0.10)≈{cv10:.0f}dB/モード — 異種通貨の対比")
    print(f"   σ≳0.2 では単一ラウンド GKP は床で頭打ち→連接が必要(DVと同様)")
    print(f" • 制御なし QPE は CV で 制御U=0・非ガウス=0 (2次H)、"
          f"DV は T≈{dv_T:,} — 構造的 CV 優位")
    print(" • CV の利点はノイズ閾値ではなく『ガウス演算が無料』に由来")
    return R


if __name__ == '__main__':
    main()
