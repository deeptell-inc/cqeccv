#!/usr/bin/env python3
"""make_figs.py — 論文用の図を実データ(JSON)から生成."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
# Embed Type-42 (TrueType) with Unicode mapping so Greek/math glyphs
# render in all PDF viewers (avoids "tofu" boxes) and meets journal
# submission requirements.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
import matplotlib.pyplot as plt

SR = Path(__file__).resolve().parent.parent / "simulation_results"
OUT = Path(__file__).resolve().parent

nc = json.loads((SR / "noise_comparison_results.json").read_text())
nm = json.loads((SR / "normalized_metric_results.json").read_text())

# ASCII labels (avoid missing-glyph boxes in PDF)
LAB = {"QPE(制御なし)": "QPE (control-free)", "qDRIFT": "qDRIFT",
       "QKAN": "QKAN", "Regev": "Regev"}

# --- Fig 1: squeeze sweep, improvement vs dB with 95% CI band ---
ss = nc["squeeze_sweep"]
plt.figure(figsize=(3.4, 2.6))
for name in ss:
    u = ss[name]["uncorrected"]["mean"]
    uci = ss[name]["uncorrected"]["ci95"]
    dbs = sorted(ss[name]["by_dB"], key=lambda x: x["dB"])
    x = [e["dB"] for e in dbs]
    imp = [u / e["mean"] for e in dbs]
    # 改善率の伝播 CI (相対誤差合成)
    band = []
    for e in dbs:
        ru = uci / u if u else 0
        rc = e["ci95"] / e["mean"] if e["mean"] else 0
        band.append((u / e["mean"]) * np.sqrt(ru**2 + rc**2))
    imp = np.array(imp); band = np.array(band)
    plt.plot(x, imp, marker="o", ms=3, label=LAB.get(name, name))
    plt.fill_between(x, imp - band, imp + band, alpha=0.2)
plt.axhline(1.0, color="k", lw=0.8, ls="--")
plt.xlabel("squeezing (dB)"); plt.ylabel("GKP improvement  $u/c$")
plt.legend(fontsize=6); plt.tight_layout()
plt.savefig(OUT / "fig_squeeze_sweep.pdf"); plt.close()

# --- Fig 2: normalized infidelity, uncorrected vs 18 dB ---
names = list(nm.keys())
Iu = [nm[n]["I_uncorrected"] for n in names]
I18 = [nm[n]["I_18dB"] for n in names]
y = np.arange(len(names))
plt.figure(figsize=(3.4, 2.6))
plt.barh(y - 0.18, Iu, 0.36, label="uncorrected")
plt.barh(y + 0.18, I18, 0.36, label="18 dB GKP")
plt.yticks(y, [LAB.get(n, n) for n in names], fontsize=6)
plt.xlabel("normalized infidelity  $I$")
plt.legend(fontsize=6); plt.tight_layout()
plt.savefig(OUT / "fig_normalized_rank.pdf"); plt.close()

# --- Fig 3: 2D phase diagram of GKP improvement (4 panels) ---
pd_path = SR / "noise_phase_diagram_results.json"
if pd_path.exists():
    pd = json.loads(pd_path.read_text())["phase_diagram"]
    sg, lg = pd["sigma_grid"], pd["loss_grid"]
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 5.0))
    for ax, name in zip(axes.flat, pd["improvement"]):
        M = np.array(pd["improvement"][name])
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=0.4, vmax=1.6,
                       extent=[lg[0], lg[-1], sg[0], sg[-1]])
        # floor-crossing contour (improvement = 1)
        try:
            ax.contour(np.linspace(lg[0], lg[-1], len(lg)),
                       np.linspace(sg[0], sg[-1], len(sg)),
                       M, levels=[1.0], colors="k", linewidths=1.0)
        except Exception:
            pass
        ax.set_title(LAB.get(name, name), fontsize=7, pad=4)
        ax.set_xlabel("loss $1-\\eta$", fontsize=6, labelpad=2)
        ax.set_ylabel("$\\sigma_d$", fontsize=6, labelpad=2)
        ax.tick_params(labelsize=5)
    fig.subplots_adjust(wspace=0.45, hspace=0.55,
                        left=0.10, right=0.86, top=0.94, bottom=0.10)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.04,
                 label="GKP improvement (12 dB)")
    fig.savefig(OUT / "fig_phase_diagram.pdf"); plt.close(fig)
print("figures written to", OUT)

# --- Fig 4: Diatomic Morse spectra (H₂, HF, I₂) via CV-QPE ---
mol_path = SR / "physics_h2_morse_results.json"
if mol_path.exists():
    md = json.loads(mol_path.read_text())
    mols = md.get("molecules", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.8))
    colors = {"H2": "C0", "HF": "C1", "I2": "C2"}
    markers = {"H2": "o", "HF": "s", "I2": "^"}
    # left: recovered vs analytic
    for key, out in mols.items():
        mol = key.split("_")[0]
        rows = out["recovered"]
        if not rows:
            continue
        eans = [r["analytic"] for r in rows]
        erec = [r["recovered"] for r in rows]
        ax1.plot(eans, erec, markers[mol], color=colors[mol], ms=5,
                 label=f"{mol} ({key.split('_')[1]})", alpha=0.85)
    ax1.plot([0, 7], [0, 7], "k:", lw=0.8, label="ideal")
    ax1.set_xlabel("$E_v/\\omega_e$ (analytic)", fontsize=8)
    ax1.set_ylabel("$E_v/\\omega_e$ (CV-QPE)", fontsize=8)
    ax1.legend(fontsize=6, loc="upper left"); ax1.tick_params(labelsize=6)
    # right: error vs v (log scale)
    for key, out in mols.items():
        mol = key.split("_")[0]
        rows = out["recovered"]
        if not rows:
            continue
        ax2.plot([r["v"] for r in rows], [r["error_cm"] for r in rows],
                 markers[mol], color=colors[mol], ms=5,
                 label=f"{mol}", alpha=0.85)
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8,
                label="1 cm$^{-1}$")
    ax2.set_xlabel("$v$", fontsize=8)
    ax2.set_ylabel("$|\\Delta E|$ (cm$^{-1}$)", fontsize=8)
    ax2.set_yscale("log")
    ax2.legend(fontsize=6); ax2.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(OUT / "fig_h2_morse.pdf"); plt.close()
print("molecule figure written")
