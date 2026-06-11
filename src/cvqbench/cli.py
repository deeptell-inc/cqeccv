"""Command-line dispatcher for cvqbench reproduce targets.

Usage:
    cvqbench <target> [...]
    python -m cvqbench <target>

Targets map one-to-one to the module ``main()`` functions. Outputs
(JSON, and figures when applicable) are written under ``simulation_results/``
relative to the current working directory.
"""
import argparse
import importlib
import sys

TARGETS = {
    "qec":          "cv_qec_port",
    "qpe":          "qpe_cv",
    "qdrift":       "qdrift_cv",
    "qkan":         "qkan_cv",
    "qkan-deep":    "qkan_deep",
    "qpe-nongauss": "qpe_nongaussian",
    "regev":        "regev_cv",
    "noise":        "noise_comparison",
    "normalized":   "normalized_metric",
    "gkp-validate": "gkp_validate",
    "dv-baseline":  "dv_baseline",
    "phase":        "noise_phase_diagram",
    "h2-morse":     "physics_h2_morse",
    "h2-qkan":      "physics_h2_qkan",
}

ALL_ORDER = ["qec", "qpe", "qdrift", "qkan", "regev", "noise",
             "normalized", "gkp-validate", "dv-baseline", "phase",
             "qpe-nongauss", "qkan-deep", "h2-morse", "h2-qkan"]


def _run(target: str) -> int:
    mod = importlib.import_module(f"cvqbench.{TARGETS[target]}")
    if not hasattr(mod, "main"):
        print(f"module {TARGETS[target]} has no main()", file=sys.stderr)
        return 2
    mod.main()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="cvqbench",
        description="Reproduce cvqbench benchmarks (writes simulation_results/).")
    ap.add_argument("target", choices=list(TARGETS) + ["all"],
                    help="benchmark to run, or 'all'")
    a = ap.parse_args(argv)
    if a.target == "all":
        rc = 0
        for t in ALL_ORDER:
            print(f"\n########## cvqbench: {t} ##########")
            try:
                rc |= _run(t)
            except Exception as e:  # keep going, report at end
                print(f"[{t}] FAILED: {e}", file=sys.stderr)
                rc |= 1
        return rc
    return _run(a.target)


if __name__ == "__main__":
    sys.exit(main())
