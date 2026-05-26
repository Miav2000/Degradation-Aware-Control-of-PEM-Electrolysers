"""
make_degradation_rate_figure.py
================================
Publication figure: voltage degradation rate vs current density
at representative temperatures, showing the calibrated polynomial
and Arrhenius temperature correction.

Output
------
  results/figures/degradation_rate.png / .pdf
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "run"))


from matplotlib.colors import LinearSegmentedColormap
from plot_style import apply_style, GREEN, ORANGE, DARK

# ── Degradation model parameters (loaded from plant_parameters.yaml) ──────────
from pemwe.plant import load_plant
_plant = load_plant(REPO_ROOT / "configs" / "plant_parameters.yaml")
_deg = _plant["degradation"]
_stk = _plant["stack"]

ALPHA_V = float(_deg["alpha_V_m4_per_A2_h"])
BETA_V  = float(_deg["beta_V_m2_per_A_h"])
GAMMA_V = float(_deg["gamma_V_per_h"])
EA_EFF  = float(_deg["Ea_eff_J_per_mol"])
T_REF_K = float(_deg["T_ref_K"])
R_GAS   = 8.314462618  # [J/(mol·K)]

# Calibration anchor — rate at j_max, T_ref (read back from model for consistency)
J_CAL_Acm2 = float(_stk["j_max_A_per_m2"]) / 1e4   # 2.0 A/cm²
T_CAL_C    = T_REF_K - 273.15                        # 60 °C
RATE_CAL_uVh = (ALPHA_V * (J_CAL_Acm2*1e4)**2
                + BETA_V * (J_CAL_Acm2*1e4)
                + GAMMA_V) * 1e6                     # µV/h at T_ref

# Operating range
J_MIN_Acm2 = float(_stk["j_min_A_per_m2"]) / 1e4
J_MAX_Acm2 = float(_stk["j_max_A_per_m2"]) / 1e4

OUTDIR = REPO_ROOT / "results" / "figures"


def arrhenius(T_C: float) -> float:
    T_K = T_C + 273.15
    return math.exp(-EA_EFF / R_GAS * (1.0 / T_K - 1.0 / T_REF_K))


def rate_uV_per_h(j_Acm2: np.ndarray, T_C: float) -> np.ndarray:
    """Degradation rate [µV/h] for array of j [A/cm²] at temperature T_C."""
    j_m2 = j_Acm2 * 1e4          # A/cm² → A/m²
    poly  = ALPHA_V * j_m2**2 + BETA_V * j_m2 + GAMMA_V
    return poly * arrhenius(T_C) * 1e6   # V/h → µV/h


def main() -> None:
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    j = np.linspace(0.0, J_MAX_Acm2, 400)

    temperatures = [50, 60, 70, 80, 90]
    _cmap  = LinearSegmentedColormap.from_list(
        "pemwe_T", [(0.0, GREEN), (0.30, "#FFD9A8"), (0.65, "#FF9A50"), (1.0, ORANGE)], N=256
    )
    _norm  = plt.Normalize(50, 90)
    colors = [_cmap(_norm(T)) for T in temperatures]
    linestyles = ["-", "-", "-", "-", "-"]
    linewidths = [1.8, 1.8, 1.8, 1.8, 1.8]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for T_C, color, ls, lw in zip(temperatures, colors, linestyles, linewidths):
        rate = rate_uV_per_h(j, T_C)
        label = (
            rf"$T = {T_C}\,^\circ$C  "
            rf"($f_T = {arrhenius(T_C):.2f}$)"
        )
        ax.plot(j, rate, color=color, ls=ls, lw=lw, label=label)

    # Calibration anchor
    ax.scatter(
        [J_CAL_Acm2], [RATE_CAL_uVh],
        color=DARK, s=60, zorder=5,
        label=rf"Calibration: {RATE_CAL_uVh:.1f}\,$\mu$V/h at $j={J_CAL_Acm2}$\,A/cm$^2$, $T={T_CAL_C:.0f}\,^\circ$C",
    )


    ax.set_xlabel(r"Current density $j$ [A\,cm$^{-2}$]")
    ax.set_ylabel(r"Degradation rate $\dot{V}_\mathrm{deg}$ [$\mu$V\,h$^{-1}$]")
    ax.set_xlim(0.0, J_MAX_Acm2 * 1.05)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8.5, loc="upper left")

    fig.tight_layout()

    png = OUTDIR / "degradation_rate.png"
    pdf = OUTDIR / "degradation_rate.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
