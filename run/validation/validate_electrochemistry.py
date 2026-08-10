"""
validate_electrochemistry.py — Electrochemical model validation.

Fig 1  : Polarization curve — model vs Liso et al. at T=60/80deg C (atm) and
         model at operating pressures (p_an=5 bar, p_cath=30 bar).
Table 1: Polarization error metrics (MAE, RMSE, max error vs Liso data).
Table 2: Faraday / species consistency check.
Table 3: Thermal energy-balance closure (first-law residual).
Fig 2  : Thermal sanity check — current step from cold start.

Usage:
  python3 run/validation/validate_electrochemistry.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

VAL_DIR   = Path(__file__).resolve().parent
REPO_ROOT = VAL_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VAL_DIR.parent))

from plot_style import apply_style, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, DARK  # noqa: F401
apply_style()

from pemwe.plant import load_plant
from pemwe.blocks.stack.electrochemistry import electrochem_step
from pemwe.blocks.stack.heat import heat_step
from pemwe.blocks.stack.species import species_step
from pemwe.blocks.recirculation import water_management_step
from pemwe.blocks.thermal import thermal_step
from pemwe.blocks.recirculation import min_water_feed_kg_s

OUTDIR = REPO_ROOT / "results" / "validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Plant constants ───────────────────────────────────────────────────────────
plant     = load_plant(REPO_ROOT / "configs" / "plant_parameters.yaml")
p_an_op   = float(plant["stack"]["p_an_Pa"])
p_cath_op = float(plant["stack"]["p_cath_Pa"])
p_atm     = 101_325.0
N_cells   = int(plant["stack"]["N_cells"])
A_cell    = float(plant["stack"]["A_cell_m2"])
U_tn      = float(plant["thermal"]["U_tn_V"])
C_th      = float(plant["thermal"]["C_th_J_per_K"])
F_const   = float(plant["electrochemistry"]["F_C_per_mol"])
n_e       = float(plant["electrochemistry"]["n_e"])
eta_F     = float(plant["stack"]["eta_F"])
M_H2      = float(plant["fluids"]["molar_masses"]["H2_kg_per_mol"])
M_O2      = float(plant["fluids"]["molar_masses"]["O2_kg_per_mol"])
M_H2O     = float(plant["fluids"]["molar_masses"]["H2O_kg_per_mol"])
HHV       = float(plant["fluids"]["hydrogen"]["HHV_kWh_per_kg"])   # kWh/kg

T_60 = 333.15   # 60 deg C
T_80 = 353.15   # 80 deg C

# ── Shared j sweep ────────────────────────────────────────────────────────────
j_cm2 = np.linspace(0.001, 2.0, 300)
j_m2  = j_cm2 * 1e4


def pol_curve(j_m2_arr, T_K, p_an, p_cath):
    return np.array([
        electrochem_step(j_A_per_m2=j, T_stack_K=T_K,
                         p_an_Pa=p_an, p_cath_Pa=p_cath,
                         plant=plant, V_deg_V=0.0).V_cell_V
        for j in j_m2_arr
    ])


# ═════════════════════════════════════════════════════════════════════════════
# Fig 1 — Polarization curve validation
# ═════════════════════════════════════════════════════════════════════════════
_d60 = np.loadtxt(VAL_DIR / "Experimental_Polarization_Liso_T60.csv",
                  delimiter=",", skiprows=1)
_d80 = np.loadtxt(VAL_DIR / "Experimental_Polarization_Liso_T80.csv",
                  delimiter=",", skiprows=1)
j_exp_60, V_exp_60 = _d60[:, 0], _d60[:, 1]
j_exp_80, V_exp_80 = _d80[:, 0], _d80[:, 1]

V_60_atm = pol_curve(j_m2, T_60, p_atm,   p_atm)
V_80_atm = pol_curve(j_m2, T_80, p_atm,   p_atm)
V_60_op  = pol_curve(j_m2, T_60, p_an_op, p_cath_op)
V_80_op  = pol_curve(j_m2, T_80, p_an_op, p_cath_op)

fig1, ax = plt.subplots(figsize=(7, 5))

ax.plot(j_exp_60, V_exp_60, "o", color=GREEN,  ms=5, zorder=4,
        label=r"$T=60\,^\circ\mathrm{C}$, atm.\ (Liso et al.)")
ax.plot(j_exp_80, V_exp_80, "o", color=ORANGE, ms=5, zorder=4,
        label=r"$T=80\,^\circ\mathrm{C}$, atm.\ (Liso et al.)")
ax.plot(j_cm2, V_60_atm, "-",  color=GREEN,  lw=2.0,
        label=r"$T=60\,^\circ\mathrm{C}$, atm.\ (model)")
ax.plot(j_cm2, V_80_atm, "-",  color=ORANGE, lw=2.0,
        label=r"$T=80\,^\circ\mathrm{C}$, atm.\ (model)")
ax.plot(j_cm2, V_60_op,  "--", color=GREEN,  lw=1.8, alpha=0.55,
        label=r"$T=60\,^\circ\mathrm{C}$, $5/30\,\mathrm{bar}$ (model)")
ax.plot(j_cm2, V_80_op,  "--", color=ORANGE, lw=1.8, alpha=0.55,
        label=r"$T=80\,^\circ\mathrm{C}$, $5/30\,\mathrm{bar}$ (model)")

ax.set_xlabel(r"Current density $j$ [A\,cm$^{-2}$]")
ax.set_ylabel(r"Cell voltage $V_\mathrm{cell}$ [V]")
ax.set_xlim(0, 2.0)
ax.set_ylim(1.2, 2.15)
ax.legend(fontsize=11, loc="upper left")
fig1.tight_layout()
fig1.savefig(OUTDIR / "val_fig1_polarization.pdf", bbox_inches="tight")
fig1.savefig(OUTDIR / "val_fig1_polarization.png", bbox_inches="tight")
plt.close(fig1)
print("  Saved: val_fig1_polarization.pdf/.png")


# ═════════════════════════════════════════════════════════════════════════════
# Table 1 — Polarization error metrics vs Liso et al.
# ═════════════════════════════════════════════════════════════════════════════
def pol_errors(j_exp, V_exp, j_model, V_model):
    """Interpolate model onto experimental j points and return error stats [mV]."""
    V_interp = interp1d(j_model, V_model, kind="linear",
                        fill_value="extrapolate")(j_exp)
    err_mV = (V_interp - V_exp) * 1e3
    return dict(
        n      = len(j_exp),
        mae    = np.mean(np.abs(err_mV)),
        rmse   = np.sqrt(np.mean(err_mV**2)),
        max_e  = np.max(np.abs(err_mV)),
        max_pct= np.max(np.abs(err_mV) / (V_exp * 1e3) * 100),
        bias   = np.mean(err_mV),
    )

e60 = pol_errors(j_exp_60, V_exp_60, j_cm2, V_60_atm)
e80 = pol_errors(j_exp_80, V_exp_80, j_cm2, V_80_atm)

hdr = f"\n{'─'*62}\n  Table 1 — Polarization error (model − Liso et al.) [mV]\n{'─'*62}"
row = "  {:<8}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>7}"
print(hdr)
print(row.format("T [°C]", "N pts", "MAE", "RMSE", "Max |e|", "Max [%]", "Bias"))
print(f"  {'─'*57}")
for T_C, e in [(60, e60), (80, e80)]:
    print(row.format(f"{T_C}", e["n"],
                     f"{e['mae']:.2f}", f"{e['rmse']:.2f}",
                     f"{e['max_e']:.2f}", f"{e['max_pct']:.2f}",
                     f"{e['bias']:+.2f}"))
print(f"{'─'*62}")

with open(OUTDIR / "val_table1_pol_errors.txt", "w") as fh:
    fh.write("Table 1 — Polarization error (model − Liso et al.) [mV]\n")
    fh.write(f"{'T [°C]':<8}  {'N pts':>6}  {'MAE':>8}  {'RMSE':>8}  "
             f"{'Max |e|':>8}  {'Max [%]':>8}  {'Bias':>7}\n")
    for T_C, e in [(60, e60), (80, e80)]:
        fh.write(f"{T_C:<8}  {e['n']:>6}  {e['mae']:>8.2f}  {e['rmse']:>8.2f}  "
                 f"{e['max_e']:>8.2f}  {e['max_pct']:>8.2f}  {e['bias']:>+7.2f}\n")
print("  Saved: val_table1_pol_errors.txt")


# ═════════════════════════════════════════════════════════════════════════════
# Table 2 — Faraday / species consistency
# ═════════════════════════════════════════════════════════════════════════════
# Check that n_dot_H2 from species_step agrees with Faraday's law,
# and that O2 and H2O stoichiometry is internally consistent.

j_check_cm2 = [0.5, 1.0, 1.5, 2.0]

print(f"\n{'─'*88}")
print("  Table 2 — Faraday / species consistency  (T=60°C, op. pressure)")
print(f"{'─'*88}")
hdr2 = ("  {:>5}  {:>12}  {:>8}  {:>12}  {:>8}  {:>12}  {:>8}"
        .format("j", "nH2 exp.", "err H2", "nO2 exp.", "err O2",
                "nH2O exp.", "err H2O"))
print(f"  {'[A/cm²]':>5}  {'[mmol/s]':>12}  {'[%]':>8}  "
      f"{'[mmol/s]':>12}  {'[%]':>8}  {'[mmol/s]':>12}  {'[%]':>8}")
print(f"  {'─'*83}")

rows_faraday = []
for j_val in j_check_cm2:
    j_val_m2 = j_val * 1e4
    ec = electrochem_step(j_A_per_m2=j_val_m2, T_stack_K=T_60,
                          p_an_Pa=p_an_op, p_cath_Pa=p_cath_op,
                          plant=plant, V_deg_V=0.0)
    sp = species_step(I_stack_A=ec.I_stack_A, T_stack_K=T_60,
                      p_an_Pa=p_an_op, p_cath_Pa=p_cath_op,
                      m_dot_w_in_kg_s=1.0, plant=plant)

    n_far     = N_cells * ec.I_stack_A / (n_e * F_const)   # raw Faraday [mol/s]
    # Expected values: H2 reduced by eta_F, O2 and H2O at full Faraday stoichiometry
    n_H2_exp  = eta_F * n_far
    n_O2_exp  = 0.5   * n_far
    n_H2O_exp = 1.0   * n_far
    # Actual values from species_step
    n_H2_sp   = sp.m_dot_H2_dry_kg_s      / M_H2
    n_O2_sp   = sp.m_dot_O2_kg_s          / M_O2
    n_H2O_sp  = sp.m_dot_H2O_consumed_kg_s / M_H2O

    err_H2  = (n_H2_sp  - n_H2_exp)  / n_H2_exp  * 100
    err_O2  = (n_O2_sp  - n_O2_exp)  / n_O2_exp  * 100
    err_H2O = (n_H2O_sp - n_H2O_exp) / n_H2O_exp * 100

    rows_faraday.append((j_val, n_H2_exp*1e3, err_H2,
                         n_O2_exp*1e3, err_O2, n_H2O_exp*1e3, err_H2O))
    print(f"  {j_val:>5.1f}  {n_H2_exp*1e3:>12.5f}  {err_H2:>8.2e}  "
          f"{n_O2_exp*1e3:>12.5f}  {err_O2:>8.2e}  "
          f"{n_H2O_exp*1e3:>12.5f}  {err_H2O:>8.2e}")

print(f"  {'─'*83}")
print(f"  Expected: H2 = eta_F * n_Faraday,  O2 = n_Faraday/2,  H2O = n_Faraday")
print(f"{'─'*88}")

with open(OUTDIR / "val_table2_faraday.txt", "w") as fh:
    fh.write("Table 2 — Faraday / species consistency (T=60°C, op. pressure)\n")
    fh.write(f"{'j':>5}  {'nH2_exp':>12}  {'errH2[%]':>10}  "
             f"{'nO2_exp':>12}  {'errO2[%]':>10}  "
             f"{'nH2O_exp':>12}  {'errH2O[%]':>10}\n")
    fh.write(f"{'[A/cm2]':>5}  {'[mmol/s]':>12}  {'':>10}  "
             f"{'[mmol/s]':>12}  {'':>10}  {'[mmol/s]':>12}  {'':>10}\n")
    for r in rows_faraday:
        fh.write(f"{r[0]:>5.1f}  {r[1]:>12.5f}  {r[2]:>10.2e}  "
                 f"{r[3]:>12.5f}  {r[4]:>10.2e}  "
                 f"{r[5]:>12.5f}  {r[6]:>10.2e}\n")
print("  Saved: val_table2_faraday.txt")


# ═════════════════════════════════════════════════════════════════════════════
# Table 3 — Thermal energy-balance closure
# ═════════════════════════════════════════════════════════════════════════════
# First law: C_th * dT/dt = Q_gen - Q_water - Q_amb
# Verify residual = Q_gen - Q_water - Q_amb - C_th*(T_next-T)/dt ≈ 0
# (validates that thermal_step is thermodynamically consistent)

DT_CHECK = 3600.0 / 60  # matches simulation sub-step duration

print(f"\n{'─'*80}")
print(f"  Table 3 — Thermal energy-balance closure  (T=60°C, m=m_stoich, n={plant['fast_control']['n_substeps']})")
print(f"{'─'*80}")
hdr3 = ("  {:>5}  {:>9}  {:>9}  {:>9}  {:>9}  {:>10}  {:>7}"
        .format("j", "Q_gen", "Q_water", "Q_amb", "C*dT/dt",
                "residual", "err [%]"))
print(hdr3)
print(f"  {'─'*74}")

rows_energy = []
for j_val in j_check_cm2:
    j_val_m2 = j_val * 1e4
    m_cmd = min_water_feed_kg_s(j_val_m2, plant)

    ec = electrochem_step(j_A_per_m2=j_val_m2, T_stack_K=T_60,
                          p_an_Pa=p_an_op, p_cath_Pa=p_cath_op,
                          plant=plant, V_deg_V=0.0)
    q_gen = heat_step(I_stack_A=ec.I_stack_A, V_cell_V=ec.V_cell_V, plant=plant)
    sp = species_step(I_stack_A=ec.I_stack_A, T_stack_K=T_60,
                      p_an_Pa=p_an_op, p_cath_Pa=p_cath_op,
                      m_dot_w_in_kg_s=m_cmd, plant=plant)
    wm = water_management_step(m_dot_total_cmd_kg_s=m_cmd,
                               m_dot_liq_return_kg_s=sp.m_dot_liq_out_kg_s,
                               T_liq_return_K=T_60, plant=plant)
    th = thermal_step(T_stack_K=T_60, Q_gen_W=q_gen,
                      m_dot_w_in_kg_s=wm.m_dot_stack_in_kg_s,
                      T_w_in_K=wm.T_stack_in_K,
                      dt_s=DT_CHECK, plant=plant)

    C_dTdt   = C_th * (th.T_stack_K - T_60) / DT_CHECK
    residual = q_gen - th.Q_water_W - th.Q_amb_W - C_dTdt
    err_pct  = residual / max(abs(q_gen), 1.0) * 100

    rows_energy.append((j_val, q_gen, th.Q_water_W, th.Q_amb_W,
                        C_dTdt, residual, err_pct))
    print(f"  {j_val:>4.1f}  {q_gen/1e3:>8.2f}  {th.Q_water_W/1e3:>8.2f}  "
          f"{th.Q_amb_W/1e3:>8.2f}  {C_dTdt/1e3:>8.2f}  "
          f"{residual:>10.3f}  {err_pct:>6.2e}")

print(f"  {'─'*74}")
print("  Units: heat flows [kW], residual [W].  Target residual: 0 W exactly.")
print(f"{'─'*80}")

with open(OUTDIR / "val_table3_energy_balance.txt", "w") as fh:
    fh.write(f"Table 3 — Thermal energy-balance closure (T=60°C, m=m_stoich, n={plant['fast_control']['n_substeps']})\n")
    fh.write(f"{'j':>5}  {'Q_gen[kW]':>10}  {'Q_water[kW]':>12}  "
             f"{'Q_amb[kW]':>10}  {'C*dT/dt[kW]':>12}  {'resid[W]':>10}  {'err[%]':>8}\n")
    for r in rows_energy:
        fh.write(f"{r[0]:>5.1f}  {r[1]/1e3:>10.3f}  {r[2]/1e3:>12.3f}  "
                 f"{r[3]/1e3:>10.3f}  {r[4]/1e3:>12.3f}  "
                 f"{r[5]:>10.3f}  {r[6]:>8.2e}\n")
print("  Saved: val_table3_energy_balance.txt")


# ═════════════════════════════════════════════════════════════════════════════
# Fig 2 — Thermal sanity check: current step from cold start
# ═════════════════════════════════════════════════════════════════════════════
# Apply a current step at t=0 from T_amb, fixed stoichiometric water flow.
# Validates: thermal time constant, equilibrium temperature, heat-flow balance.

DT_SIM  = 60.0    # [s] sub-step
T_SIM_H = 4.0     # [h] simulation length
N_SIM   = int(T_SIM_H * 3600 / DT_SIM)

J_STEPS = [5000.0, 10000.0, 15000.0]   # 0.5, 1.0, 1.5 A/cm²
LABELS  = [r"$j=0.5\,\mathrm{A\,cm}^{-2}$",
           r"$j=1.0\,\mathrm{A\,cm}^{-2}$",
           r"$j=1.5\,\mathrm{A\,cm}^{-2}$"]
COLORS  = [GREEN, BLUE, RED]

sim_results = {}
for j_step in J_STEPS:
    m_cmd   = min_water_feed_kg_s(j_step, plant)
    T_now   = float(plant["thermal"]["T_amb_K"])
    t_arr, T_arr, Qgen_arr, Qwat_arr, Qamb_arr = [], [], [], [], []

    for i in range(N_SIM):
        ec = electrochem_step(j_A_per_m2=j_step, T_stack_K=T_now,
                              p_an_Pa=p_an_op, p_cath_Pa=p_cath_op,
                              plant=plant, V_deg_V=0.0)
        qg = heat_step(I_stack_A=ec.I_stack_A, V_cell_V=ec.V_cell_V, plant=plant)
        sp = species_step(I_stack_A=ec.I_stack_A, T_stack_K=T_now,
                          p_an_Pa=p_an_op, p_cath_Pa=p_cath_op,
                          m_dot_w_in_kg_s=m_cmd, plant=plant)
        wm = water_management_step(m_dot_total_cmd_kg_s=m_cmd,
                                   m_dot_liq_return_kg_s=sp.m_dot_liq_out_kg_s,
                                   T_liq_return_K=T_now, plant=plant)
        th = thermal_step(T_stack_K=T_now, Q_gen_W=qg,
                          m_dot_w_in_kg_s=wm.m_dot_stack_in_kg_s,
                          T_w_in_K=wm.T_stack_in_K,
                          dt_s=DT_SIM, plant=plant)

        t_arr.append(i * DT_SIM / 60.0)   # minutes
        T_arr.append(T_now - 273.15)
        Qgen_arr.append(qg / 1e3)
        Qwat_arr.append(th.Q_water_W / 1e3)
        Qamb_arr.append(th.Q_amb_W / 1e3)
        T_now = th.T_stack_K

    sim_results[j_step] = dict(t=np.array(t_arr), T=np.array(T_arr),
                                Qgen=np.array(Qgen_arr),
                                Qwat=np.array(Qwat_arr),
                                Qamb=np.array(Qamb_arr))

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: temperature vs time for all j steps
ax_T = axes2[0]
for j_step, label, col in zip(J_STEPS, LABELS, COLORS):
    r = sim_results[j_step]
    ax_T.plot(r["t"] / 60, r["T"], color=col, lw=2.0, label=label)
    T_eq = r["T"][-1]
    ax_T.axhline(T_eq, color=col, ls=":", lw=0.8, alpha=0.6)

ax_T.axhline(plant["thermal"]["T_amb_K"] - 273.15,
             color=GREY, ls="--", lw=1.0, label=r"$T_\mathrm{amb}=25\,^\circ\mathrm{C}$")
ax_T.set_xlabel(r"Time [h]")
ax_T.set_ylabel(r"$T_\mathrm{stack}$ [$^\circ$C]")
ax_T.set_title(r"Temperature response to current step")
ax_T.legend(fontsize=11)

# Right: heat flows vs time for j=1.0 A/cm²
ax_Q = axes2[1]
r1 = sim_results[10000.0]
ax_Q.plot(r1["t"] / 60, r1["Qgen"],                      color=RED,   lw=2.0,
          label=r"$\dot{Q}_\mathrm{gen}$")
ax_Q.plot(r1["t"] / 60, r1["Qwat"],                      color=BLUE,  lw=2.0,
          label=r"$\dot{Q}_\mathrm{water}$")
ax_Q.plot(r1["t"] / 60, r1["Qamb"],                      color=GREEN, lw=1.6, ls="--",
          label=r"$\dot{Q}_\mathrm{amb}$")
ax_Q.plot(r1["t"] / 60, r1["Qgen"] - r1["Qwat"] - r1["Qamb"],
          color=PURPLE, lw=1.4, ls=":", label=r"$\dot{Q}_\mathrm{stored}$ (residual)")

ax_Q.axhline(0, color=GREY, lw=0.6)
ax_Q.set_xlabel(r"Time [h]")
ax_Q.set_ylabel(r"Heat flow [kW]")
ax_Q.set_title(r"Heat flows at $j=1.0\,\mathrm{A\,cm}^{-2}$")
ax_Q.legend(fontsize=11)

fig2.tight_layout()
fig2.savefig(OUTDIR / "val_fig2_thermal_step.pdf", bbox_inches="tight")
fig2.savefig(OUTDIR / "val_fig2_thermal_step.png", bbox_inches="tight")
plt.close(fig2)
print("  Saved: val_fig2_thermal_step.pdf/.png")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  VALIDATION SUMMARY")
print("=" * 62)
ec_ref = electrochem_step(j_A_per_m2=20000, T_stack_K=T_60,
                          p_an_Pa=p_an_op, p_cath_Pa=p_cath_op,
                          plant=plant, V_deg_V=0.0)
ec_atm = electrochem_step(j_A_per_m2=20000, T_stack_K=T_60,
                          p_an_Pa=p_atm, p_cath_Pa=p_atm,
                          plant=plant, V_deg_V=0.0)
print(f"  Pol. MAE 60°C / 80°C:  {e60['mae']:.1f} mV / {e80['mae']:.1f} mV")
print(f"  V_cell j=2.0 T=60 atm: {ec_atm.V_cell_V:.3f} V")
print(f"  V_cell j=2.0 T=60 op:  {ec_ref.V_cell_V:.3f} V")
print(f"  Faraday err (max):      {max(r[3] for r in rows_faraday):.2e} %")
print(f"  Energy resid (max):     {max(abs(r[5]) for r in rows_energy):.3f} W")
T_eq_1 = sim_results[10000.0]["T"][-1]
print(f"  T_eq at j=1.0 A/cm²:   {T_eq_1:.1f} °C  (m=m_stoich)")
print("=" * 62)
print(f"\nAll outputs saved to: {OUTDIR}")
