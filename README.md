# PEMWE System Model

Simulation and comparison of four supervisory control strategies for a PEM water electrolyzer (PEMWE) driven by variable renewable power (wind). The model evaluates hydrogen production cost (LCOH), stack degradation, and stack lifetime across a 1- or 5-year horizon.

**The four controllers compared:**

| Name | Abbreviation | Objective |
|---|---|---|
| Load-following | LF | Track available power |
| Price-aware | PA | Minimise electricity cost |
| Degradation-aware | DA | Minimise electricity cost + degradation penalty |
| Lifetime-aware | LA | DA with accumulated degradation feedback — degradation weight grows progressively as the stack ages |

The plant physics model is identical for all four controllers. Only the supervisory objective changes.

---

## Installation

You need **Python 3.10 or newer**. If you are unsure which version you have, run `python3 --version` in a terminal.

**Step 1 — Clone the repository**
```bash
git clone <repo-url>
cd PEMWE_system_model
```

**Step 2 — Create a virtual environment** (keeps dependencies isolated from your system)
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

**Step 3 — Install the model and all dependencies**
```bash
pip install -e .
```

All required packages (numpy, scipy, pandas, matplotlib, cobyqa, pyyaml) are listed in `pyproject.toml` and will be installed automatically.

---

## How to run a case

All simulations are launched from the **repository root folder** (`PEMWE_system_model/`). The easiest entry point is the Makefile.

**See all available runs:**
```bash
make help
```

**Compare all four controllers (wind power + spot price) — sequential:**
```bash
make compare_wind
```
This runs all four controllers one after another and saves results and figures to `results/comparison_wind_spot_dk1/`.

**Run all four controllers in parallel (faster):**

`nohup` means "no hang up" — it keeps the process running in the background even if you close the terminal. Each controller gets its own log file so you can monitor progress independently.

```bash
make run_all
```

Or launch each controller individually in parallel:
```bash
nohup python3 run/single_simulation.py --plant configs/plant_parameters.yaml \
    --controller configs/controllers/load_following.yaml \
    --power-profile configs/power_profiles/wind.yaml \
    --price-profile configs/price_profiles/spot_dk1.yaml \
    > /tmp/lf.log 2>&1 &

nohup python3 run/single_simulation.py --plant configs/plant_parameters.yaml \
    --controller configs/controllers/lifetime_aware.yaml \
    --power-profile configs/power_profiles/wind.yaml \
    --price-profile configs/price_profiles/spot_dk1.yaml \
    > /tmp/la.log 2>&1 &
```

Monitor progress while running:
```bash
tail -f /tmp/lf.log
tail -f /tmp/la.log
```

Once all four CSVs exist, generate the comparison figures:
```bash
make plot_comparison_wind
```

**Run a single controller (sequential, foreground):**
```bash
make load_following_wind
make price_aware_wind
make degradation_aware_wind
make lifetime_aware_wind
```

---

## Overview of all main runs

### 1-year simulations

| Make target | What it does |
|---|---|
| `make compare_wind` | All 4 controllers, wind + spot price (main comparison) |
| `make compare_sine` | All 4 controllers, sinusoidal power + constant price |
| `make load_following_wind` | Single run: load-following only |
| `make lifetime_aware_wind` | Single run: lifetime-aware only |

### 5-year lifetime simulations

Requires the 1-year runs to have completed first.

```bash
make lifetime_5yr          # run all 4 controllers for 5 years (parallel background jobs)
make lifetime_5yr_plots    # replot from existing 5yr CSVs
```

Monitor progress:
```bash
tail -f /tmp/lt5yr_load_following.log
tail -f /tmp/lt5yr_lifetime_aware.log
```

**Changing the simulation duration:** The horizon is controlled by a single line in `run/sensitivity/lifetime_5yr.py`:
```python
N_YEARS = 5   # <- change to any number of years, e.g. 10
```
The 1-year wind and price data is automatically tiled to fill the full requested duration. No other changes are needed.

### Sensitivity studies

| Make target | What it does |
|---|---|
| `make sensitivity_fixed_T_5yr` | Fixed operating temperature (7 cases, 55–70 °C) |
| `make sensitivity_substeps` | Effect of optimizer substep resolution |
| `make sensitivity_tornado` | Tornado chart: LCOH sensitivity to key parameters |
| `make sensitivity_temp_decomp` | Decompose degradation into current-density vs temperature contribution |

### Run everything at once

```bash
make run_all
```
Launches 16 parallel background processes (4 × 1yr + 4 × 5yr + 8 substep tests). Sensitivity studies must be run separately afterwards once the 1-year CSVs exist.

---

## Repository structure

```
PEMWE_system_model/
│
├── configs/
│   ├── plant_parameters.yaml          # Stack, thermal, auxiliary, degradation parameters
│   ├── controllers/
│   │   ├── load_following.yaml
│   │   ├── price_aware.yaml
│   │   ├── degradation_aware.yaml
│   │   └── lifetime_aware.yaml
│   ├── power_profiles/                # Wind profile (Energinet DK1 offshore wind data)
│   └── price_profiles/                # Spot price profile (DK1)
│
├── pemwe/
│   ├── blocks/                        # Physics: electrochemistry, thermal, species, auxiliaries
│   ├── control/                       # Supervisory optimizer, fast PLC layer, controller policies
│   └── degradation/                   # Voltage degradation model
│
├── run/
│   ├── single_simulation.py           # Run one controller
│   ├── comparison_simulation.py       # Run and compare all four controllers
│   ├── make_plots.py                  # Generate comparison figures from CSVs
│   ├── plot_style.py                  # Shared figure style
│   ├── sensitivity/                   # All sensitivity and multi-year studies
│   └── validation/                    # Electrochemistry and thermal validation scripts
│
├── results/                           # Auto-generated (created on first run)
│   └── <run_name>/                    # One folder per run: CSV + all figures
│
├── pyproject.toml                     # Package metadata and dependencies
├── Makefile                           # All run commands
└── LICENSE                            # CC BY-NC 4.0
```

---

## Running directly from Python (without Make)

If you prefer not to use Make, the equivalent direct command for a full comparison is:

```bash
python3 run/comparison_simulation.py \
    --plant configs/plant_parameters.yaml \
    --power-profile configs/power_profiles/wind.yaml \
    --price-profile configs/price_profiles/spot_dk1.yaml \
    --load-following-controller    configs/controllers/load_following.yaml \
    --price-aware-controller       configs/controllers/price_aware.yaml \
    --degradation-aware-controller configs/controllers/degradation_aware.yaml \
    --lifetime-aware-controller    configs/controllers/lifetime_aware.yaml
```

---

## License

This code is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.
Free to use and adapt for non-commercial purposes with attribution. See `LICENSE` for details.
