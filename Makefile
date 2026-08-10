# Makefile — PEMWE model runs
#
# Results are auto-organized: results/<controller>_<power>_<price>/
# Each folder contains the CSV and all figures for that run.

PYTHON := python3
PLANT := configs/plant_parameters.yaml

# Clear bytecode cache before every run to avoid stale .pyc issues
CLEAR_CACHE := @find pemwe/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Controllers
CTRL_LOAD_FOLLOWING   := configs/controllers/load_following.yaml
CTRL_PRICE_AWARE      := configs/controllers/price_aware.yaml
CTRL_DEGRADATION_AWARE := configs/controllers/degradation_aware.yaml
CTRL_LIFETIME_AWARE   := configs/controllers/lifetime_aware.yaml
# Power profiles
POWER_WIND := configs/power_profiles/wind.yaml
POWER_SINE := configs/power_profiles/sine.yaml

# Price profiles
PRICE_SPOT := configs/price_profiles/spot_dk1.yaml
PRICE_CONST := configs/price_profiles/constant_5ct.yaml

.PHONY: help clean \
	load_following_wind degradation_aware_wind price_aware_wind lifetime_aware_wind \
	load_following_sine degradation_aware_sine \
	compare_wind compare_sine \
	plot_comparison_wind plot_comparison_sine plot_comparison_5yr lcoh_composition_5yr \
	nday5 nday5_plots \
	lifetime_5yr lifetime_5yr_plots \
	sensitivity_fixed_T sensitivity_fixed_T_5yr sensitivity_fixed_T_5yr_plots \
	sensitivity_temp_decomp \
	sensitivity_optimizer sensitivity_optimizer_plots sensitivity_optimizer_conv \
	sensitivity_substeps sensitivity_substeps_plots \
	substep_test \
	run_all

help:
	@echo "PEMWE_system_model — Make targets"
	@echo ""
	@echo "Single runs (CSV + figures in results/<name>/):"
	@echo "  make load_following_wind     Load-following + wind + spot price"
	@echo "  make degradation_aware_wind  Degradation-aware + wind + spot price"
	@echo "  make price_aware_wind        Price-aware + wind + spot price"
	@echo "  make lifetime_aware_wind     Lifetime-aware + wind + spot price"
	@echo "  make load_following_sine     Load-following + sine + constant price"
	@echo "  make degradation_aware_sine  Degradation-aware + sine + constant price"
	@echo ""
	@echo "Comparisons (runs all + comparison figures):"
	@echo "  make compare_wind      All four controllers (wind + spot price)"
	@echo "  make compare_sine      Load-following vs degradation-aware (sine + constant price)"
	@echo ""
	@echo "Re-plot comparison only (from existing CSVs):"
	@echo "  make plot_comparison_wind"
	@echo "  make plot_comparison_sine"
	@echo "  make plot_comparison_5yr"
	@echo ""
	@echo "Multi-year / sensitivity (require CSVs from 1yr runs):"
	@echo "  make lifetime_5yr            5-year lifetime runs (4 controllers, parallel)"
	@echo "  make lifetime_5yr_plots      Replot 5yr comparison from existing CSVs"
	@echo "  make sensitivity_fixed_T     Fixed-T Pareto frontier (needs 1yr CSVs)"
	@echo "  make sensitivity_temp_decomp j vs T degradation decomposition"
	@echo ""
	@echo "Substep sensitivity test:"
	@echo "  make substep_test      1yr runs at n_substeps=50 and 60 (8 parallel)"
	@echo ""
	@echo "Launch everything at once (16 parallel processes):"
	@echo "  make run_all           1yr + 5yr + substep (sensitivity runs separately)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean             Remove results/"

# --- Single runs ---

load_following_wind:
	$(CLEAR_CACHE)
	@rm -rf results/load_following_wind_spot_dk1
	$(PYTHON) run/single_simulation.py \
		--plant $(PLANT) \
		--controller $(CTRL_LOAD_FOLLOWING) \
		--power-profile $(POWER_WIND) \
		--price-profile $(PRICE_SPOT)

degradation_aware_wind:
	$(CLEAR_CACHE)
	@rm -rf results/degradation_aware_wind_spot_dk1
	$(PYTHON) run/single_simulation.py \
		--plant $(PLANT) \
		--controller $(CTRL_DEGRADATION_AWARE) \
		--power-profile $(POWER_WIND) \
		--price-profile $(PRICE_SPOT)

price_aware_wind:
	$(CLEAR_CACHE)
	@rm -rf results/price_aware_wind_spot_dk1
	$(PYTHON) run/single_simulation.py \
		--plant $(PLANT) \
		--controller $(CTRL_PRICE_AWARE) \
		--power-profile $(POWER_WIND) \
		--price-profile $(PRICE_SPOT)

lifetime_aware_wind:
	$(CLEAR_CACHE)
	@rm -rf results/lifetime_aware_wind_spot_dk1
	$(PYTHON) run/single_simulation.py \
		--plant $(PLANT) \
		--controller $(CTRL_LIFETIME_AWARE) \
		--power-profile $(POWER_WIND) \
		--price-profile $(PRICE_SPOT)

load_following_sine:
	$(CLEAR_CACHE)
	@rm -rf results/load_following_sine_constant_5ct
	$(PYTHON) run/single_simulation.py \
		--plant $(PLANT) \
		--controller $(CTRL_LOAD_FOLLOWING) \
		--power-profile $(POWER_SINE) \
		--price-profile $(PRICE_CONST)

degradation_aware_sine:
	@rm -rf results/degradation_aware_sine_constant_5ct
	$(PYTHON) run/single_simulation.py \
		--plant $(PLANT) \
		--controller $(CTRL_DEGRADATION_AWARE) \
		--power-profile $(POWER_SINE) \
		--price-profile $(PRICE_CONST)

# --- Comparisons ---

compare_wind:
	$(CLEAR_CACHE)
	@rm -rf results/load_following_wind_spot_dk1 results/price_aware_wind_spot_dk1 results/degradation_aware_wind_spot_dk1 results/lifetime_aware_wind_spot_dk1 results/comparison_wind_spot_dk1
	$(PYTHON) run/comparison_simulation.py \
		--plant $(PLANT) \
		--power-profile $(POWER_WIND) \
		--price-profile $(PRICE_SPOT) \
		--load-following-controller    $(CTRL_LOAD_FOLLOWING) \
		--price-aware-controller       $(CTRL_PRICE_AWARE) \
		--degradation-aware-controller $(CTRL_DEGRADATION_AWARE) \
		--lifetime-aware-controller    $(CTRL_LIFETIME_AWARE)

compare_sine:
	$(CLEAR_CACHE)
	@rm -rf results/load_following_sine_constant_5ct results/price_aware_sine_constant_5ct results/degradation_aware_sine_constant_5ct results/lifetime_aware_sine_constant_5ct results/comparison_sine_constant_5ct
	$(PYTHON) run/comparison_simulation.py \
		--plant $(PLANT) \
		--power-profile $(POWER_SINE) \
		--price-profile $(PRICE_CONST) \
		--load-following-controller    $(CTRL_LOAD_FOLLOWING) \
		--price-aware-controller       $(CTRL_PRICE_AWARE) \
		--degradation-aware-controller $(CTRL_DEGRADATION_AWARE) \
		--lifetime-aware-controller    $(CTRL_LIFETIME_AWARE)

# --- Re-plot comparison (skip simulation) ---

plot_comparison_wind:
	@mkdir -p results/comparison_wind_spot_dk1
	$(PYTHON) run/make_plots.py \
		--load-following    results/load_following_wind_spot_dk1/load_following_wind_spot_dk1.csv \
		--price-aware       results/price_aware_wind_spot_dk1/price_aware_wind_spot_dk1.csv \
		--degradation-aware results/degradation_aware_wind_spot_dk1/degradation_aware_wind_spot_dk1.csv \
		--lifetime-aware    results/lifetime_aware_wind_spot_dk1/lifetime_aware_wind_spot_dk1.csv \
		--name              comparison_wind_spot_dk1 \
		--outdir            results/comparison_wind_spot_dk1

plot_comparison_sine:
	@mkdir -p results/comparison_sine_constant_5ct
	$(PYTHON) run/make_plots.py \
		--load-following    results/load_following_sine_constant_5ct/load_following_sine_constant_5ct.csv \
		--degradation-aware results/degradation_aware_sine_constant_5ct/degradation_aware_sine_constant_5ct.csv \
		--name              comparison_sine_constant_5ct \
		--outdir            results/comparison_sine_constant_5ct

plot_comparison_5yr:
	@mkdir -p results/lifetime_5yr/comparison_5yr
	$(PYTHON) run/make_plots.py \
		--load-following    results/lifetime_5yr/load_following_5yr/load_following_5yr.csv \
		--price-aware       results/lifetime_5yr/price_aware_5yr/price_aware_5yr.csv \
		--degradation-aware results/lifetime_5yr/degradation_aware_5yr/degradation_aware_5yr.csv \
		--lifetime-aware    results/lifetime_5yr/lifetime_aware_5yr/lifetime_aware_5yr.csv \
		--name              comparison_5yr \
		--outdir            results/lifetime_5yr/comparison_5yr

lcoh_composition_5yr:
	@mkdir -p results/lifetime_5yr/comparison_5yr
	$(PYTHON) run/make_plots.py \
		--load-following    results/lifetime_5yr/load_following_5yr/load_following_5yr.csv \
		--price-aware       results/lifetime_5yr/price_aware_5yr/price_aware_5yr.csv \
		--degradation-aware results/lifetime_5yr/degradation_aware_5yr/degradation_aware_5yr.csv \
		--lifetime-aware    results/lifetime_5yr/lifetime_aware_5yr/lifetime_aware_5yr.csv \
		--outdir            results/lifetime_5yr/comparison_5yr \
		--lcoh-composition

# --- N-day validation (parallel, nohup) ---

nday5:
	$(CLEAR_CACHE)
	@rm -rf results/nday_comparison/5day
	@mkdir -p results/nday_comparison/5day
	@for c in load_following price_aware degradation_aware lifetime_aware; do \
		nohup $(PYTHON) run/sensitivity/run_nday_comparison.py --days 5 --controller $$c \
			> /tmp/nday5_$$c.log 2>&1 & \
	done
	@echo "Launched all 4 controllers in parallel."
	@echo "Monitor: tail -f /tmp/nday5_load_following.log  (or price_aware / degradation_aware / lifetime_aware)"
	@echo "When all 4 show 'saved ...csv': make nday5_plots"

nday5_plots:
	$(PYTHON) run/sensitivity/run_nday_comparison.py --days 5 --plots-only

# --- 5-year lifetime runs (parallel, nohup) ---

lifetime_5yr:
	$(CLEAR_CACHE)
	@rm -rf results/lifetime_5yr
	@mkdir -p results/lifetime_5yr
	@for c in load_following price_aware degradation_aware lifetime_aware; do \
		nohup bash -c "echo 'START '$$c': '$$(date -u +%Y-%m-%dT%H:%M:%S); \
			time $(PYTHON) run/sensitivity/lifetime_5yr.py --controller $$c; \
			echo 'END '$$c': '$$(date -u +%Y-%m-%dT%H:%M:%S)" \
			> /tmp/lt5yr_$$c.log 2>&1 & \
		echo "  started 5yr $$c (PID $$!)"; \
	done
	@echo "Launched all 4 lifetime controllers in parallel."
	@echo "Monitor: tail -f /tmp/lt5yr_load_following.log  (or price_aware / degradation_aware / lifetime_aware)"
	@echo "When all 4 show 'saved ...csv': make lifetime_5yr_plots"

lifetime_5yr_plots: plot_comparison_5yr
	$(PYTHON) run/sensitivity/lifetime_5yr.py --plots-only

# --- Sensitivity studies (require 1yr CSVs to exist first) ---

sensitivity_fixed_T:
	$(CLEAR_CACHE)
	$(PYTHON) run/sensitivity/sensitivity_fixed_T_pareto.py --run

sensitivity_fixed_T_5yr:
	$(CLEAR_CACHE)
	@rm -rf results/sensitivity_fixed_T_5yr
	@echo "Launching 8 fixed-T 5yr simulations in parallel (nohup)..."
	@for T in 56 58 60 62 64 66 68 70; do \
		nohup bash -c "echo 'START fixedT_'$$T'C: '$$(date -u +%Y-%m-%dT%H:%M:%S); \
			time $(PYTHON) run/sensitivity/sensitivity_fixed_T_pareto.py --run --only-T $$T; \
			echo 'END fixedT_'$$T'C: '$$(date -u +%Y-%m-%dT%H:%M:%S)" \
			> /tmp/fixedT_$${T}C_5yr.log 2>&1 & \
		echo "  started fixed-T $$T°C (PID $$!)"; \
	done
	@echo "All launched. Logs: /tmp/fixedT_*C_5yr.log"
	@echo "Plot when done: make sensitivity_fixed_T_5yr_plots"

sensitivity_fixed_T_5yr_plots:
	$(PYTHON) run/sensitivity/sensitivity_fixed_T_pareto.py --plots-only

sensitivity_optimizer:
	$(CLEAR_CACHE)
	nohup $(PYTHON) run/sensitivity/sensitivity_optimizer_comparison.py --run \
		> /tmp/optimizer_benchmark.log 2>&1 &
	@echo "Benchmark launched. Log: /tmp/optimizer_benchmark.log"
	@echo "Plot when done: make sensitivity_optimizer_plots"

sensitivity_optimizer_plots:
	$(PYTHON) run/sensitivity/sensitivity_optimizer_comparison.py --plots-only

sensitivity_optimizer_conv:
	$(PYTHON) run/sensitivity/sensitivity_optimizer_comparison.py --run --conv-only

sensitivity_temp_decomp:
	$(CLEAR_CACHE)
	$(PYTHON) run/sensitivity/sensitivity_temp_vs_j_decomposition.py

sensitivity_substeps:
	$(CLEAR_CACHE)
	time $(PYTHON) run/sensitivity/sensitivity_substeps.py

sensitivity_substeps_plots:
	$(PYTHON) run/sensitivity/sensitivity_substeps.py --plots-only

sensitivity_tornado:
	$(CLEAR_CACHE)
	@rm -rf results/sensitivity_tornado
	@mkdir -p results/sensitivity_tornado
	nohup $(PYTHON) run/sensitivity/sensitivity_tornado.py \
		> results/sensitivity_tornado/run.log 2>&1 &
	@echo "Tornado sensitivity launched. Log: results/sensitivity_tornado/run.log"

# --- Substep sensitivity test (parallel, nohup) ---
# Tests n_substeps=50 and n_substeps=60 for all 4 controllers (8 runs total).
# Results land in results/substep_test/sub<N>/<controller>/

substep_test:
	$(CLEAR_CACHE)
	@rm -rf results/substep_test
	@mkdir -p results/substep_test
	@for s in 50 60; do \
		for c in load_following price_aware degradation_aware lifetime_aware; do \
			nohup bash -c "echo 'START '$${c}'_sub'$$s': '$$(date -u +%Y-%m-%dT%H:%M:%S); \
				time $(PYTHON) run/single_simulation.py \
					--plant $(PLANT) \
					--controller configs/controllers/$$c.yaml \
					--power-profile $(POWER_WIND) \
					--price-profile $(PRICE_SPOT) \
					--n-substeps $$s \
					--name $${c}_wind_spot_dk1_sub$$s \
					--outdir results/substep_test/sub$$s/$$c; \
				echo 'END '$${c}'_sub'$$s': '$$(date -u +%Y-%m-%dT%H:%M:%S)" \
				> /tmp/substep_$${c}_sub$$s.log 2>&1 & \
			echo "  started $$c sub$$s (PID $$!)"; \
		done; \
	done
	@echo "Launched 8 substep-test runs (50 and 60 substeps × 4 controllers)."
	@echo "Monitor: tail -f /tmp/substep_load_following_sub50.log  (etc.)"

# --- Run all independent simulations in parallel ---
# Launches: 4×1yr + 4×5yr + 8×substep = 16 background processes.
# NOTE: sensitivity_fixed_T and sensitivity_temp_decomp depend on 1yr CSVs;
#       run those separately after 1yr simulations finish.

run_all:
	$(CLEAR_CACHE)
	@echo "=== Launching all independent simulations (16 processes) ==="
	@echo ""
	@echo "--- 1-year single runs ---"
	@for c in load_following price_aware degradation_aware lifetime_aware; do \
		rm -rf results/$${c}_wind_spot_dk1; \
		nohup bash -c "echo 'START 1yr_'$$c': '$$(date -u +%Y-%m-%dT%H:%M:%S); \
			time $(PYTHON) run/single_simulation.py \
				--plant $(PLANT) \
				--controller configs/controllers/$$c.yaml \
				--power-profile $(POWER_WIND) \
				--price-profile $(PRICE_SPOT); \
			echo 'END 1yr_'$$c': '$$(date -u +%Y-%m-%dT%H:%M:%S)" \
			> /tmp/1yr_$$c.log 2>&1 & \
		echo "  started 1yr $$c (PID $$!)"; \
	done
	@echo ""
	@echo "--- 5-year lifetime runs ---"
	@rm -rf results/lifetime_5yr && mkdir -p results/lifetime_5yr
	@for c in load_following price_aware degradation_aware lifetime_aware; do \
		nohup bash -c "echo 'START 5yr_'$$c': '$$(date -u +%Y-%m-%dT%H:%M:%S); \
			time $(PYTHON) run/sensitivity/lifetime_5yr.py --controller $$c; \
			echo 'END 5yr_'$$c': '$$(date -u +%Y-%m-%dT%H:%M:%S)" \
			> /tmp/lt5yr_$$c.log 2>&1 & \
		echo "  started 5yr $$c (PID $$!)"; \
	done
	@echo ""
	@echo "--- Substep sensitivity test (sub50, sub60) ---"
	@rm -rf results/substep_test && mkdir -p results/substep_test
	@for s in 50 60; do \
		for c in load_following price_aware degradation_aware lifetime_aware; do \
			nohup bash -c "echo 'START '$${c}'_sub'$$s': '$$(date -u +%Y-%m-%dT%H:%M:%S); \
				time $(PYTHON) run/single_simulation.py \
					--plant $(PLANT) \
					--controller configs/controllers/$$c.yaml \
					--power-profile $(POWER_WIND) \
					--price-profile $(PRICE_SPOT) \
					--n-substeps $$s \
					--name $${c}_wind_spot_dk1_sub$$s \
					--outdir results/substep_test/sub$$s/$$c; \
				echo 'END '$${c}'_sub'$$s': '$$(date -u +%Y-%m-%dT%H:%M:%S)" \
				> /tmp/substep_$${c}_sub$$s.log 2>&1 & \
			echo "  started substep $$c sub$$s (PID $$!)"; \
		done; \
	done
	@echo ""
	@echo "All 16 processes launched. Monitor logs in /tmp/:"
	@echo "  1yr:     tail -f /tmp/1yr_load_following.log"
	@echo "  5yr:     tail -f /tmp/lt5yr_load_following.log"
	@echo "  substep: tail -f /tmp/substep_load_following_sub50.log"
	@echo ""
	@echo "After 1yr runs finish, run sensitivity studies:"
	@echo "  make sensitivity_fixed_T_5yr"
	@echo "  make sensitivity_substeps"

clean:
	@echo "Removing results/"
	@rm -rf results
