# AI-Enabled HVAC Resource Optimization for Green Hospitality

Code and self-curated dataset for the article:

> Elnagar, A. K. *Data-Driven Sustainable Tourism and Green Hospitality: Assessing the Effects of AI-Enabled Resource Optimization on Environmental and Operational Performance.* Submitted to PeerJ Computer Science (manuscript ID 150256).

## Description

This repository holds the full analysis pipeline for the article.
It tests whether ensemble machine-learning models predict heating, ventilation and air-conditioning (HVAC) performance indicators better than a linear baseline.
Two targets are modelled from hourly desiccant-dehumidification data:

1. **Moisture removal capacity (MRC, kg water/h)**, the environmental-performance target.
2. **A sensible-plus-latent energy-demand proxy (kW, relative units)**, the operational-performance target (Equation 1 of the article).

Three models are compared: Linear Regression (baseline), Random Forest and XGBoost.
Evaluation uses a chronological 70/15/15 split, five-fold time-series cross-validation, a paired bootstrap test and a feature ablation.

## Dataset Information

**File:** `final_manuscript_dataset_v2.csv` (self-curated dataset, CSV, UTF-8, comma-separated)

| Property | Value |
|---|---|
| Rows | 14,669 hourly observations |
| Period | 26 February 2018 to 31 December 2019 |
| Missing values | None |
| Duplicate rows | None |

**Source of component data.**
The self-curated dataset was assembled from a public dataset:

> Wu, C.-T. (2023). *In situ real-time dataset as compared to the predicted thermal wheel performance of an energy recovery system* (Version 2) [Data set]. Mendeley Data. https://doi.org/10.17632/pwm6cg2shj.2

The source dataset is described in a *Data in Brief* (2023) article.
The related research paper is *Optimization of a rotary desiccant wheel for enthalpy recovery of air-conditioning in a humid hospitality environment*, *Heliyon* (2022).

**Assembly steps.**
Raw measured channels (outdoor and supply air temperature and humidity ratio, simulated supply air, MRC) come from the source dataset.
Four redundant sensors per temperature and humidity channel were averaged to one hourly value.
A commissioning period without valid readings was removed.
The file has no missing cells and no duplicate rows.
The hourly index is continuous except for one recording gap from 2018-12-25 22:00 to 2019-02-26 09:00 (1,498 hours).

Derived columns were computed from the measured channels as follows (every deterministic formula below reproduces the file exactly):

- `Energy_Proxy_Sensible_kW` = 1.2 × 1.0 × 1.005 × max(0, T_supply − T_outdoor)
- `Energy_Proxy_Total_kW` = `Energy_Proxy_Sensible_kW` + 1.2 × 1.0 × 2260 × max(0, W_outdoor − W_supply)
- `Environmental_Efficiency_kgWater_per_kWh` = MRC / `Energy_Proxy_kW`
- `Sim_*_Error` = |simulated − measured| supply air
- `Comfort_Deviation_Temp_C` = |T_supply − 24|; `Comfort_Deviation_Humidity_kgkg` = |W_supply − 0.0105|
- `Guest_Comfort_Score_0to100` = 100 × (1 − `Comfort_Deviation_Humidity_kgkg` / 0.006)
- `SYNTHETIC_Occupancy_Rate_pct` = seasonal step levels (≈30%, 62% and 90%) plus random noise; illustrative only
- `SYNTHETIC_Resource_per_Occupant_kgWater` = MRC / `SYNTHETIC_Occupancy_Rate_pct`

**Columns.**

| Column | Description | Unit |
|---|---|---|
| `DateTime` | Hourly timestamp (local time) | – |
| `Outdoor_Air_Temperature_C` | Outdoor (ambient) air temperature, mean of four sensors | °C |
| `Outdoor_Air_Humidity_kgkg` | Outdoor air humidity ratio, mean of four sensors | kg water/kg dry air |
| `Indoor_Supply_Air_Temperature_C` | Supply (process) air temperature downstream of the desiccant wheel (≈60 °C process outlet) | °C |
| `Indoor_Supply_Air_Humidity_kgkg` | Supply (process) air humidity ratio downstream of the wheel | kg water/kg dry air |
| `AI_Simulated_Supply_Air_Temperature_C` | Physics-based (numerical) simulation of supply air temperature from the source dataset | °C |
| `AI_Simulated_Supply_Air_Humidity_kgkg` | Physics-based (numerical) simulation of supply air humidity ratio | kg water/kg dry air |
| `Moisture_Removal_Capacity_kgWater_per_h` | Moisture removal capacity (**target 1, MRC**) | kg water/h |
| `Month`, `Hour`, `DayOfYear` | Calendar fields from `DateTime` | – |
| `Season` | Winter (Dec–Feb), Spring (Mar–May), Summer (Jun–Aug), Autumn (Sep–Nov) | – |
| `Energy_Proxy_Sensible_kW` | Sensible term of Eq. 1 | kW |
| `Energy_Proxy_Total_kW` | Sensible + latent energy-demand proxy, Eq. 1 (**target 2**) | kW |
| `Energy_Proxy_kW` | Earlier sensible-only proxy (v1); kept for transparency, not modelled | kW |
| `Environmental_Efficiency_kgWater_per_kWh` | MRC / `Energy_Proxy_kW` | kg water/kWh |
| `Sim_Temperature_Error_C`, `Sim_Humidity_Error_kgkg` | Absolute difference between simulated and measured supply air | °C, kg/kg |
| `Comfort_Deviation_Temp_C`, `Comfort_Deviation_Humidity_kgkg` | Absolute deviation of supply air from 24 °C and 0.0105 kg/kg | °C, kg/kg |
| `Guest_Comfort_Score_0to100` | Humidity-control score (Figure 6) | 0–100 |
| `SYNTHETIC_Occupancy_Rate_pct` | **Synthetic** illustrative occupancy curve — not real data; not used in any reported result | % |
| `SYNTHETIC_Resource_per_Occupant_kgWater` | MRC / synthetic occupancy — illustrative only | kg water |

## Code Information

| File | Purpose |
|---|---|
| `AI_Resource_Optimization_Green_Hospitality_v2.ipynb` | End-to-end pipeline: data checks, feature engineering, chronological split, model training, metrics, cross-validation, bootstrap test, ablation and figures |
| `requirements.txt` | Pinned Python dependencies |

The notebook runs locally or on Google Colab. Locally, it reads the CSV from its own folder.

The notebook follows Algorithm 1 of the article:

1. Load the data and assert no missing values or duplicates.
2. Add cyclical month and hour features (sine and cosine).
3. Split the data chronologically into train (70%), validation (15%) and test (15%).
4. Fit Linear Regression, Random Forest and XGBoost on each target.
5. Record R², MAE, RMSE and MAPE on every split.
6. Run five-fold `TimeSeriesSplit` cross-validation on train plus validation.
7. Run a paired bootstrap test (1,000 resamples) on the test set against the baseline.
8. Run the ablation: climate-only features versus the full feature set.

## Usage Instructions

```bash
# 1. Clone or download this repository
git clone {{REPO_URL}}
cd {{REPO_DIR}}

# 2. Create an environment (Python 3.12)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the full pipeline
jupyter nbconvert --to notebook --execute AI_Resource_Optimization_Green_Hospitality_v2.ipynb --output executed.ipynb
# or open it interactively:
jupyter notebook AI_Resource_Optimization_Green_Hospitality_v2.ipynb
```

Keep `final_manuscript_dataset_v2.csv` in the same folder as the notebook.
The full run takes about 30–35 seconds on a standard CPU.
No GPU is needed.
All stochastic steps use `random_state=42`, so results reproduce exactly.

**Outputs** (written next to the notebook):

- `results_metrics_summary.csv` → Table 4
- `results_significance_tests_bootstrap.csv` → Table 5
- `results_significance_tests_cv.csv` and `results_cv_folds.csv` → cross-validation diagnostic
- `results_ablation_study.csv` → Table 6
- `figures/` → article figures
- `manuscript_handoff_summary.json` → every reported number in one file

## Requirements

- Python 3.12.3
- pandas 3.0.2
- NumPy 2.4.4
- scikit-learn 1.8.0
- XGBoost 3.4.1
- SciPy 1.17.1
- matplotlib 3.10.8
- seaborn 0.13.2
- Jupyter (to run the notebook)

These exact versions were used to verify that the notebook reproduces Tables 4–6.

## Methodology

**Features.**
Six predictors are used: outdoor air temperature, outdoor humidity ratio, and sine and cosine terms for month and hour.
Supply-air channels, simulation outputs and derived indices are excluded to prevent target leakage.

**Energy-demand proxy.**
E = ρ·V·c_p·max(0, T_indoor − T_outdoor) + ρ·V·h_fg·max(0, W_outdoor − W_indoor)

**Hyperparameters.**

| Model | Settings |
|---|---|
| Linear Regression | scikit-learn defaults |
| Random Forest | n_estimators=150, max_depth=10, min_samples_leaf=3, random_state=42 |
| XGBoost | n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method=hist, random_state=42 |

**Validation.**
The split is chronological to avoid leakage from autocorrelated hourly data.
Hyperparameters were selected on the validation split only.
The test split (October–December 2019) is used only for final reporting.

## Citations

If you use this code or data, please cite the article above and the source dataset:

- Wu, C.-T. (2023). In situ real-time dataset as compared to the predicted thermal wheel performance of an energy recovery system (Version 2). Mendeley Data. https://doi.org/10.17632/pwm6cg2shj.2

## License & Contribution Guidelines

- **Code:** MIT License.
- **Data:** the self-curated dataset is derived from the Mendeley Data source above and is redistributed under that source's licence terms. Cite the original dataset when reusing it.

Issues and pull requests are welcome.
Please describe the change and confirm the notebook still runs end to end.

## Contact

Ahmed K. Elnagar, ahmed_k_elnagar@creatrox.info, ORCID 0009-0006-2143-7246
