# BMA-Stacking: Ensemble Runoff Prediction

A research project on ensemble hydrological forecasting based on the CAMELS dataset, comparing two ensemble strategies — Bayesian Model Averaging (BMA) and Stacking — for daily runoff prediction.

## Background

This project uses the CAMELS (Catchment Attributes and MEteorology for Large-sample Studies) dataset to produce daily runoff forecasts with lead times of 1–7 days for 670 catchments across the contiguous United States. Three machine learning models (SVR, Random Forest, and MLP) are combined through BMA and Stacking, and their predictive performance is compared.

## Notes
- Catchment 03281100 is skipped in the pipeline due to data issues.

## Project Structure

```
bma-stacking/
├── demo/
│   ├── main.py               # Recursive strategy: full pipeline (run from demo/)
│   └── main_for_multi.py     # Direct strategy
├── models/
│   ├── BMA.py                # BMA implementation (EM algorithm)
│   └── Stacking.py           # Stacking ensemble
├── util/
│   ├── BOX_COX.py            # Box-Cox transformation parameter optimization
│   ├── data_process.py       # Data loading, preprocessing, sliding-window construction
│   ├── estimators.py         # Base model definitions and evaluation metrics
│   ├── grid_search.py        # Hyperparameter grid search (TimeSeriesSplit)
│   ├── predict.py            # Recursive multi-step forecasting
│   └── result.py             # Result comparison utilities
├── data/                     # CAMELS data (NetCDF)
│   ├── camels_streamflow.nc  # Daily streamflow
│   └── camels_forcingmete.nc # Meteorological forcing
└── camels_name.txt           # List of CAMELS catchment IDs
```

## Methodology

### Base Models
- **SVR** (Support Vector Regression) — RBF kernel
- **Random Forest** — ensemble of decision trees
- **MLP** (BP Neural Network) — multi-layer perceptron

### Ensemble Methods
1. **BMA (Bayesian Model Averaging)**: learns optimal model weights via the EM algorithm, accounting for the predictive variance of each model. Requires a Box-Cox transformation to approximate normality.
2. **Stacking**: uses Ridge regression as the meta-learner to combine base model predictions.

All ensembles use the fixed combination of the three base models (SVR + Random Forest + MLP).

### Evaluation Metrics
- R² (coefficient of determination)
- RMSE (root mean square error)
- MAPE (mean absolute percentage error)

## Data

The CAMELS dataset is used, including:
- **Meteorological forcing**: precipitation (prcp), maximum temperature (tmax), minimum temperature (tmin)
- **Hydrological data**: daily discharge

Time period: 1995-01-01 to 2004-12-31, with an 80%/20% train/test split (chronological order preserved). Features are constructed with a 7-day sliding window.

## Usage

### Recursive strategy: full pipeline
```bash
cd demo
python main.py
```

### Direct strategy
```bash
python demo/main_for_multi.py
```

The scripts skip catchments whose output files already exist, so interrupted runs can be resumed. To re-run a specific catchment, delete its corresponding output files.



