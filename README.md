# BiXGBoost-MC

**A Hybrid Bi-LSTM and XGBoost Architecture for High-Resolution 
Microclimate Forecasting in Bifacial Solar Farms**

## Datasets
- [NREL SRRL BMS](https://midcdmz.nrel.gov/srrl_bms/)
- [DOE PVDAQ](https://pvdaq.nrel.gov/)
- [OEDI Bifacial PV](https://oedi-data-lake.s3.amazonaws.com/pvdaq/)

## Requirements
- Python 3.9+
- PyTorch 2.0+
- XGBoost 1.7+
- NumPy, Pandas, Scikit-learn, Optuna, SHAP

## Reproducibility
All experiments use `random seed 42`:
- NumPy: `np.random.seed(42)`
- PyTorch: `torch.manual_seed(42)`

## Citation
If you use this work, please cite:
Mehta, A. et al. (2025). A Hybrid Bi-LSTM and XGBoost Architecture 
for High-Resolution Microclimate Forecasting in Bifacial Solar Farms.
