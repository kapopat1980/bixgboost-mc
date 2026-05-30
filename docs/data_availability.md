# Data Availability Guide

All datasets used in BiXGBoost-MC are **freely and publicly available** from
official NREL and DOE repositories. No registration or license approval is
required to download them.

---

## Dataset 1 — NREL SRRL Baseline Measurement System (BMS)

**Used for:** GHI forecasting task (Tables 3–4, Figure 4)

| Property | Value |
|----------|-------|
| Source | National Renewable Energy Laboratory |
| URL | https://midcdmz.nrel.gov/srrl_bms/ |
| Location | Golden, Colorado, USA (39.74°N, 105.18°W) |
| Period used | 2018–2022 (5 years) |
| Native resolution | 1 minute |
| Resampled to | 10 minutes (Section 3.2) |
| Records (post-filter) | ~143,000 daytime observations |

### Variables used
| Variable | SRRL column name |
|----------|-----------------|
| GHI | Global CMP22 (vent/cor) [W/m^2] |
| DNI | Direct NIP [W/m^2] |
| DHI | Diffuse 8-48 (vent/cor) [W/m^2] |
| Ambient temperature | Tower Dry Bulb Temp [deg C] |
| Relative humidity | Tower RH [%] |
| Wind speed | Avg Wind Speed @ 10m [m/s] |
| Wind direction | Avg Wind Direction @ 10m [deg] |
| Station pressure | Station Pressure [mBar] |

### Download steps
1. Visit https://midcdmz.nrel.gov/srrl_bms/
2. Click **Download Data** in the left panel
3. Select **1-minute** resolution
4. Select years 2018, 2019, 2020, 2021, 2022
5. Download each year as a CSV file
6. Place files in `data/raw/nrel_srrl/` named `srrl_bms_YYYY.csv`

---

## Dataset 2 — DOE PVDAQ Bifacial Module Temperature

**Used for:** Module temperature forecasting task (Table 5, Figure 5)

| Property | Value |
|----------|-------|
| Source | U.S. Department of Energy PVDAQ |
| URL | https://pvdaq.nrel.gov/ |
| System ID | 4 (bifacial PV testbed) |
| Location | National Wind Technology Center, Colorado |
| Period used | 2016–2021 (6 years) |
| Native resolution | 15 minutes |
| Resampled to | 10 minutes |
| Records (post-filter) | ~104,000 daytime observations |

### Download steps
1. Visit https://pvdaq.nrel.gov/
2. In the **Data Access** panel, select System ID = 4
3. Select years 2016–2021
4. Download as CSV (15-min resolution)
5. Place files in `data/raw/pvdaq/` named `pvdaq_system4_YYYY.csv`

---

## Dataset 3 — OEDI Bifacial PV Performance Dataset

**Used for:** Bifacial energy yield forecasting task (Table 6, Figure 6)

| Property | Value |
|----------|-------|
| Source | Open Energy Data Initiative (OEDI) |
| URL | https://data.openei.org/submissions/4568 |
| Direct S3 | https://oedi-data-lake.s3.amazonaws.com/pvdaq/ |
| Location | NREL Flatirons Campus, Colorado |
| Period used | 2019–2021 (3 years) |
| Native resolution | 5 minutes |
| Resampled to | 10 minutes |
| Records (post-filter) | ~57,000 daytime observations |

### Download steps
1. Visit https://data.openei.org/submissions/4568
2. Click **Download** for all files under `bifacial_field_data/`
3. Place files in `data/raw/oedi/` named `bifacial_YYYY_MM.csv`

---

## Pre-processing Notes

All pre-processing is automated via `scripts/preprocess.py`. Key steps:

1. **Resampling** — 1-min / 5-min / 15-min data resampled to 10-min means via `pd.resample("10T").mean()`
2. **Nighttime filter** — Records with solar elevation ≤ 5° removed (using SZA column where available, otherwise GHI ≤ 0)
3. **Gap filling** — Linear interpolation for gaps ≤ 30 minutes (3 consecutive missing steps); longer gaps dropped
4. **BFAL augmentation** — Five physics-informed features added per dataset (see `src/features/bfal.py`)
5. **Chronological split** — 70% train / 10% validation / 20% test, no shuffling
6. **Normalisation** — Min-max scaling fitted exclusively on the training set, applied to val/test

---

## Citation of Datasets

If you use these datasets in your own work, please cite them directly:

**NREL SRRL BMS:**
> Andreas, A.; Stoffel, T. (1981). NREL Solar Radiation Research Laboratory (SRRL): Baseline Measurement System (BMS). Data collected 1981–present. https://dx.doi.org/10.5439/1052221

**PVDAQ:**
> Klise, G.; Freeman, J. (2013). PV Data Acquisition (PVDAQ) System. NREL Technical Report. https://pvdaq.nrel.gov/

**OEDI Bifacial:**
> Marion, B. et al. (2021). Bifacial PV Field Performance Dataset. Open Energy Data Initiative. https://data.openei.org/submissions/4568
