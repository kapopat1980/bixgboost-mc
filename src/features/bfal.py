"""
Bifacial Feature Augmentation Layer (BFAL)

Computes five physics-informed derived features from raw meteorological
observations, as described in Section 3.3 of the paper.

Features:
  BGC  — Bifacial Gain Coefficient       (Eq. 1)
  KT   — Clearness Index                 (Eq. 2)
  TRI  — Thermal Resistance Index        (Eq. 3)
  HSI  — Humidity-Spectral Index         (Eq. 4)
  WCCF — Wind Chill Correction Factor    (Eq. 5)
"""

import numpy as np
import pandas as pd


def sky_view_factor(tilt_deg: float, row_height_m: float, row_pitch_m: float) -> float:
    """
    Compute the sky-view factor phi_sky from installation geometry using
    the view-factor integral (Deline et al., 2020 — Ref [18] in paper).

    Parameters
    ----------
    tilt_deg : float
        Module tilt angle from horizontal (degrees).
    row_height_m : float
        Hub height of the module above the ground (metres).
    row_pitch_m : float
        Centre-to-centre distance between adjacent rows (metres).

    Returns
    -------
    float
        Sky-view factor in [0, 1].
    """
    tilt_rad = np.radians(tilt_deg)
    h = row_height_m
    p = row_pitch_m
    # Simplified two-dimensional view-factor integral
    phi = 0.5 * (1 + np.cos(tilt_rad)) - (h / p) * np.sin(tilt_rad)
    return float(np.clip(phi, 0.0, 1.0))


def extraterrestrial_irradiance(doy: np.ndarray) -> np.ndarray:
    """
    Extra-terrestrial horizontal irradiance using the Spencer (1971) formula.

    Parameters
    ----------
    doy : array-like
        Day-of-year (1–365/366).

    Returns
    -------
    np.ndarray
        GHI_extra in W/m².
    """
    b = 2 * np.pi * (doy - 1) / 365.0
    E0 = 1.000110 + 0.034221 * np.cos(b) + 0.001280 * np.sin(b) \
         + 0.000719 * np.cos(2 * b) + 0.000077 * np.sin(2 * b)
    return 1361.0 * E0   # solar constant × eccentricity correction


def precipitable_water(rh: np.ndarray, t_celsius: np.ndarray) -> np.ndarray:
    """
    Estimate precipitable water vapour (cm) from relative humidity and
    temperature using the Magnus formula approximation.
    """
    # Saturation vapour pressure (hPa) via Magnus formula
    es = 6.1078 * np.exp(17.269 * t_celsius / (237.3 + t_celsius))
    e = (rh / 100.0) * es
    # Precipitable water approximation (Leckner, 1978)
    pw = 0.493 * e / (t_celsius + 273.15)
    return pw


class BifacialFeatureAugmentationLayer:
    """
    Computes the five BFAL physics-informed features and concatenates them
    with the eight raw meteorological input variables to produce a 13-dim
    input vector for each time step.

    Raw inputs (8):
        ghi, dni, dhi, t_amb, rh, wind_speed, wind_dir, p_atm

    Derived BFAL features (5):
        BGC, KT, TRI, HSI, WCCF

    Parameters
    ----------
    tilt_deg : float
        Module tilt angle (degrees). Default: 25.
    row_height_m : float
        Hub height above ground (metres). Default: 1.5.
    row_pitch_m : float
        Row pitch (metres). Default: 6.0.
    wccf_alpha : float
        Empirical WCCF constant calibrated on PVDAQ training set. Default: 0.012.
    t_ref : float
        Standard reference temperature (°C). Default: 25.
    ground_albedo : float or None
        Fixed ground albedo if dataset lacks a measured albedo column.
        If None, the 'albedo' column from the DataFrame is used. Default: 0.2.
    """

    def __init__(
        self,
        tilt_deg: float = 25.0,
        row_height_m: float = 1.5,
        row_pitch_m: float = 6.0,
        wccf_alpha: float = 0.012,
        t_ref: float = 25.0,
        ground_albedo: float | None = 0.2,
    ):
        self.tilt_deg = tilt_deg
        self.row_height_m = row_height_m
        self.row_pitch_m = row_pitch_m
        self.wccf_alpha = wccf_alpha
        self.t_ref = t_ref
        self.ground_albedo = ground_albedo
        self.phi_sky = sky_view_factor(tilt_deg, row_height_m, row_pitch_m)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute BFAL features and append to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: ghi, t_amb, rh, wind_speed, t_mod, ghi_poa.
            Optional: albedo (used if ground_albedo is None), doy.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with five new columns appended:
            bgc, kt, tri, hsi, wccf.
        """
        df = df.copy()
        ghi = df["ghi"].values.astype(float)

        # --- Eq. 1: Bifacial Gain Coefficient (BGC) ---
        if self.ground_albedo is not None:
            rho = self.ground_albedo
        else:
            rho = df["albedo"].values.astype(float)
        ghi_front = np.where(ghi > 0, ghi, 1.0)   # avoid /0 at night
        bgc = rho * ghi * self.phi_sky / ghi_front
        bgc = np.where(ghi > 0, bgc, 0.0)
        df["bgc"] = bgc

        # --- Eq. 2: Clearness Index (KT) ---
        if "doy" in df.columns:
            doy = df["doy"].values
        else:
            doy = df.index.dayofyear.values
        ghi_extra = extraterrestrial_irradiance(doy)
        ghi_extra = np.where(ghi_extra > 0, ghi_extra, 1.0)
        kt = np.clip(ghi / ghi_extra, 0.0, 1.1)
        df["kt"] = kt

        # --- Eq. 3: Thermal Resistance Index (TRI) ---
        t_mod = df["t_mod"].values.astype(float)
        t_amb = df["t_amb"].values.astype(float)
        ghi_poa = df["ghi_poa"].values.astype(float)
        ghi_poa_safe = np.where(ghi_poa > 5.0, ghi_poa, np.nan)
        tri = (t_mod - t_amb) / ghi_poa_safe
        tri = np.where(np.isfinite(tri), tri, 0.0)
        df["tri"] = tri

        # --- Eq. 4: Humidity-Spectral Index (HSI) ---
        rh = df["rh"].values.astype(float)
        pw = precipitable_water(rh, t_amb)
        hsi = rh * np.exp(-0.0065 * pw)
        df["hsi"] = hsi

        # --- Eq. 5: Wind Chill Correction Factor (WCCF) ---
        v_wind = df["wind_speed"].values.astype(float)
        t_diff = np.abs(t_amb - self.t_ref)
        t_diff_safe = np.where(t_diff > 0.1, t_diff, 0.1)   # avoid /0
        wccf = 1.0 + self.wccf_alpha * (v_wind ** 0.8) / t_diff_safe
        df["wccf"] = wccf

        return df

    @property
    def feature_names(self) -> list[str]:
        return ["bgc", "kt", "tri", "hsi", "wccf"]

    @property
    def raw_feature_names(self) -> list[str]:
        return ["ghi", "dni", "dhi", "t_amb", "rh", "wind_speed", "wind_dir", "p_atm"]

    @property
    def all_feature_names(self) -> list[str]:
        return self.raw_feature_names + self.feature_names
