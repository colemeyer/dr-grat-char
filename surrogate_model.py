"""
surrogate_model.py
==================
Fast interpolated surrogate for the GD-CALC diffraction-efficiency model.

GD-CALC (Grating Diffraction Calculator, K. C. Johnson) is run *offline* in
MATLAB over a regular grid of grating parameters, and its outputs are saved as
CSV files under ``MODEL_DIR`` (see the README). Re-running GD-CALC inside an
MCMC loop would be far too slow, so this module loads those precomputed grids
once and wraps them in a ``scipy`` ``RegularGridInterpolator``. The resulting
``interp`` callable returns a model efficiency spectrum for arbitrary parameter
values in microseconds.

Model grid
----------
Each CSV is named ``tB<tB>_dt<dt>_land<land>.csv`` and holds, per row:
``[wavelength_um, eff_order_-2, eff_order_-1, eff_order_0, eff_order_+1,
eff_order_+2]``. The grids are stored per groove density in subdirectories
``<MODEL_DIR>/800grmm`` and ``<MODEL_DIR>/2000grmm``.

Grid axes:
  * ``tB``   : blaze-related thickness parameter, 10..18 step 2
  * ``dt``   : facet asymmetry angle (deg), -5..5 step 1
  * ``land`` : land (flat-top) duty cycle (%), 0..100 step 20
  * wavelength: read from the CSVs (converted um -> nm by the 1000x factor)
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import config


def init_surrogate():
    """
    Load the GD-CALC model grids and build the interpolating surrogate.

    Returns
    -------
    wav_grid : 1-D float ndarray
        Model wavelength axis in nm.
    interp : callable
        ``interp(f, m, dt, land, tB=13.8) -> 1-D ndarray`` giving the modelled
        diffraction efficiency (percent) versus ``wav_grid`` for groove density
        ``f`` (800 or 2000), signed order ``m`` (-2..+2) and grating parameters
        ``dt`` and ``land`` (``tB`` is held at its default unless overridden).

    Notes
    -----
    A separate ``RegularGridInterpolator`` is built for each (groove density,
    order) pair. Efficiencies are scaled to percent (the raw CSV values are
    multiplied by 100) to match the units used downstream.
    """
    # Parameter grid axes (must match the GD-CALC sampling used to make the CSVs).
    tB_grid = np.arange(10, 20, 2)     # [10, 12, 14, 16, 18]
    dt_grid = np.arange(-5, 6)         # [-5, ..., 5]
    land_grid = np.arange(0, 120, 20)  # [0, 20, 40, 60, 80, 100]

    # Map signed diffraction order -> CSV efficiency column (1-based; column 0
    # is wavelength). Order -2 is column 1, ..., order +2 is column 5.
    col = {-2: 1, -1: 2, 0: 3, 1: 4, 2: 5}

    interps = {}
    wav_grid = None

    for f in [800, 2000]:
        model_dir = f'{config.MODEL_DIR}/{f}grmm'

        # Wavelength axis from a reference CSV (um -> nm via the 1000x factor).
        wav_grid = 1000 * np.genfromtxt(
            f'{model_dir}/tB10_dt0_land0.csv',
            delimiter=',', dtype='float'
        )[:, 0]

        # raw[i, j, k, :, c] = efficiency spectrum for (tB_i, dt_j, land_k), order c.
        raw = np.zeros((len(tB_grid), len(dt_grid), len(land_grid), len(wav_grid), 5))

        for i, tB in enumerate(tB_grid):
            for j, dt in enumerate(dt_grid):
                for k, land in enumerate(land_grid):
                    arr = np.genfromtxt(
                        f'{model_dir}/tB{int(tB)}_dt{int(dt)}_land{int(land)}.csv',
                        delimiter=',', dtype='float'
                    )
                    raw[i, j, k, :, :] = 100 * arr[:, 1:6]  # 5 order columns -> percent

        # One interpolator per (groove density, order).
        for m, c in col.items():
            interps[(f, m)] = RegularGridInterpolator(
                (tB_grid, dt_grid, land_grid, wav_grid),
                raw[:, :, :, :, c - 1]
            )

    def interp(f, m, dt, land, tB=13.8):
        """Return the modelled efficiency spectrum (percent) over ``wav_grid``."""
        pts = np.column_stack([
            np.full(len(wav_grid), tB),
            np.full(len(wav_grid), dt),
            np.full(len(wav_grid), land),
            wav_grid,
        ])
        return interps[(f, m)](pts)

    return wav_grid, interp
