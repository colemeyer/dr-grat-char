"""
config.py
=========
Single source of truth for every path and tunable constant used by the
grating diffraction-efficiency pipeline.

Both pipeline steps import from this module, so changing a value here changes
it everywhere downstream. Nothing in this file performs computation; it only
declares constants.

Directory layout
----------------
The default directory names are *relative*, so the scripts must be launched
from the repository, e.g.::

    uv run step1_compute_efficiencies.py

Expected layout under the root::

    data/
        inc_frames/                 incident (reference) .sif frames
        dif_frames/<panel>/<order>/ diffracted .sif frames
        # the four HDF5 products below are CREATED by step 1:
        count_rates.h5  count_rates_err.h5
        diffr_effs.h5   diffr_effs_err.h5
    model/
        800grmm/   GD-CALC efficiency grids for the 800 gr/mm panel
        2000grmm/  GD-CALC efficiency grids for the 2000 gr/mm panel
    plots/         figures written by step 2
"""

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_DIR = "data"      # raw frames + derived HDF5 products
MODEL_DIR = "model"    # GD-CALC model grids (one sub-dir per groove density)
PLOTS_DIR = "plots"    # output figures


# ---------------------------------------------------------------------------
# Detector / experiment description
# ---------------------------------------------------------------------------
# Diffraction panels measured in the paper and their ruling densities (gr/mm).
# Panels 1 and 3 are the 800 gr/mm panels; panel 2 is the 2000 gr/mm panel.
PANEL_GR_DENS = {"panel1": 800, "panel2": 2000, "panel3": 800}

# Mapping from the HDF5 group name of a diffraction order to its signed integer
# order index. CCW = counter-clockwise (negative orders), CW = clockwise
# (positive orders). The insertion order here defines the processing order.
ORDER_MAP = {
    "m=-2(CCW)": -2,
    "m=-1(CCW)": -1,
    "m=0": 0,
    "m=+1(CW)": +1,
    "m=+2(CW)": +2,
}


# ---------------------------------------------------------------------------
# Step 1 — efficiency computation
# ---------------------------------------------------------------------------
# Extra relative systematic uncertainty added in quadrature to every measured
# diffraction efficiency (accounts for experiment-level reproducibility that is
# not captured by per-frame shot/background noise).
SIGMA_EXP = 0.05  # 5% relative error on the diffraction efficiency


# ---------------------------------------------------------------------------
# Step 2 — surrogate model + MCMC fit
# ---------------------------------------------------------------------------
# Monochromator half-bandpass (nm). The model is averaged over
# [wav - BANDPASS, wav + BANDPASS] at each measurement wavelength to mimic the
# finite spectral resolution of the measurement.
BANDPASS = 6  # nm

# emcee settings. These are the values used for the published run.
NWALKERS = 12
NSTEPS = 5000
BURNIN = 2000

# Set to an integer for bit-reproducible chains. Leave as None to reproduce the
# original (unseeded) behaviour; the posteriors are statistically equivalent
# either way because the chains are well converged.
RNG_SEED = None

# Hard grid bounds for the two fitted parameters:
#   dt   : facet asymmetry angle (deg)
#   land : land (flat-top) duty cycle (%)
BOUNDS = {
    "dt": (-5.0, 5.0),
    "land": (0.0, 100.0),
}

# Gaussian priors (mean, std) on the fitted parameters.
PRIOR_MEANS = {"dt": 0.0, "land": 0.0}
PRIOR_STDS = {"dt": 2.0, "land": 10.0}

# Initial walker scatter around the prior mean.
DT_SCATTER = 1
LAND_SCATTER = 10
