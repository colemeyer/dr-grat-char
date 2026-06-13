"""
step1_compute_efficiencies.py
=============================
PIPELINE STEP 1 of 2 — turn raw CCD frames into measured diffraction
efficiencies with uncertainties.

This script merges what were previously two separate scripts
(``calc_count-rates.py`` and ``calc_diffr-effs.py``) into a single, ordered
run. It performs two stages and writes four HDF5 products into ``DATA_DIR``:

  Stage A — count rates  (``compute_count_rates``)
  ------------------------------------------------
  For every ``.sif`` frame (incident reference frames and diffracted frames):
    1. Load the frame and subtract its stray-light background
       (``background_subtraction.sub_bg``).
    2. Integrate to a count rate = sum(background-subtracted counts) / exposure.
    3. Propagate a combined shot + background uncertainty (relative).
  Writes:  count_rates.h5,  count_rates_err.h5

  Stage B — diffraction efficiencies  (``compute_diffraction_efficiencies``)
  --------------------------------------------------------------------------
  For each panel / order, divide the diffracted count rate by the incident
  (reference) count rate to get the diffraction efficiency, propagate the
  uncertainty, and add a fixed experimental systematic in quadrature.
  Writes:  diffr_effs.h5,  diffr_effs_err.h5

The two stages communicate through the on-disk HDF5 files, exactly as the
original two-script workflow did, so the numerical results are identical.

Per-frame configuration
------------------------
Background subtraction occasionally needs per-frame tuning (a different
low-threshold factor ``k_low``, or a hand-drawn ``beam_mask`` for frames where
automatic detection fails). These overrides live in ``FRAME_CFG`` and are
resolved by ``_resolve_cfg``; see the comments there for the keying rules.

Run with:
    uv run step1_compute_efficiencies.py
"""

import os
import time

import h5py
import numpy as np
import sif_parser

import config
from background_subtraction import sub_bg


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame background-subtraction overrides
# ─────────────────────────────────────────────────────────────────────────────
# Two keying styles can be mixed freely:
#
#   Group-level (applies to every frame in a directory):
#     'dif/panel1/m=-1(CCW)': {'k_low': [2.0, 3.5, 4.0, 2.5, 3.0, 4.0]}
#       A list must have one entry per .sif file in that directory (in the same
#       sorted order as the processing loop). A scalar value applies to every
#       frame. 'beam_mask' may also be a list of arrays/None on the same index
#       scheme.
#
#   File-level (applies to one specific frame):
#     'inc/550_0001.sif':                  {'k_low': 2.5}
#     'dif/panel1/m=-1(CCW)/450_0001.sif': {'beam_mask': rect_mask(...)}
#
# File-level entries take priority over group-level entries.

def rect_mask(shape, r0, r1, c0, c1):
    """
    Build a rectangular boolean beam mask.

    Parameters
    ----------
    shape : tuple of int
        Output array shape (ny, nx).
    r0, r1 : int
        Inclusive/exclusive row bounds of the True rectangle.
    c0, c1 : int
        Inclusive/exclusive column bounds of the True rectangle.

    Returns
    -------
    mask : 2-D bool ndarray
        True inside the rectangle, False elsewhere.
    """
    m = np.zeros(shape, dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


# Diagnostic plotting switch. When True, frames whose group/file key appears in
# PLOT_CFG are shown with the six-panel sub_bg diagnostic. Default off.
PLOT_BG_SUB = False

# Keys (group or file) to plot when PLOT_BG_SUB is True. Empty set => plot none.
PLOT_CFG = set()

FRAME_CFG = {

    'inc': {
        'k_low': [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    },
    # -----------------
    'dif/panel1/m=+1(CW)': {
        'k_low': [1.5, 20.0, 20.0, 20.0, 20.0, 20.0],
    },
    'dif/panel1/m=+2(CW)': {
        'k_low': [1.5, 20.0, 20.0, 20.0, 20.0],
    },
    'dif/panel1/m=-1(CCW)': {
        'k_low': [0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    'dif/panel1/m=-2(CCW)': {
        'k_low': [1.0, 1.0, 1.0, 1.0, 1.0],
    },
    'dif/panel1/m=0': {
        'k_low': [20.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    # -----------------
    'dif/panel2/m=+1(CW)': {
        'k_low': [1.0, 1.0, 1.0],
    },
    'dif/panel2/m=-1(CCW)': {
        'k_low': [1.0, 1.0, 1.0],
    },
    'dif/panel2/m=-2(CCW)': {
        'k_low': [1.0],
    },
    'dif/panel2/m=+2(CW)': {
        'k_low': [1.0],
    },
    'dif/panel2/m=0': {
        'k_low': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    # -----------------
    'dif/panel3/m=+1(CW)': {
        'k_low': [1.0, 1.0, 20.0, 20.0, 20.0, 1.0],
    },
    'dif/panel3/m=+2(CW)': {
        'k_low': [1.0, 20.0, 1.0, 1.0, 1.0],
    },
    'dif/panel3/m=-1(CCW)': {
        'k_low': [0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
    },
    'dif/panel3/m=-2(CCW)': {
        'k_low': [1.0, 5.0, 1.0, 1.0, 1.0],
    },
    'dif/panel3/m=0': {
        'k_low': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    },

    # File-level overrides (beat the group entry above for that one frame).
    'dif/panel1/m=-1(CCW)/700nm_2119s.sif': {
        'beam_mask': rect_mask((1024, 1024), 180, 900, 180, 900),
    },
    'dif/panel1/m=-2(CCW)/400nm_1592s.sif': {
        'beam_mask': rect_mask((1024, 1024), 150, 975, 200, 700),
    },
    'dif/panel1/m=-2(CCW)/500nm_2016s.sif': {
        'beam_mask': rect_mask((1024, 1024), 150, 975, 225, 800),
    },
    'dif/panel1/m=-2(CCW)/600nm_1923s.sif': {
        'beam_mask': rect_mask((1024, 1024), 200, 975, 225, 900),
    },
    'dif/panel1/m=0/200nm_1634s.sif': {
        'beam_mask': rect_mask((1024, 1024), 100, 650, 50, 950),
    },
    'dif/panel2/m=-1(CCW)/200nm_1852s.sif': {
        'beam_mask': rect_mask((1024, 1024), 100, 800, 300, 525),
    },
    'dif/panel2/m=0/200nm_1613s.sif': {
        'beam_mask': rect_mask((1024, 1024), 100, 650, 50, 950),
    },
    'dif/panel3/m=0/200nm_1701s.sif': {
        'beam_mask': rect_mask((1024, 1024), 400, 999, 200, 900),
    },
    'dif/panel3/m=+1(CW)/200nm_2016s.sif': {
        'beam_mask': rect_mask((1024, 1024), 150, 900, 400, 800),
    },
    'dif/panel3/m=+1(CW)/300nm_293.s.sif': {
        'beam_mask': rect_mask((1024, 1024), 25, 999, 400, 999),
    },
    'dif/panel3/m=-2(CCW)/500nm_1761s.sif': {
        'beam_mask': rect_mask((1024, 1024), 200, 975, 100, 600),
    },
    'dif/panel3/m=-2(CCW)/600nm_2294s.sif': {
        'beam_mask': rect_mask((1024, 1024), 200, 950, 200, 950),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sif_files(directory):
    """Return the sorted list of ``.sif`` filenames in ``directory``."""
    return [f for f in np.sort(os.listdir(directory)) if f.endswith('.sif')]


def _resolve_cfg(group_key, file_key, file_index):
    """
    Merge group-level and file-level ``sub_bg`` config for a single frame.

    Priority: file-level > group-level > defaults. For list-valued group
    entries, the element at ``file_index`` is selected. ``None`` values are
    dropped so that ``sub_bg`` falls back to its own defaults.

    Parameters
    ----------
    group_key : str
        e.g. ``'inc'`` or ``'dif/panel1/m=0'``.
    file_key : str
        Group key plus the filename, e.g. ``'dif/panel1/m=0/200nm_1634s.sif'``.
    file_index : int
        Position of this frame within its directory's sorted file list.

    Returns
    -------
    cfg : dict
        Keyword arguments to pass to ``sub_bg``.
    """
    cfg = {}

    # 1. Group-level config, extracting the per-frame element from any lists.
    group_cfg = FRAME_CFG.get(group_key, {})
    for k, v in group_cfg.items():
        cfg[k] = v[file_index] if isinstance(v, list) else v

    # 2. File-level config overwrites group-level config.
    cfg.update(FRAME_CFG.get(file_key, {}))

    # 3. Drop any None values (treated as "use the sub_bg default").
    return {k: v for k, v in cfg.items() if v is not None}


def _run_sub_bg(group_key, file_key, file_index, image):
    """
    Resolve the per-frame config and run ``sub_bg`` on a single frame.

    Returns the full 5-tuple from ``sub_bg``:
    ``(sub, bg_model, mask_tight, residuals, bg_contribution)``.
    """
    cfg = _resolve_cfg(group_key, file_key, file_index)
    should_plot = PLOT_BG_SUB and (group_key in PLOT_CFG or file_key in PLOT_CFG)
    return sub_bg(image, verbose=should_plot, **cfg)


def _frame_count_rate_and_error(sub, bg, mask, exp):
    """
    Compute the count rate and its *relative* uncertainty for one frame.

    Parameters
    ----------
    sub : 2-D float ndarray
        Background-subtracted frame.
    bg : 2-D float ndarray
        Fitted background model for the frame.
    mask : 2-D bool ndarray
        Beam footprint (True = beam).
    exp : float
        Exposure time (s).

    Returns
    -------
    count_rate : float
        sum(sub) / exposure.
    rel_err : float
        Combined shot + background uncertainty, expressed relative to the
        count rate.

    Notes
    -----
    ``sub + bg`` recovers the original (pre-subtraction) counts, so the shot
    term ``sqrt(sum(sub[mask] + bg[mask]))`` is the Poisson noise on the total
    in-beam signal. The background term scales the per-pixel background scatter
    by the number of beam pixels. Both are converted to a rate (divide by
    exposure) before being combined in quadrature, then normalised by the
    count rate to give a relative uncertainty.
    """
    count_rate = np.nansum(sub) / exp

    sigma_bg = np.nanstd(sub[~mask])                       # per-pixel BG scatter
    sigma_shot = np.sqrt(np.nansum(sub[mask] + bg[mask]))  # Poisson on total signal

    sigma_tot = np.sqrt(sigma_shot ** 2 / exp ** 2
                        + sigma_bg ** 2 * np.nansum(mask) / exp ** 2)
    rel_err = sigma_tot / count_rate
    return count_rate, rel_err


def _parse_filename(file_i):
    """
    Parse a frame filename into (wavelength_nm, exposure_s).

    Filenames follow the pattern ``<wav>nm_<exp>s.sif`` where ``<wav>`` is a
    3-digit wavelength and ``<exp>`` is a 4-digit exposure time, e.g.
    ``700nm_2119s.sif`` -> (700, 2119.0).
    """
    wav = int(file_i[:3])
    exp = float(file_i[-9:-5])
    return wav, exp


# ─────────────────────────────────────────────────────────────────────────────
# Stage A — count rates
# ─────────────────────────────────────────────────────────────────────────────

def compute_count_rates():
    """
    Build ``count_rates.h5`` and ``count_rates_err.h5`` from the raw frames.

    Iterates over the incident reference frames (``data/inc_frames``) and the
    diffracted frames (``data/dif_frames/<panel>/<order>``), background-
    subtracts each frame, and stores the per-wavelength count rate and its
    relative uncertainty under a group named ``'inc'`` or
    ``'dif/<panel>/<order>'``.
    """
    inc_dir = os.path.join(config.DATA_DIR, 'inc_frames') + os.sep
    dif_dir = os.path.join(config.DATA_DIR, 'dif_frames') + os.sep

    rates_path = os.path.join(config.DATA_DIR, 'count_rates.h5')
    rates_err_path = os.path.join(config.DATA_DIR, 'count_rates_err.h5')

    with h5py.File(rates_path, 'w') as f, h5py.File(rates_err_path, 'w') as f_err:

        # ── Incident (reference) frames ─────────────────────────────────────
        wavs, count_rates, count_rates_err = [], [], []
        for i, file_i in enumerate(_sif_files(inc_dir)):
            wav, exp = _parse_filename(file_i)
            print('Working on inc for', wav, 'nm...')

            image, _ = sif_parser.np_open(inc_dir + file_i)
            image = np.squeeze(image)

            sub, bg, mask, _, _ = _run_sub_bg(
                group_key='inc',
                file_key=f'inc/{file_i}',
                file_index=i,
                image=image,
            )

            cr, rel_err = _frame_count_rate_and_error(sub, bg, mask, exp)
            wavs.append(wav)
            count_rates.append(cr)
            count_rates_err.append(rel_err)

        grp = f.create_group('inc')
        grp.create_dataset('wav', data=wavs)
        grp.create_dataset('count_rate', data=count_rates)

        grp = f_err.create_group('inc')
        grp.create_dataset('wav', data=wavs)
        grp.create_dataset('count_rate_err', data=count_rates_err)

        # ── Diffracted frames ───────────────────────────────────────────────
        for panel in np.sort(os.listdir(dif_dir)):
            if not os.path.isdir(dif_dir + panel):
                continue
            for order in np.sort(os.listdir(dif_dir + panel)):
                order_path = dif_dir + panel + '/' + order
                if not os.path.isdir(order_path):
                    continue

                group_key = f'dif/{panel}/{order}'
                wavs, count_rates, count_rates_err = [], [], []

                for i, file_i in enumerate(_sif_files(order_path)):
                    wav, exp = _parse_filename(file_i)
                    print('Working on dif for', panel + ',', order, '@', wav, 'nm...')

                    image, _ = sif_parser.np_open(order_path + '/' + file_i)
                    image = np.squeeze(image)

                    sub, bg, mask, _, _ = _run_sub_bg(
                        group_key=group_key,
                        file_key=f'{group_key}/{file_i}',
                        file_index=i,
                        image=image,
                    )

                    cr, rel_err = _frame_count_rate_and_error(sub, bg, mask, exp)
                    wavs.append(wav)
                    count_rates.append(cr)
                    count_rates_err.append(rel_err)

                grp = f.create_group(group_key)
                grp.create_dataset('wav', data=wavs)
                grp.create_dataset('count_rate', data=count_rates)

                grp = f_err.create_group(group_key)
                grp.create_dataset('wav', data=wavs)
                grp.create_dataset('count_rate_err', data=count_rates_err)


# ─────────────────────────────────────────────────────────────────────────────
# Stage B — diffraction efficiencies
# ─────────────────────────────────────────────────────────────────────────────

def compute_diffraction_efficiencies():
    """
    Build ``diffr_effs.h5`` and ``diffr_effs_err.h5`` from the count rates.

    The diffraction efficiency of each (panel, order) is the diffracted count
    rate divided by the incident reference count rate. The uncertainty combines
    the (relative) incident and diffracted errors and then adds a fixed
    experimental systematic ``config.SIGMA_EXP`` in quadrature.

    Efficiencies and their errors are stored as fractions (not percent).
    """
    rates_path = os.path.join(config.DATA_DIR, 'count_rates.h5')
    rates_err_path = os.path.join(config.DATA_DIR, 'count_rates_err.h5')
    effs_path = os.path.join(config.DATA_DIR, 'diffr_effs.h5')
    effs_err_path = os.path.join(config.DATA_DIR, 'diffr_effs_err.h5')

    with h5py.File(effs_path, 'w') as f_eff, \
            h5py.File(effs_err_path, 'w') as f_eff_err, \
            h5py.File(rates_path, 'r') as f_rate, \
            h5py.File(rates_err_path, 'r') as f_rate_err:

        # Incident reference count rate + relative error (the denominator).
        refs = f_rate['inc']['count_rate'][:]
        refs_err = f_rate_err['inc']['count_rate_err'][:]

        for panel in config.PANEL_GR_DENS:
            for order in config.ORDER_MAP:
                try:
                    grp = f_rate['dif/' + panel + '/' + order]
                    wavs = grp['wav'][:]
                    difs = grp['count_rate'][:]

                    grp = f_rate_err['dif/' + panel + '/' + order]
                    difs_err = grp['count_rate_err'][:]

                    # Efficiency = diffracted / incident (fraction).
                    diffr_effs = (difs / refs[:len(difs)])

                    # Combine relative incident + diffracted errors -> absolute,
                    # then add the fixed experimental systematic in quadrature.
                    diffr_effs_err = diffr_effs * np.sqrt(
                        (refs_err[:len(difs)]) ** 2 + (difs_err) ** 2
                    )
                    diffr_effs_err = np.sqrt(diffr_effs_err ** 2 + config.SIGMA_EXP ** 2)

                    grp = f_eff.create_group(panel + '/' + order)
                    grp.create_dataset('wav', data=wavs)
                    grp.create_dataset('diffr_eff', data=diffr_effs)

                    grp = f_eff_err.create_group(panel + '/' + order)
                    grp.create_dataset('wav', data=wavs)
                    grp.create_dataset('diffr_eff_err', data=diffr_effs_err)

                except Exception:
                    # Some (panel, order) combinations were not measured; skip
                    # them silently rather than failing the whole run.
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run both stages in order and report the wall-clock time."""
    t0 = time.time()
    os.makedirs(config.DATA_DIR, exist_ok=True)

    print('=== Stage A: count rates ===')
    compute_count_rates()

    print('\n=== Stage B: diffraction efficiencies ===')
    compute_diffraction_efficiencies()

    print('\nDone! Took', round(time.time() - t0, 1), 's to run...')


if __name__ == "__main__":
    main()
