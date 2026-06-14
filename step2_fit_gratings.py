"""
step2_fit_gratings.py
=====================
PIPELINE STEP 2 of 2 — fit a grating-profile model to the measured diffraction
efficiencies with MCMC.

For each measured panel this script:
  1. Loads the measured efficiencies + errors written by step 1
     (``diffr_effs.h5`` / ``diffr_effs_err.h5``).
  2. Builds the GD-CALC surrogate (``surrogate_model.init_surrogate``).
  3. Runs an ``emcee`` ensemble sampler over two grating-profile parameters,
       dt   = facet asymmetry angle (deg)
       land = land (flat-top) duty cycle (%),
     fitting all measured orders of the panel simultaneously.
  4. Prints the best-fit parameters and a per-order reduced chi-squared.
  5. Saves a combined corner + trace diagnostic figure per panel
     (``plots/mcmc_app_<panel>.pdf``).

An optional "keystone" figure (the stacked best-fit efficiency curves for all
three panels) can be produced by setting ``MAKE_KEYSTONE = True`` below; it is
off by default to match the archived run.

Reproducibility note
---------------------
MCMC is stochastic. With ``config.RNG_SEED = None`` (the default, matching the
original run) the chains differ run-to-run but the recovered posteriors are
statistically equivalent because the chains are well converged. Set
``config.RNG_SEED`` to an integer for bit-reproducible chains.

Run with:
    uv run step2_fit_gratings.py
"""

import os

import h5py
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from funkyfresh import set_style
from funkyfresh import standard_colors as sc

import config
from surrogate_model import init_surrogate

set_style('AAS', silent=True)

# Toggle to regenerate the stacked main-text efficiency-fit ("keystone") figure.
MAKE_KEYSTONE = False

# ---------------------------------------------------------------------------
# Build the GD-CALC surrogate once, at import time. `model_wav` is the model
# wavelength axis (nm); `model_de_pos` evaluates the surrogate. Both are used
# as module-level globals by the model/prediction helpers below.
# ---------------------------------------------------------------------------
model_wav, model_de_pos = init_surrogate()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(panel):
    """
    Load measured efficiencies and errors for one panel.

    Parameters
    ----------
    panel : str
        Panel HDF5 group name (``'panel1'``, ``'panel2'`` or ``'panel3'``).

    Returns
    -------
    data : list of (m, wavs, effs)
        One tuple per available order: integer order ``m``, wavelengths (nm),
        and efficiencies in percent.
    err : list of (m, wavs, effs_err)
        Matching per-order efficiency errors in percent.

    Orders that were not measured for this panel are simply skipped.
    """
    effs_path = os.path.join(config.DATA_DIR, 'diffr_effs.h5')
    effs_err_path = os.path.join(config.DATA_DIR, 'diffr_effs_err.h5')

    data = []
    with h5py.File(effs_path, 'r') as f:
        for key, m in config.ORDER_MAP.items():
            try:
                grp = f[panel + '/' + key]
                wavs = grp['wav'][:]
                effs = grp['diffr_eff'][:] * 100.0   # fraction -> percent
                data.append((m, wavs, effs))
            except KeyError:
                continue

    err = []
    with h5py.File(effs_err_path, 'r') as f:
        for key, m in config.ORDER_MAP.items():
            try:
                grp = f[panel + '/' + key]
                wavs = grp['wav'][:]
                effs = grp['diffr_eff_err'][:] * 100.0  # fraction -> percent
                err.append((m, wavs, effs))
            except KeyError:
                continue

    return data, err


# ─────────────────────────────────────────────────────────────────────────────
# Model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def model_de(gr_dens, m, dt, land):
    """Surrogate diffraction-efficiency spectrum (percent) over ``model_wav``."""
    return model_de_pos(gr_dens, m, dt, land)


def smooth_model(wav_array, de_array, bandpass):
    """
    Boxcar-average a high-resolution model spectrum (used for plotting only).

    Parameters
    ----------
    wav_array : 1-D ndarray
        Uniformly spaced model wavelengths (nm).
    de_array : 1-D ndarray
        Model efficiencies to smooth.
    bandpass : float
        Monochromator half-bandpass (nm); the boxcar window is ``2*bandpass``
        wide.

    Returns
    -------
    smoothed_de : 1-D ndarray
        Moving-average of ``de_array`` (same length, ``mode='same'``).
    """
    dw = wav_array[1] - wav_array[0]              # wavelength step
    window_len = max(1, int(2 * bandpass / dw))   # points spanning the bandpass

    boxcar = np.ones(window_len) / window_len
    smoothed_de = np.convolve(de_array, boxcar, mode='same')

    return smoothed_de


def predict(gr_dens, m, dt, land, wavs):
    """
    Predict the bandpass-averaged efficiency at each measurement wavelength.

    At each ``wav`` the surrogate is averaged over ``[wav - BANDPASS,
    wav + BANDPASS]`` to mimic the monochromator's finite spectral resolution.
    If no model samples fall in the window (shouldn't normally happen), the
    nearest model point is used as a fallback. This is the model used in the
    likelihood.

    Parameters
    ----------
    gr_dens : int
        Groove density (800 or 2000).
    m : int
        Diffraction order.
    dt, land : float
        Grating-profile parameters.
    wavs : 1-D ndarray
        Measurement wavelengths (nm).

    Returns
    -------
    preds : 1-D ndarray
        Predicted efficiency (percent) at each wavelength in ``wavs``.
    """
    full_curve = model_de(gr_dens, m, dt, land)
    preds = []
    for wav in wavs:
        mask = (model_wav >= wav - config.BANDPASS) & (model_wav <= wav + config.BANDPASS)
        if mask.sum() > 0:
            preds.append(np.mean(full_curve[mask]))
        else:  # fallback: nearest model point
            preds.append(full_curve[np.argmin(np.abs(model_wav - wav))])
    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# MCMC: priors, likelihood, posterior
# ─────────────────────────────────────────────────────────────────────────────

def log_prior(params):
    """
    Bounded Gaussian log-prior on (dt, land).

    Returns ``-inf`` outside the hard grid bounds (``config.BOUNDS``); otherwise
    the sum of independent Gaussian log-priors (``config.PRIOR_MEANS`` /
    ``config.PRIOR_STDS``).
    """
    dt, land = params

    # 1. Hard grid bounds.
    if not (config.BOUNDS['dt'][0] <= dt <= config.BOUNDS['dt'][1]):
        return -np.inf
    if not (config.BOUNDS['land'][0] <= land <= config.BOUNDS['land'][1]):
        return -np.inf

    # 2. Gaussian prior (sum of log-probabilities).
    lp_dt = -0.5 * ((dt - config.PRIOR_MEANS['dt']) / config.PRIOR_STDS['dt']) ** 2
    lp_land = -0.5 * ((land - config.PRIOR_MEANS['land']) / config.PRIOR_STDS['land']) ** 2

    return lp_dt + lp_land


def log_likelihood(params, gr_dens, data, err):
    """
    Gaussian log-likelihood summed over all orders of a panel.

    For each order the predicted efficiencies are compared with the measured
    efficiencies using the measured per-point errors.
    """
    dt, land = params
    ll = 0.0
    for i, (m, wavs, effs_obs) in enumerate(data):
        effs_model = predict(gr_dens, m, dt, land, wavs)
        ll += -0.5 * np.sum((effs_obs - effs_model) ** 2 / err[i][2] ** 2)
    return ll


def log_prob(params, gr_dens, data, err):
    """Full log-posterior = log_prior + log_likelihood (with prior short-circuit)."""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, gr_dens, data, err)


def run_mcmc(panel, nwalkers=config.NWALKERS, nsteps=config.NSTEPS, burnin=config.BURNIN):
    """
    Run the ensemble sampler for one panel and report the marginal summaries.

    Parameters
    ----------
    panel : str
        Panel group name.
    nwalkers, nsteps, burnin : int
        Sampler configuration (defaults come from ``config`` and match the
        published run).

    Returns
    -------
    sampler : emcee.EnsembleSampler
        The sampler after the production run (burn-in discarded).
    best : (dt_med, land_med)
        Posterior medians of the two parameters.
    """
    gr_dens = config.PANEL_GR_DENS[panel]
    data, err = load_data(panel)

    n_pts = sum(len(d[1]) for d in data)
    print(f"\n{panel} ({gr_dens} gr/mm): {n_pts} data points across {len(data)} orders")

    # Initial walker positions scattered around the prior mean, clipped to bounds.
    p0 = np.array([config.PRIOR_MEANS['dt'], config.PRIOR_MEANS['land']]) \
        + np.array([config.DT_SCATTER, config.LAND_SCATTER]) * np.random.randn(nwalkers, 2)
    p0[:, 0] = np.clip(p0[:, 0], *config.BOUNDS['dt'])
    p0[:, 1] = np.clip(p0[:, 1], *config.BOUNDS['land'])

    sampler = emcee.EnsembleSampler(nwalkers, 2, log_prob, args=(gr_dens, data, err))

    print("Burn-in...")
    state = sampler.run_mcmc(p0, burnin, progress=True)
    sampler.reset()

    print("Production run...")
    sampler.run_mcmc(state, nsteps, progress=True)

    print('Acceptance fraction:', np.mean(sampler.acceptance_fraction))

    try:
        # c=None uses emcee's default automated autocorrelation window.
        tau = sampler.get_autocorr_time()
        print(f"  Autocorr times (steps): dt = {tau[0]:.1f}, facet duty cycle = {tau[1]:.1f}")
    except emcee.autocorr.AutocorrError:
        print("  Autocorr warning: chain too short to reliably estimate autocorrelation time.")

    flat = sampler.get_chain(flat=True)
    dt_med, land_med = np.median(flat, axis=0)
    dt_std, land_std = np.std(flat, axis=0)

    # `land_med` is the fit parameter; report it outwardly as facet duty cycle
    # (= 100 - land). The standard deviation is unchanged by this offset.
    facet_med = 100.0 - land_med

    print(f"\n  dt               = {dt_med:.3f} +/- {dt_std:.3f} deg")
    print(f"  facet duty cycle = {facet_med:.2f}  +/- {land_std:.2f} %")

    return sampler, (dt_med, land_med)


def compute_reduced_chi2(panel, dt, land):
    """
    Per-order reduced chi-squared at a given (dt, land).

    Parameters
    ----------
    panel : str
        Panel group name.
    dt : float
        Best-fit facet asymmetry.
    land : float
        Best-fit land duty cycle (%).

    Returns
    -------
    order_chi2 : dict
        ``{order_int: reduced_chi2}``. The reduced chi-squared uses degrees of
        freedom = number of points in the order (the two free parameters are
        shared globally across all orders, so they are not subtracted per order).
    """
    gr_dens = config.PANEL_GR_DENS[panel]
    data, err = load_data(panel)

    # Map order -> its error array to guarantee alignment with `data`.
    err_lookup = {m: err_effs for m, _, err_effs in err}

    order_chi2 = {}
    for m, wavs, effs_obs in data:
        if m not in err_lookup:
            continue

        effs_model = predict(gr_dens, m, dt, land, wavs)
        effs_err = err_lookup[m]

        chi2 = np.sum(((effs_obs - effs_model) / effs_err) ** 2)
        dof = len(wavs)  # free params shared globally; not subtracted per order
        order_chi2[m] = chi2 / dof if dof > 0 else np.nan

    return order_chi2


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_fit(ax, sampler, panel, dt, land, num_samples=300):
    """
    Overplot the best-fit model on the data with 1-sigma uncertainty envelopes.

    Draws, per order: the measured points with error bars, the best-fit
    (bandpass-smoothed) model curve, and a shaded 16th-84th percentile envelope
    built from ``num_samples`` random posterior draws.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes (used by the keystone figure to place each panel).
    sampler : emcee.EnsembleSampler
        Sampler whose flat chain provides the posterior draws.
    panel : str
        Panel group name (controls the legend proxy artists).
    dt, land : float
        Best-fit parameters for the central model curve.
    num_samples : int, optional
        Number of posterior draws for the envelope. Default 300.
    """
    gr_dens = config.PANEL_GR_DENS[panel]
    data, err = load_data(panel)

    flat_samples = sampler.get_chain(flat=True)
    inds = np.random.randint(len(flat_samples), size=num_samples)

    colors = {-2: 'blue', -1: 'green', 0: 'red', 1: 'green', 2: 'blue'}
    markers = {-2: '*', -1: '*', 0: 'o', 1: 'v', 2: 'v'}
    markersizes = {-2: 7, -1: 7, 0: 5, 1: 5, 2: 5}

    wav_fine = model_wav

    # Legend proxy artists (data markers for panel1, model lines for panel2).
    if panel == 'panel1':
        for m in [-2, -1, 0, 2, 1]:
            ax.errorbar([], [], c=sc[colors[m]], fmt='o', marker=markers[m],
                        markersize=markersizes[m], label=f'$m={m:+d}$ (obs.)', zorder=4)
    elif panel == 'panel2':
        for m in [-2, -1, 0, 2, 1]:
            if m < 0:
                ax.plot([], [], c=sc[colors[m]], lw=0.5, zorder=3, dashes=(8, 4),
                        label=f'$m={m:+d}$ (mod.)')
            else:
                ax.plot([], [], c=sc[colors[m]], lw=0.5, zorder=3,
                        label=f'$m={m:+d}$ (mod.)')

    for i, (m, wavs, effs_obs) in enumerate(data):

        ax.errorbar(wavs, effs_obs, yerr=err[i][2],
                    c=sc[colors[m]], fmt='o', marker=markers[m],
                    markersize=markersizes[m], zorder=4)

        best_curve = smooth_model(wav_fine, model_de(gr_dens, m, dt, land), config.BANDPASS)
        if m < 0:
            ax.plot(wav_fine, best_curve, c=sc[colors[m]], lw=0.5, zorder=3, dashes=(8, 4))
        else:
            ax.plot(wav_fine, best_curve, c=sc[colors[m]], lw=0.5, zorder=3)

        # 16th-84th percentile envelope from random posterior draws.
        model_realizations = np.zeros((num_samples, len(wav_fine)))
        for j, ind in enumerate(inds):
            sample_dt, sample_land = flat_samples[ind]
            model_realizations[j] = smooth_model(
                wav_fine, model_de(gr_dens, m, sample_dt, sample_land), config.BANDPASS
            )

        lower_bound = np.percentile(model_realizations, 16, axis=0)
        upper_bound = np.percentile(model_realizations, 84, axis=0)

        ax.fill_between(wav_fine, lower_bound, upper_bound,
                        color=sc[colors[m]], alpha=0.2, zorder=2, lw=0)

        ax.grid()


def plot_combined_mcmc(sampler, panel, figsize=(14, 5), width_ratios=(1, 1.5)):
    """
    Save the per-panel corner + trace diagnostic figure.

    The figure is split into two independent sub-figures so the ``corner`` plot
    (left) and the trace/chain plots (right) do not interfere. Saved to
    ``<PLOTS_DIR>/mcmc_app_<panel>.pdf``.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Sampler to visualise.
    panel : str
        Panel group name (used in the output filename).
    figsize : tuple, optional
        Overall figure size.
    width_ratios : tuple, optional
        Width ratio of the corner sub-figure to the trace sub-figure.
    """
    fig = plt.figure(figsize=figsize)

    # Two independent sub-figures so `corner` is strictly confined to the left.
    subfigs = fig.subfigures(1, 2, width_ratios=width_ratios, wspace=0.05)

    # Convert the land duty cycle (fit parameter) to facet duty cycle (= 100 - land)
    # for display only. Transforming the samples keeps the corner-plot titles,
    # quantiles, and trace axes all in facet-duty-cycle units, with the asymmetric
    # error bars correctly reflected.
    flat_chain = sampler.get_chain(flat=True).copy()
    full_chain = sampler.get_chain().copy()
    flat_chain[:, 1] = 100.0 - flat_chain[:, 1]
    full_chain[:, :, 1] = 100.0 - full_chain[:, :, 1]
    labels = [r'$\Delta\theta$ (deg)', '$D$ (\%)']

    # LEFT: corner plot.
    fig2 = corner.corner(
        flat_chain,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f',
        fig=subfigs[0],
        label_kwargs={"fontsize": 10},
        title_kwargs={"fontsize": 10},
    )
    for ax in fig2.get_axes():
        ax.tick_params(axis='both', labelsize=10)

    # RIGHT: trace / chain plots.
    axs = subfigs[1].subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.1})
    for i, ax in enumerate(axs):
        ax.plot(full_chain[:, :, i], c='k', alpha=0.15, lw=0.5)
        ax.set_ylabel(labels[i], fontsize=10)

    axs[-1].set_xlabel('Step', fontsize=10)
    axs[0].tick_params(axis='both', labelsize=10)
    axs[1].tick_params(axis='both', labelsize=10)
    plt.setp(axs[0].get_xticklabels(), visible=False)  # cleaner top panel

    subfigs[0].text(0.02, 0.98, '(a)', fontsize=12, va='top', ha='left')
    subfigs[1].text(-0.02, 0.98, '(b)', fontsize=12, va='top', ha='left')

    plt.savefig(os.path.join(config.PLOTS_DIR, f'mcmc_app_{panel}.pdf'),
                dpi=600, bbox_inches='tight')
    plt.show()


def make_keystone_figure(results, num_samples=1000):
    """
    Build the stacked best-fit efficiency figure for all three panels.

    This reproduces the main-text "keystone" figure: one row per panel, each
    showing the measured efficiencies and the best-fit model with its
    uncertainty envelope. Disabled by default (see ``MAKE_KEYSTONE``).

    Parameters
    ----------
    results : dict
        ``{panel: (sampler, (dt_best, land_best))}`` produced by the main loop.
    num_samples : int, optional
        Posterior draws per order for the uncertainty envelopes. Default 1000.
    """
    _, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    panels = ['panel1', 'panel2', 'panel3']
    for i, panel in enumerate(panels):
        sampler, (dt_best, land_best) = results[panel]
        plot_fit(axs[i], sampler, panel, dt_best, land_best, num_samples=num_samples)

    axs[2].set_xlabel('Wavelength (nm)', fontsize=9)
    for i in range(3):
        axs[i].set_ylabel('Diffraction Efficiency (\%)', fontsize=9)
    for i in range(2):
        axs[i].legend(loc=2, ncol=2, frameon=False, fontsize=9)

    axs[0].text(760, 7.5, r'Panel 1, 800 gr mm$^{-1}$', rotation=-90, fontsize=9)
    axs[1].text(760, 9, r'Panel 2, 2000 gr mm$^{-1}$', rotation=-90, fontsize=9)
    axs[2].text(760, 7.5, r'Panel 3, 800 gr mm$^{-1}$', rotation=-90, fontsize=9)

    axs[2].set_xlim([150, 750])
    axs[0].set_ylim([-2, 60])
    axs[1].set_ylim([-2, 100])
    axs[2].set_ylim([-2, 60])

    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'mcmc_keystone.pdf'), dpi=600)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Fit every panel, print summaries, and save the diagnostic figures."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    if config.RNG_SEED is not None:
        np.random.seed(config.RNG_SEED)

    results = {}
    for panel in ['panel1', 'panel2', 'panel3']:
        sampler, (dt_best, land_best) = run_mcmc(panel)

        chi2_breakdown = compute_reduced_chi2(panel, dt_best, land_best)
        print(f"  Reduced Chi2 Breakdown for {panel}:")
        for m, r_chi2 in chi2_breakdown.items():
            print(f"    Order m = {m:+d}: Reduced Chi2 = {r_chi2:.3f}")

        plot_combined_mcmc(sampler, panel, figsize=(9, 4), width_ratios=[1, 1.5])

        results[panel] = (sampler, (dt_best, land_best))

    if MAKE_KEYSTONE:
        make_keystone_figure(results)


if __name__ == "__main__":
    main()
