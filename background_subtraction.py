"""
background_subtraction.py
=========================
Stray-light (background) subtraction for the FUV beam images measured on the
CCD. Used by ``step1_compute_efficiencies.py`` to clean each raw frame before
integrating it to a count rate.

Why this is needed
------------------
The stray-light background is spatially smooth but *not* uniform, so a single
scalar (dark) subtraction would leave a residual gradient across the frame.
Instead we fit and remove a smooth 2-D polynomial surface estimated from the
pixels that do not contain beam light.

Algorithm (the ``sub_bg`` automatic path)
-----------------------------------------
1. Estimate the background level from the image *corners* (most likely to be
   beam-free) using a robust median + MAD estimator.
2. Run Otsu thresholding to find the bright beam core.
3. Grow that core outward via HYSTERESIS THRESHOLDING: any pixel connected to
   the core and above a lower, corner-based threshold is also labelled beam.
   This captures beams with strong internal gradients whose dim end would
   otherwise be mistaken for background.
4. Dilate the final mask to push the exclusion boundary off the beam edges.
5. Fit a 2-D polynomial to the surviving background pixels (ordinary least
   squares).
6. Subtract the fitted surface from the original image.
7. Optionally repeat 1-6 so the mask refines on progressively cleaner residuals.

Because the polynomial model is smooth by construction, it does not absorb the
high-frequency grating features (grooves, fringes) on the beam itself.

A manual override path is also provided: pass ``beam_mask`` to skip detection
entirely and use a hand-drawn exclusion region (see ``sub_bg``).

Public API
----------
``sub_bg`` is the only function intended for external use.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, gaussian_filter, label
from skimage.filters import threshold_otsu


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers — beam detection
# ─────────────────────────────────────────────────────────────────────────────

def _corner_background(image, corner_frac=0.08):
    """
    Robust background estimate from the four image corners.

    Uses median + 1.4826 * MAD so that a few hot pixels, or a beam that grazes
    a single corner, do not dominate the estimate.

    Parameters
    ----------
    image : 2-D float ndarray
        Frame to estimate from (should already be Gaussian-smoothed).
    corner_frac : float, optional
        Side length of each square corner patch, as a fraction of the image
        size in each dimension. Default 0.08.

    Returns
    -------
    median : float
        Robust background median over the corner pixels.
    sigma : float
        MAD-derived Gaussian-equivalent sigma, floored at 1.0 to guard against
        a degenerate (zero-spread) estimate.
    """
    ny, nx = image.shape
    ch = max(int(ny * corner_frac), 8)
    cw = max(int(nx * corner_frac), 8)

    corners = [
        image[:ch, :cw],    # top-left
        image[:ch, -cw:],   # top-right
        image[-ch:, :cw],   # bottom-left
        image[-ch:, -cw:],  # bottom-right
    ]

    values = np.concatenate([c.ravel() for c in corners])
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    sigma = mad * 1.4826  # convert MAD -> equivalent Gaussian sigma

    return float(med), max(float(sigma), 1.0)  # guard against sigma = 0


def _hysteresis_beam_mask(smoothed, thresh_high, thresh_low):
    """
    Hysteresis thresholding for beam detection.

    Seeds from pixels above ``thresh_high`` (the bright core), then keeps every
    connected component of the ``thresh_low`` mask that contains at least one
    core pixel. This grows the detection from the bright core into the dim beam
    wings / internal gradient while rejecting isolated noise blobs.

    Parameters
    ----------
    smoothed : 2-D float ndarray
        Gaussian-smoothed frame.
    thresh_high : float
        Bright-core threshold (e.g. the Otsu level).
    thresh_low : float
        Lower bound used for connectivity growth.

    Returns
    -------
    beam_mask : 2-D bool ndarray
        True where beam light is detected.
    """
    core_mask = smoothed > thresh_high
    extended_mask = smoothed > thresh_low

    # Label all connected components in the extended (low-threshold) mask.
    labeled, _ = label(extended_mask)

    # Keep only components that contain at least one core pixel.
    core_labels = set(labeled[core_mask].ravel()) - {0}

    beam_mask = np.zeros(smoothed.shape, dtype=bool)
    for lbl in core_labels:
        beam_mask |= (labeled == lbl)

    return beam_mask


def _estimate_beam_mask(image, dilation_iters, sigma_smooth, k_low=4.0):
    """
    Detect the beam and build a dilated background-exclusion mask.

    Parameters
    ----------
    image : 2-D float ndarray
        Frame to analyse.
    dilation_iters : int
        Number of binary-dilation steps used to grow the buffer around the
        detected beam (in pixels).
    sigma_smooth : float
        Gaussian smoothing sigma applied before thresholding.
    k_low : float, optional
        Sets the low hysteresis threshold as ``bg_median + k_low * bg_sigma``.
        Lower values catch fainter beam wings; raise it if noisy images produce
        false detections. Default 4.0.

    Returns
    -------
    tight : 2-D bool ndarray
        The detected beam footprint.
    exclusion : 2-D bool ndarray
        ``tight`` dilated by ``dilation_iters`` to create a buffer zone; this
        is the region excluded from the background fit.
    thresh_high, thresh_low : float
        The high (Otsu) and low (corner-based) thresholds used.
    bg_med, bg_sig : float
        Corner-based background median and sigma.
    """
    smoothed = gaussian_filter(image.astype(float), sigma=sigma_smooth)

    # High threshold: Otsu on the smoothed image (isolates the bright core).
    thresh_high = threshold_otsu(smoothed)

    # Low threshold: corner-based background level plus k_low sigma.
    bg_med, bg_sig = _corner_background(smoothed)
    thresh_low = bg_med + k_low * bg_sig

    # thresh_low must sit below thresh_high to be useful. If the corners are
    # contaminated (e.g. the beam touches an edge), fall back to a fraction of
    # the Otsu threshold instead.
    if thresh_low >= thresh_high:
        thresh_low = 0.25 * thresh_high + 0.75 * bg_med

    # Hysteresis growth from bright core into dim wings.
    tight = _hysteresis_beam_mask(smoothed, thresh_high, thresh_low)

    # Dilate to create the background exclusion buffer.
    exclusion = binary_dilation(tight, iterations=dilation_iters)

    return tight, exclusion, thresh_high, thresh_low, bg_med, bg_sig


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers — polynomial background fit
# ─────────────────────────────────────────────────────────────────────────────

def _poly2d_matrix(xn, yn, degree):
    """
    Build the 2-D polynomial design matrix.

    Includes every monomial ``x**a * y**b`` with ``a + b <= degree``.

    Parameters
    ----------
    xn, yn : 1-D float ndarray
        Normalised pixel coordinates (each in roughly [-1, 1]).
    degree : int
        Maximum total polynomial degree.

    Returns
    -------
    A : 2-D float ndarray
        Design matrix of shape (n_points, n_terms).
    """
    cols = []
    for d in range(degree + 1):
        for a in range(d + 1):
            cols.append(xn ** a * yn ** (d - a))
    return np.column_stack(cols)


def _fit_bg_poly2d(image, exclusion_mask, degree=3):
    """
    Fit a smooth 2-D polynomial to the background pixels (``~exclusion_mask``).

    Pixel coordinates are normalised to [-1, 1] before fitting for numerical
    stability. The fit is rejected (``fit_ok = False``) if too few background
    pixels survive the mask.

    Parameters
    ----------
    image : 2-D float ndarray
        Frame to fit.
    exclusion_mask : 2-D bool ndarray
        True where pixels are *excluded* from the fit (i.e. the beam buffer).
    degree : int, optional
        Polynomial degree. Default 3.

    Returns
    -------
    bg_model : 2-D float ndarray
        Polynomial surface evaluated on every pixel.
    residuals : 1-D float ndarray
        Fit residuals on the background pixels (a single-element zero array if
        the fit was rejected).
    fit_ok : bool
        False when there were too few background pixels to fit.
    """
    ny, nx = image.shape
    y_idx, x_idx = np.indices((ny, nx), dtype=float)
    xn = (x_idx - nx / 2.0) / (nx / 2.0)
    yn = (y_idx - ny / 2.0) / (ny / 2.0)

    bg_pix = ~exclusion_mask
    n_bg = bg_pix.sum()
    n_terms = sum(d + 1 for d in range(degree + 1))

    # Require a healthy excess of background pixels over free parameters.
    if n_bg < max(20, 3 * n_terms):
        return np.zeros_like(image, dtype=float), np.array([0.0]), False

    xfit = xn[bg_pix].ravel()
    yfit = yn[bg_pix].ravel()
    zfit = image[bg_pix].ravel().astype(float)

    A_fit = _poly2d_matrix(xfit, yfit, degree)
    A_full = _poly2d_matrix(xn.ravel(), yn.ravel(), degree)

    coeffs, _, _, _ = np.linalg.lstsq(A_fit, zfit, rcond=None)

    bg_model = (A_full @ coeffs).reshape(ny, nx)
    residuals = zfit - (A_fit @ coeffs)

    return bg_model.astype(float), residuals, True


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def sub_bg(image,
           poly_degree=3,
           dilation_iters=20,
           n_iter=1,
           sigma_smooth=2.0,
           k_low=1.0,
           beam_mask=None,
           verbose=False):
    """
    Subtract a smooth 2-D polynomial stray-light background from a beam image.

    Two modes are supported:

    * **Automatic** (default): the beam is detected and masked automatically
      (see the module docstring for the algorithm), optionally iterating
      ``n_iter`` times so the mask refines on progressively cleaner residuals.
    * **Manual override**: if ``beam_mask`` is supplied, automatic detection is
      skipped entirely and the supplied mask is used *as-is* as the exclusion
      zone (no dilation or modification). In this mode ``n_iter``,
      ``dilation_iters``, ``sigma_smooth`` and ``k_low`` are all ignored.

    Parameters
    ----------
    image : 2-D array_like
        Raw beam frame (digital numbers). Cast to float internally.
    poly_degree : int, optional
        Degree of the 2-D background polynomial. Default 3.
    dilation_iters : int, optional
        Pixels by which the detected beam mask is dilated to form the
        background-exclusion buffer (automatic mode only). Default 20.
    n_iter : int, optional
        Number of detect-fit-subtract iterations (automatic mode only).
        Default 1.
    sigma_smooth : float, optional
        Gaussian smoothing sigma applied before thresholding (automatic mode
        only). Default 2.0.
    k_low : float, optional
        Low-threshold factor passed to the detector (automatic mode only);
        ``thresh_low = bg_median + k_low * bg_sigma``. Default 1.0.
    beam_mask : 2-D bool array_like, optional
        External beam mask (True = beam / exclude from fit). Supplying this
        triggers the manual override path described above. Default None.
    verbose : bool, optional
        If True, show the six-panel diagnostic figure. Default False.

    Returns
    -------
    sub : 2-D float ndarray
        Background-subtracted image (``image - bg_model``).
    bg_model : 2-D float ndarray
        The fitted background surface.
    mask_tight : 2-D bool ndarray
        The detected (or supplied) beam footprint.
    residuals : 1-D float ndarray
        Background-fit residuals on the background pixels.
    bg_contribution : float
        Fraction of the in-beam counts attributable to the fitted background,
        ``sum(bg_model[beam]) / sum(image[beam])``. (This is the quantity quoted
        as the "fitted background over the beam footprint" percentage.)
    """
    image = np.asarray(image, dtype=float)

    # ── Manual override path: use the supplied mask verbatim ────────────────
    if beam_mask is not None:
        mask_tight = np.asarray(beam_mask, dtype=bool)
        mask_exclusion = mask_tight  # used as-is, no dilation
        thresh_info = {}

        bg_model, residuals, fit_ok = _fit_bg_poly2d(
            image, mask_exclusion, degree=poly_degree
        )
        if not fit_ok:
            bg_model = np.zeros_like(image)
            residuals = np.array([0.0])

        sub = image - bg_model

        if verbose:
            _diagnostic_plot(image, mask_tight, mask_exclusion,
                             bg_model, sub, residuals, thresh_info)

        bg_contribution = np.nansum(bg_model[mask_tight]) / np.nansum(image[mask_tight])
        return sub, bg_model, mask_tight, residuals, bg_contribution

    # ── Automatic detection path ────────────────────────────────────────────
    sub = image.copy()
    mask_tight = None
    mask_exclusion = None
    bg_model = np.zeros_like(image)
    residuals = np.array([0.0])
    thresh_info = {}

    for _ in range(n_iter):
        tight, exclusion, t_high, t_low, bg_med, bg_sig = _estimate_beam_mask(
            sub, dilation_iters, sigma_smooth, k_low=k_low
        )

        # If almost nothing survives as background, retry with a smaller buffer.
        bg_frac = (~exclusion).mean()
        if bg_frac < 0.05:
            fallback = max(dilation_iters // 3, 3)
            tight, exclusion, t_high, t_low, bg_med, bg_sig = _estimate_beam_mask(
                sub, fallback, sigma_smooth, k_low=k_low
            )
            if (~exclusion).mean() < 0.02:
                break

        mask_tight = tight
        mask_exclusion = exclusion
        thresh_info = dict(t_high=t_high, t_low=t_low,
                           bg_med=bg_med, bg_sig=bg_sig)

        bg_model, residuals, fit_ok = _fit_bg_poly2d(
            image, mask_exclusion, degree=poly_degree
        )
        if not fit_ok:
            break

        sub = image - bg_model

    if verbose and mask_tight is not None:
        _diagnostic_plot(image, mask_tight, mask_exclusion,
                         bg_model, sub, residuals, thresh_info)

    bg_contribution = np.nansum(bg_model[mask_tight]) / np.nansum(image[mask_tight])
    return sub, bg_model, mask_tight, residuals, bg_contribution


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic plot (only drawn when sub_bg is called with verbose=True)
# ─────────────────────────────────────────────────────────────────────────────

def _diagnostic_plot(image, mask_tight, mask_exclusion,
                     bg_model, sub, residuals, thresh_info):
    """
    Six-panel diagnostic figure for a single frame.

    Layout::

        [0,0] Original image with centre cross-hairs
        [0,1] Beam mask + exclusion zone + threshold annotations
        [0,2] Fitted background model
        [1,0] Background-subtracted image
        [1,1] Row profile through the frame centre
        [1,2] Column profile through the frame centre + residual histogram inset

    This is a visual sanity check only; it has no effect on the returned values.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Background Subtraction — Diagnostics", fontsize=13)

    ny, nx = image.shape
    cy, cx = ny // 2, nx // 2
    col_x = np.arange(nx)
    row_y = np.arange(ny)

    vlo, vhi = np.percentile(image, [30, 75])
    slo, shi = np.percentile(sub, [1, 99])
    kw = dict(origin="lower", aspect="auto", interpolation="nearest")

    # ── [0,0]  Original ─────────────────────────────────────────────────────
    ax = axes[0, 0]
    im = ax.imshow(image, cmap="inferno", vmin=vlo, vmax=vhi, **kw)
    ax.axhline(cy, color="cyan", lw=0.7, ls="--", alpha=0.7)
    ax.axvline(cx, color="lime", lw=0.7, ls="--", alpha=0.7)
    ax.set_title("Original")
    plt.colorbar(im, ax=ax, label="DN", fraction=0.046)

    # ── [0,1]  Masks ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    overlay = np.zeros((*image.shape, 3), dtype=float)
    overlay[mask_tight, :] = [1.0, 1.0, 1.0]                  # white = beam
    overlay[mask_exclusion & ~mask_tight, :] = [0.8, 0.4, 0.0]  # orange = buffer
    ax.imshow(overlay, origin="lower", aspect="auto", interpolation="nearest")

    t_high = thresh_info.get("t_high", float("nan"))
    t_low = thresh_info.get("t_low", float("nan"))
    bg_med = thresh_info.get("bg_med", float("nan"))
    bg_sig = thresh_info.get("bg_sig", float("nan"))
    bg_frac = (~mask_exclusion).mean() * 100

    ax.set_title("Beam mask\n"
                 "white = beam   orange = exclusion zone   black = BG fit region")
    ax.set_xlabel(
        f"BG fit region: {bg_frac:.1f}%  |  "
        f"thresh_high (Otsu): {t_high:.0f}  |  "
        f"thresh_low: {t_low:.0f}  |  "
        f"corner BG: {bg_med:.0f} ± {bg_sig:.0f} DN",
        fontsize=8,
    )

    # ── [0,2]  Background model ─────────────────────────────────────────────
    ax = axes[0, 2]
    im2 = ax.imshow(bg_model, cmap="inferno", vmin=vlo, vmax=vhi, **kw)
    ax.set_title("Fitted background (polynomial)")
    plt.colorbar(im2, ax=ax, label="DN", fraction=0.046)

    # ── [1,0]  Subtracted image ─────────────────────────────────────────────
    ax = axes[1, 0]
    im3 = ax.imshow(sub, cmap="inferno", vmin=slo, vmax=shi, **kw)
    ax.axhline(cy, color="cyan", lw=0.7, ls="--", alpha=0.7)
    ax.axvline(cx, color="lime", lw=0.7, ls="--", alpha=0.7)
    ax.set_title("Background-subtracted")
    plt.colorbar(im3, ax=ax, label="DN", fraction=0.046)

    # ── [1,1]  Row profile ──────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(col_x, image[cy, :], lw=1.0, color="C0", label="Original")
    ax.plot(col_x, bg_model[cy, :], lw=1.5, color="C1", ls="--", label="BG model")
    ax.plot(col_x, sub[cy, :], lw=1.0, color="C2", label="Subtracted")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.axhline(t_low, color="magenta", lw=0.8, ls=":", alpha=0.7,
               label=f"thresh_low={t_low:.0f}")
    ax.axhline(t_high, color="red", lw=0.8, ls=":", alpha=0.7,
               label=f"thresh_high={t_high:.0f}")
    _shade_mask(ax, col_x, mask_exclusion[cy, :], color="gray",
                alpha=0.15, label="Exclusion zone")
    ax.set_title(f"Row profile  (row {cy})")
    ax.set_xlabel("Column [px]")
    ax.set_ylabel("DN")
    ax.legend(fontsize=7, loc="upper left")

    # ── [1,2]  Column profile + residual histogram inset ────────────────────
    ax = axes[1, 2]
    ax.plot(row_y, image[:, cx], lw=1.0, color="C0", label="Original")
    ax.plot(row_y, bg_model[:, cx], lw=1.5, color="C1", ls="--", label="BG model")
    ax.plot(row_y, sub[:, cx], lw=1.0, color="C2", label="Subtracted")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.axhline(t_low, color="magenta", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(t_high, color="red", lw=0.8, ls=":", alpha=0.7)
    _shade_mask(ax, row_y, mask_exclusion[:, cx], color="gray",
                alpha=0.15, label="Exclusion zone")
    ax.set_title(f"Column profile  (col {cx})")
    ax.set_xlabel("Row [px]")
    ax.set_ylabel("DN")
    ax.legend(fontsize=7, loc="upper right")

    # Residual histogram inset.
    ax_in = ax.inset_axes([0.03, 0.52, 0.40, 0.40])
    ax_in.hist(residuals, bins=60, color="C0", density=True,
               edgecolor="none", alpha=0.8)
    ax_in.axvline(0, color="k", lw=1)
    ax_in.set_title("BG residuals", fontsize=7)
    ax_in.tick_params(labelsize=6)
    mu, sigma_r = residuals.mean(), residuals.std()
    ax_in.text(0.97, 0.93, f"μ={mu:.1f}\nσ={sigma_r:.1f}",
               transform=ax_in.transAxes, ha="right", va="top",
               fontsize=6, bbox=dict(fc="white", alpha=0.7, pad=1))

    plt.show()


def _shade_mask(ax, x, bool_arr, **kw):
    """Shade vertical bands on ``ax`` wherever ``bool_arr`` is True."""
    in_band, x0 = False, None
    for i, flag in enumerate(bool_arr):
        if flag and not in_band:
            x0, in_band = x[i], True
        elif not flag and in_band:
            ax.axvspan(x0, x[i - 1], **kw)
            in_band = False
    if in_band:
        ax.axvspan(x0, x[-1], **kw)
