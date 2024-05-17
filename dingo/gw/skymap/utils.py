"""Util functions for skymap conversions and analysis"""
import numpy as np
import healpy as hp
import pandas as pd

from ligo.skymap.postprocess import crossmatch, find_greedy_credible_levels
from ligo.skymap.moc import rasterize


def skymap_from_uniq_to_nested(skymap, normalize=True):
    """Convert skymap from unique identifier convention to nested convention.

    See https://healpix.sourceforge.io/html/intro_Geometric_Algebraic_Propert.htm.
    HEALPix pixels are defined by three quantities:

        nside: resolution, = 2 ** order
        p: pixel index

    For a fixed nside, the pixel values can simply be provided as a list. The length of
    the list determines nside [nside = hp.npix2nside(len(pixels))] and each position in
    the list corresponds to a specific position. There are two conventions for the
    positions, Ring and Nested.

    In addition to the Ring and Nested conventions, there is the Unique Identifier
    scheme, in which the resolution nside and pixel position p are mergerd into a
    single unique identifier u.

        u = p + 4 * nside ** 2

    With this convention, the pixel position is no longer defined by the position in
    the list, but rather by the unique id u. This is useful when specifying the pixels
    at multiple different resolutions (or for sparse maps).

    ligo.skymap often uses the unique identifier scheme and this function converts from
    this convention to nested convention.

    Parameters
    ----------
    skymap: input skymap in unique identifier convention
    normalize: if True, normalize skymap to sum 1.

    Returns
    -------
    skymap: output skymap as numpy array in nested convention
    """
    if "UNIQ" not in skymap.columns:
        raise ValueError("Skymap convention must be unique identifier scheme.")

    # step 1: rasterize to remove multi-order pixels
    # pixel labeling: UNIQ ids => Ring
    skymap = pd.DataFrame(rasterize(skymap))
    skymap = np.array(skymap["PROBDENSITY"])

    # step 2: convert from ring to nested pixel labels
    nside = hp.npix2nside(len(skymap))
    reordering = hp.ring2nest(nside, np.arange(len(skymap)))
    skymap = skymap[reordering]

    # step 3: optionally normalize
    if normalize:
        skymap /= skymap.sum()

    return skymap


def credible_areas(skymap, credible_levels):
    """Compute areas (in deg^2) at the specified credible levels."""
    if np.min(credible_levels) < 0 or np.max(credible_levels) > 1:
        raise ValueError(f"Contours must be in range [0, 1], got {credible_levels}")
    return crossmatch(skymap, contours=credible_levels).contour_areas


def coverage(skymap_proposal, skymap_reference, credible_levels):
    """Compute skymap_proposal's coverage fraction of skymap_reference at credible_levels.

    For each credible level p, this computes the fraction f of skymap_reference that is
    covered by the p-credible-area of skymap_proposal.

    For a perfect overlap between proposal and reference skymap, we expect f = p.

    Parameters
    ----------
    skymap_proposal: proposal skymap
    skymap_reference: reference skymap
    credible_levels: credible levels at which to evaluate coverage

    Returns
    -------
    coverages: skymap_proposal's coverage fraction of skymap_reference
    """
    if np.min(credible_levels) < 0 or np.max(credible_levels) > 1:
        raise ValueError(f"Contours must be in range [0, 1], got {credible_levels}")

    # convert to nested convention
    if "UNIQ" in skymap_proposal.columns:
        skymap_proposal = skymap_from_uniq_to_nested(skymap_proposal)
    if "UNIQ" in skymap_reference.columns:
        skymap_reference = skymap_from_uniq_to_nested(skymap_reference)

    # find credible levels
    credible_levels_proposal = find_greedy_credible_levels(skymap_proposal)
    credible_levels_reference = find_greedy_credible_levels(skymap_reference)

    # compute coverage of credible areas of skymap_reference at specified credible_levels
    # by skymap_proposal
    coverages = []
    coverages_proposal = []  # sanity check
    for contour in credible_levels:
        coverages.append(
            np.sum(skymap_reference * (credible_levels_proposal <= contour))
        )
        coverages_proposal.append(
            np.sum(skymap_reference * (credible_levels_reference <= contour))
        )

    # check that proposal coverages are close to input credible_levels
    deviation = np.array(coverages_proposal) - np.array(credible_levels)
    assert np.max(np.abs(deviation)) < 1e-3, (coverages, coverages_proposal)

    return coverages


def credible_levels_at_position(skymap, ra, dec):
    """Compute credible level at which position [ra, dec] is covered by skymap.

    Parameters
    ----------
    skymap: skymap with estimated position
    ra: ra coordinate in rad, array or float
    dec: dec coordinate in rad, array or float

    Returns
    -------
    credible_level: credible level at which [ra, dec] is covered by skymap, array or float
    """
    # convert to nested convention
    if "UNIQ" in skymap.columns:
        skymap = skymap_from_uniq_to_nested(skymap)
    # compute pixel id corresponding to [ra, dec]
    nside = hp.npix2nside(len(skymap))
    ipix = hp.ang2pix(nside, 0.5 * np.pi - dec, ra)  # ra, dec in rad!
    # find credible levels
    credible_levels = find_greedy_credible_levels(skymap)
    # return credible level at pixel location of [ra, dec]
    return credible_levels[ipix]
