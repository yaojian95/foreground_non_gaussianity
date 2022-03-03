import numpy as np
import healpy as hp
import pymaster as nmt

minmax = lambda m, a, b: a + (b - a) * (m - m.min()) / (m.max() - m.min())

def log_pol_tens_to_map(log_pol_tens):
    P = np.sqrt(log_pol_tens[1] ** 2 + log_pol_tens[2] ** 2)
    m = np.empty_like(log_pol_tens)
    exp_i = np.exp(log_pol_tens[0])
    m[0] = exp_i * np.cosh(P)
    m[1:] = log_pol_tens[1:] / P * exp_i * np.sinh(P)
    return m

def map_to_log_pol_tens(m):
    P = np.sqrt(m[1] ** 2 + m[2] ** 2)
    log_pol_tens = np.empty_like(m)
    log_pol_tens[0] = np.log(m[0] ** 2 - P ** 2) / 2.0
    log_pol_tens[1:] = m[1:] / P * np.log((m[0] + P) / (m[0] - P)) / 2.0
    return log_pol_tens

def rescale_min_max(img, a=-1, b=1, return_min_max=False):
    img_resc = (b-a)*(img-np.min(img))/(np.max(img)-np.min(img))+a
    if return_min_max:
        return img_resc, np.min(img), np.max(img)
    else:
        return img_resc

def sigmoid(x, x0, width, power=4):
    """Sigmoid function given start point and width
    Parameters
    ----------
    x : array
        input x axis
    x0 : float
        value of x where the sigmoid starts (not the center)
    width : float
        width of the transition region in unit of x
    power : float
        tweak the steepness of the curve
    Returns
    -------
    sigmoid : array
        sigmoid, same length of x"""
    return 1.0 / (1 + np.exp(-power * (x - x0 - width / 2) / width))
  
def get_large_scales(maps_input, lmax, spectra_components, ells,output_nside):
  
    alm_IQU_fullsky = hp.map2alm(maps_input, lmax=lmax, use_pixel_weights=True)
    LS_alm = np.empty_like(alm_IQU_fullsky)
    ell = np.arange(lmax + 1)

    for ii, pol in enumerate(spectra_components):

        sig_func = sigmoid(ell, x0=ells[pol], width=ells[pol] / 10)

        LS_alm[ii] = hp.almxfl(alm_IQU_fullsky[ii], (1.0 - sig_func) ** 0.2) 

    ls = hp.alm2map(LS_alm, nside=output_nside)

    return ls

def get_small_scales(maps_input, lmax, spectra_components, ells,output_nside):
  
    alm_IQU_fullsky = hp.map2alm(maps_input, lmax=lmax, use_pixel_weights=True)
    SS_alm = np.empty_like(alm_IQU_fullsky)
    ell = np.arange(lmax + 1)

    for ii, pol in enumerate(spectra_components):

        sig_func = sigmoid(ell, x0=ells[pol], width=ells[pol] / 10)

        SS_alm[ii] = hp.almxfl(alm_IQU_fullsky[ii], sig_func) 

    ss = hp.alm2map(SS_alm, nside=output_nside)

    return ss

def run_anafast(m, lmax):
    """
    two cases: 1) IQU; 2) I only
    """
    clanaf = hp.anafast(m, lmax=lmax)
    cl = {}
    
    if len(m) == 3: 
        cl["TT"] = clanaf[0]; cl["EE"] = clanaf[1]
        cl["BB"] = clanaf[2]; cl["TE"] = clanaf[3]
    
    elif m.ndim == 1:
        cl["TT"] = clanaf
    ell = np.arange(lmax + 1)

    cl_norm = ell * (ell + 1) / np.pi / 2
    cl_norm[0] = 1
    return ell, cl_norm, cl

def run_namaster(m, mask, lmax, nlbins=1):
    """
    Compute C_ell with NaMaster
    Returns
    -------
    ell : numpy array
        array of ell from 0 to lmax (length lmax+1)
    cl_norm : numpy array
        ell (ell+1)/2pi factor to turn C_ell into D_ell
        first element is set to 1
    cl : dict of numpy arrays
        dictionary of numpy arrays with all components
        of the spectra, for now only II, EE, BB, no
        cross-spectra
    """
    nside = hp.npix2nside(len(mask))

    binning = nmt.NmtBin(nside=nside, nlb=nlbins, lmax=lmax, is_Dell=False)

    cl = {}

    if len(m) == 3:
        f_0 = nmt.NmtField(mask, [m[0]])
        f_2 = nmt.NmtField(mask, m[1:].copy())  # NaMaster masks the map in-place
        cl_namaster = nmt.compute_full_master(f_2, f_2, binning)
        cl["EE"] = np.concatenate([[0, 0], cl_namaster[0]])
        cl["BB"] = np.concatenate([[0, 0], cl_namaster[3]])
        cl_namaster = nmt.compute_full_master(f_0, f_2, binning)
        cl["TE"] = np.concatenate([[0, 0], cl_namaster[0]])
    elif m.ndim == 1:
        m = m.reshape((1, -1))
        f_0 = nmt.NmtField(mask, [m[0]])

    cl_namaster_I = nmt.compute_full_master(f_0, f_0, binning)

    cl["TT"] = np.concatenate([[0, 0], cl_namaster_I[0]])
    ell = np.concatenate([[0, 1], binning.get_effective_ells()])
    cl_norm = ell * (ell + 1) / np.pi / 2
    cl_norm[0] = 1
    return ell, cl_norm, cl