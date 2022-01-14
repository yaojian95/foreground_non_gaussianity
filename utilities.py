import numpy as np
import healpy as hp

def log_pol_tens_to_map(log_pol_tens):
    P = np.sqrt(log_pol_tens[1] ** 2 + log_pol_tens[2] ** 2)
    m = np.empty_like(log_pol_tens)
    exp_i = np.exp(log_pol_tens[0])
    m[0] = exp_i * np.cosh(P)
    m[1:] = log_pol_tens[1:] / P * exp_i * np.sinh(P)
    return m
  
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