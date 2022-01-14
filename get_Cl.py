import healpy as hp
# import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
from astropy.io import fits
import time
import copy

nside = 2048; lmax = 2048

comp = "IQU"

spectra_components = ["TT", "EE", "BB"]

datadir="/global/cscratch1/sd/jianyao/Dust/"


IQU_ws = hp.read_map('/global/cscratch1/sd/jianyao/Dust/dust_gnilc_hybrid_out_nside2048_float32.fits', field = None)
IQU_ls_only = hp.read_map('/global/cscratch1/sd/jianyao/Dust/dust_IQU_with_large_scales_only.fits', field = None)

planck_mask_filename = datadir + "HFI_Mask_GalPlane-apo2_2048_R2.00.fits"
planck_mask_80 = hp.read_map(planck_mask_filename, ["GAL080"])
planck_mask_90 = hp.read_map(planck_mask_filename, ["GAL090"])
planck_mask_97 = hp.read_map(planck_mask_filename, ["GAL097"])
planck_mask_99 = hp.read_map(planck_mask_filename, ["GAL099"])
planck_mask_100 = np.ones_like(planck_mask_99)
planck_mask = [planck_mask_80, planck_mask_90, planck_mask_97, planck_mask_99, planck_mask_100]
# planck_mask = np.int_(np.ma.masked_not_equal(planck_mask, 0.0).mask)
# fsky = planck_mask.sum() / planck_mask.size

output_nside = 2048 
output_lmax = 2 * output_nside
output_ell = np.arange(output_lmax + 1)
output_cl_norm = output_ell * (output_ell + 1) / np.pi / 2
output_cl_norm[0] = 1

# log_ls = hp.alm2map(ii_LS_alm, nside=output_nside)
# ls_only = log_pol_tens_to_map(log_ls)
# print("ls only shape", ls_only.shape)

def log_pol_tens_to_map(log_pol_tens):
    P = np.sqrt(log_pol_tens[1] ** 2 + log_pol_tens[2] ** 2)
    m = np.empty_like(log_pol_tens)
    exp_i = np.exp(log_pol_tens[0])
    m[0] = exp_i * np.cosh(P)
    m[1:] = log_pol_tens[1:] / P * exp_i * np.sinh(P)
    return m

def run_namaster(m, mask, lmax, nlbins=1):
    """Compute C_ell with NaMaster
    Parameters
    ----------
    m : numpy array
        T only or TQU HEALPix map
    mask : numpy array
        mask, 1D, 0 for masked pixels,
        needs to have same Nside of the input map
    lmax : int
        maximum ell of the spherical harmonics transform
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

    f_0 = nmt.NmtField(mask, [m[0]])

    if len(m) == 3:
        f_2 = nmt.NmtField(mask, m[1:].copy())  # NaMaster masks the map in-place
        cl_namaster = nmt.compute_full_master(f_2, f_2, binning)
        cl["EE"] = np.concatenate([[0, 0], cl_namaster[0]])
        cl["BB"] = np.concatenate([[0, 0], cl_namaster[3]])
        cl_namaster = nmt.compute_full_master(f_0, f_2, binning)
        cl["TE"] = np.concatenate([[0, 0], cl_namaster[0]])
    elif m.ndim == 1:
        m = m.reshape((1, -1))

    cl_namaster_I = nmt.compute_full_master(f_0, f_0, binning)

    cl["TT"] = np.concatenate([[0, 0], cl_namaster_I[0]])
    ell = np.concatenate([[0, 1], binning.get_effective_ells()])
    cl_norm = ell * (ell + 1) / np.pi / 2
    cl_norm[0] = 1
    return ell, cl_norm, cl
  
cl_all = [];  start = time.time()
# power spectrum of IQU transformed from iqu with small scales
for i in range(5):
    print(i)
    output_ell, output_cl_norm, cl_out_i = run_namaster(IQU_ls_only, mask=planck_mask[i], lmax=output_lmax, nlbins = 10)
    cl_all.append(cl_out_i);
    print((time.time() - start)/60.0)
    
np.save('cl_all_ls_only.npy', cl_all)