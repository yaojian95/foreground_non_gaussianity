import healpy as hp
import numpy as np

from mix_tools import get_functionals
from projection_tools import *      
from utilities import *
import time 

output_nside = 2048 
output_lmax = 2 * output_nside
datadir="/global/cscratch1/sd/jianyao/Dust/"
savedir="/global/cscratch1/sd/jianyao/Dust/MFs/"

ss_cl = np.load('/global/cscratch1/sd/jianyao/Dust/small_scales_cl.npy')
ss_cl_pt = hp.read_cl('/global/cscratch1/sd/jianyao/Dust/small_scales_logpoltens_cl_lmax4096.fits')
ss_cl = np.row_stack((ss_cl, np.zeros(output_lmax + 1)))
ss_cl_pt = np.row_stack((ss_cl_pt, np.zeros(output_lmax + 1)))


lmax = 2048;nside = 2048;
comp = "IQU"
spectra_components = ["TT", "EE", "BB"]
ell_fit_low = {"TT":100, "EE":30, "BB":30}
ell_fit_high = {"TT":400, "EE":110, "BB":110}

###------ large scale of IQU -------######
dust_IQU = hp.read_map('/global/cscratch1/sd/jianyao/Dust/Dust_IQU_uK_RJ.fits', field = None) ### original maps in IQU
ls = get_large_scales(dust_IQU, lmax, spectra_components, ell_fit_high, output_nside)

###------ large scale of iqu -------######
log_pol_tens_varres = hp.read_map('/global/cscratch1/sd/jianyao/Dust/dust_gnilc_logpoltens_varres_nomono.fits', field = None) ### original maps in iqu
log_ls = get_large_scales(log_pol_tens_varres, lmax, spectra_components, ell_fit_high, output_nside)

# my_modulate_amp = hp.read_map(datadir + f"My_modulate_amp_nside{nside}.fits")
# my_modulate_amp_pol = hp.read_map(datadir + f"My_modulate_amp_pol_nside{nside}.fits")

modulate_amp = hp.read_map(datadir + f"modulate_amp_nside{nside}.fits")
modulate_amp_pol = hp.read_map(datadir + f"modulate_amp_pol_nside{nside}.fits")

ismooth = hp.smoothing(dust_IQU[0], fwhm=np.radians(5), lmax=lmax)
new_mod = minmax(ismooth, 0, 6)

header = set_header(0, 0, size_patch=3.75/60, Npix=320)
# header1 = set_header(0, 15, size_patch=3.75/60, Npix=320)
# header2 = set_header(15,15, size_patch=3.75/60, Npix=320)
# header3 = set_header(0, 30, size_patch=3.75/60, Npix=320)
# header4 = set_header(0, 45, size_patch=3.75/60, Npix=320)

# headers = [header0, header1, header2, header3, header4]

F_ss_pt = []; U_ss_pt = []; Chi_ss_pt = [];
F_ss = []; U_ss = []; Chi_ss = [];
print("Begin the loop!!")
for i in range(50):
    start = time.time()
    log_ss_pt = hp.synfast(ss_cl_pt, lmax=output_lmax, new=True,nside=output_nside) # maps generated from small iqu cls
    log_ss_pt[0] *= modulate_amp
    log_ss_pt[1:] *= modulate_amp_pol
    assert np.isnan(log_ss_pt).sum() == 0
    log_map_out = log_ls + log_ss_pt
    
    map_out_from_iqu = log_pol_tens_to_map(log_map_out)
    ss_pt = get_small_scales(map_out_from_iqu, lmax, spectra_components, ell_fit_high, output_nside)

    ss = hp.synfast(ss_cl, lmax=output_lmax, new=True,nside=output_nside) #maps generated from IQU cls
    ss[0] *= (new_mod*1.096)
    ss[1:] *= (new_mod*1.7094)  
    assert np.isnan(ss).sum() == 0

    map_out = ss + ls
    ss = get_small_scales(map_out, lmax, spectra_components, ell_fit_high, output_nside)
    
    if i == 0:
        hp.write_map(savedir + f"dust_IQU_from_iqu_with_small_scales.fits",map_out_from_iqu,dtype=np.float32,overwrite=True)
        hp.write_map(savedir + f"dust_IQU_with_small_scales.fits", map_out,dtype=np.float32,overwrite=True)
        
    hp.write_map(savedir + f"dust_IQU_from_iqu_only_small_scales_%03d.fits"%i, ss_pt, dtype=np.float32,overwrite=True)
    hp.write_map(savedir + f"dust_IQU_only_small_scales_%03d.fits"%i, ss,dtype=np.float32,overwrite=True)

    patch_ss = h2f(ss[0], header)
    patch_ss_pt = h2f(ss_pt[0], header)

    img_ss_pt = rescale_min_max(patch_ss_pt)
    rhos_ss_pt, f_ss_pt, u_ss_pt, chi_ss_pt = get_functionals(img_ss_pt)
    F_ss_pt.append(f_ss_pt);
    U_ss_pt.append(u_ss_pt);
    Chi_ss_pt.append(chi_ss_pt);

    img_ss = rescale_min_max(patch_ss)
    rhos_ss, f_ss, u_ss, chi_ss = get_functionals(img_ss)
    F_ss.append(f_ss);
    U_ss.append(u_ss);
    Chi_ss.append(chi_ss);

    end = time.time()
    print("You are at %s , time cost is %s mins!"%(i, (end - start)/60))

np.save('/global/cscratch1/sd/jianyao/Dust/MFs/F_ss_pt.npy', F_ss_pt);
np.save('/global/cscratch1/sd/jianyao/Dust/MFs/U_ss_pt.npy', U_ss_pt);
np.save('/global/cscratch1/sd/jianyao/Dust/MFs/Chi_ss_pt.npy', Chi_ss_pt);
np.save('/global/cscratch1/sd/jianyao/Dust/MFs/F_ss.npy', F_ss);
np.save('/global/cscratch1/sd/jianyao/Dust/MFs/U_ss.npy', U_ss);
np.save('/global/cscratch1/sd/jianyao/Dust/MFs/Chi_ss.npy', Chi_ss);