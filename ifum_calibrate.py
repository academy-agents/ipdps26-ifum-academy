# https://articles.adsabs.harvard.edu/pdf/1996PASP..108..277O
# https://articles.adsabs.harvard.edu/pdf/1997PASP..109..614O
# https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

import numpy as np
import scipy
import os

import scipy.optimize
import ifum_utils
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve
import matplotlib.pyplot as plt
import seaborn as sns


class Calibrate():
    '''
    Class to ...

    Attributes:
        

    Methods:
        
    '''
    def __init__(self, datafilename: str, arcfilename: str, flatfilename: str,
                 bad_masks, total_masks: int, mask_groups: int):
        self.datafilename = datafilename
        self.arcfilename = arcfilename
        self.flatfilename = flatfilename
        self.bad_masks = bad_masks
        self.total_masks = total_masks
        self.mask_groups = mask_groups

        self.isol_sky_lines = [7794.112,7808.467,7821.503,#7949.204,7913.708*0.9+7912.252*0.1,
                  7964.650,7993.332,8014.059,8025.668*0.5+8025.952*0.5,#8052.020,
                  8061.979*0.5+8062.377*0.5,#8382.392*0.8+8379.903*0.1+8380.737*0.1,8310.719,8344.602*0.9+8343.033*0.1,
                  8399.170,8415.231,8430.174,8465.208*0.6+8465.509*0.4,8493.389,
                  8504.628*0.6+8505.054*0.4,8791.186*0.9+8790.993*0.1,8885.850,#8778.333*0.7+8776.294*0.3,8827.096*0.9+8825.448*0.1,
                  8903.114,8919.533*0.5+8919.736*0.5,8943.395,8957.922*0.5+8958.246*0.5,8988.366,
                  9001.115*0.5+9001.577*0.5,9037.952*0.5+9038.162*0.5,9049.232*0.6+9049.845*0.4,9374.318*0.1+9375.961*0.9,9419.729,
                  9439.650,9458.528,9476.748*0.4+9476.912*0.6,9502.815,9519.182*0.4+9519.527*0.6,
                  9552.534,9567.090*0.6+9567.588*0.4,9607.726,9620.630*0.7+9621.300*0.3]
        self.isol_sky_lines = ifum_utils.air_to_vacuum(np.array(self.isol_sky_lines))

    def get_color_info(self,color):
        datadir = os.path.join(os.path.relpath("out"),self.datafilename+color+".fits")
        arcdir = os.path.join(os.path.relpath("out"),self.arcfilename+color+".fits")
        flatdir = os.path.join(os.path.relpath("out"),self.flatfilename+"_withbias_"+color+".fits")
        cmraymask = os.path.join(os.path.relpath("out"),self.datafilename+color+"_cmray_mask.fits")
        trace_data = os.path.join(os.path.relpath("out"),self.datafilename+color+"_trace_fits.npz")
        trace_arc = os.path.join(os.path.relpath("out"),self.arcfilename+color+"_trace_fits.npz")
        trace_flat = os.path.join(os.path.relpath("out"),self.flatfilename+color+"_trace_fits.npz")
        bad_mask = self.bad_masks[0] if color=="b" else self.bad_masks[1]
        return datadir,arcdir,flatdir,cmraymask,trace_data,trace_arc,trace_flat,bad_mask



    def get_spectra(self,sig_mult,bins,color) -> None:
        datadir,_,_,cmraymask,trace_data,_,_,bad_mask = self.get_color_info(color)
        data = fits.open(datadir)[0].data
        cmray = fits.open(cmraymask)[0].data
        npzdata = np.load(trace_data)

        wl_mask = np.zeros(data.shape)

        spectra = np.empty((self.total_masks//2,bins.shape[0]))
        for mask in range(self.total_masks//2):
            if mask not in bad_mask:
                spectra[mask],px,wl = ifum_utils.get_spectrum_fluxbins(mask, 
                                                bins, 
                                                npzdata["traces"], 
                                                npzdata["rotation_traces"], 
                                                npzdata["init_traces_sigma"],
                                                sig_mult,
                                                npzdata["rect_x"],
                                                npzdata["wl_calib"],
                                                data,
                                                cmray)
                
                wl_mask[px[:,0],px[:,1]] = wl

            else:
                spectra[mask] = np.nan

        # plt.figure(figsize=(250,5))
        # for mask in range(self.total_masks//2):
        #     plt.plot(bins,spectra[mask])
        # plt.show()

        # plt.figure(dpi=300)
        # plt.imshow(wl_mask,origin="lower",cmap="magma")
        # plt.axis("off")
        # plt.show()

        # fits.writeto("wl_mask.fits",wl_mask,overwrite=True)
        fits.writeto("spectrabins.fits",spectra,overwrite=True)
        
        save_dict = dict(npzdata)
        save_dict["wl_bins"] = bins
        save_dict["flux_bins"] = spectra
        np.savez(trace_data, **save_dict)

    def gauss_background(self,x,*var):
        H,H1,a,x0,sigma = var
        # _ = args
        return H+H1*x+a*np.exp(-(x-x0)**2/(2*sigma**2))

    def get_sky_weights(self,wl,intensity,bad_mask) -> tuple:
        sky_ints = np.empty((self.total_masks//2,len(self.isol_sky_lines)))
        sky_errs = np.empty((self.total_masks//2,len(self.isol_sky_lines)))

        for m in np.arange(self.total_masks//2):
            if m not in bad_mask:
                offset = 8

                for i,line in enumerate(self.isol_sky_lines):
                    mask_area = ((wl)>(self.isol_sky_lines[i]-offset))&((wl)<(self.isol_sky_lines[i]+offset))
                    mask_area = mask_area&(~np.isnan(intensity[m]))
                    
                    try:
                        p0 = [0,
                        0,
                        np.nanmax(intensity[m][mask_area]),
                        np.argmax(intensity[m][mask_area])+np.nanmin(wl[mask_area]),
                        5/3]
                        popt,pcov = scipy.optimize.curve_fit(self.gauss_background,
                                                                wl[mask_area],
                                                                intensity[m][mask_area],p0=p0)
                        
                        perr = np.sqrt(np.diag(pcov))
                        gauss_x = np.linspace(wl[mask_area][0],wl[mask_area][-1],100)
                        
                        popt[0] = 0
                        popt[1] = 0
                        sky_ints[m,i] = np.trapezoid(self.gauss_background(gauss_x,*popt),gauss_x)
                        # sky_int.append(popt[2])
                        sky_errs[m,i] = perr[3]
                    except:
                        sky_ints[m,i] = 0
                        sky_errs[m,i] = np.inf
                
                # sky_ints.append(np.array(sky_int))
                # sky_errs.append(np.array(sky_err))
            else:
                sky_ints[m,i] = 0
                sky_errs[m,i] = np.inf

        return np.array(sky_ints),np.array(sky_errs)
    
    def intensity_corr(self) -> None:
        sky_ints,sky_errs = np.empty((0,len(self.isol_sky_lines))),np.empty((0,len(self.isol_sky_lines)))
        for color_idx,color in enumerate(["b","r"]):
            _,_,_,_,trace_data,_,_,bad_mask = self.get_color_info(color)
            npzdata = np.load(trace_data)
            shape = npzdata["wl_bins"].shape[0]
            sky_int,sky_err = self.get_sky_weights(npzdata["wl_bins"],npzdata["flux_bins"],bad_mask)
            sky_ints = np.vstack((sky_ints,sky_int))
            sky_errs = np.vstack((sky_errs,sky_err))
        avg_sky = np.average(sky_int,axis=0,weights=1./sky_err)
        # avg_sig = np.mean(sky_err[np.isfinite(sky_err)])
        
        deg = 1
        full_intensity = np.empty((0,shape))
        for color_idx,color in enumerate(["b","r"]):
            _,_,_,_,trace_data,_,_,bad_mask = self.get_color_info(color)
            npzdata = np.load(trace_data)
            intensity = npzdata["flux_bins"]
            wl = npzdata["wl_bins"]
            for m in np.arange(self.total_masks//2):
                if m not in bad_mask:
                    # try:
                    if color == "r":
                        c_m = int(m+self.total_masks//2)
                    else:
                        c_m = m
                    sky_int = np.array(sky_ints[c_m])
                    sky_err = np.array(sky_errs[c_m])
                    # mask0 = (sky_int!=0)&(np.isfinite(sky_err))
                    ratio = avg_sky/sky_int
                    # ratio shouldn't be extremely high or low
                    mask0 = (ratio<(np.nanmedian(ratio)+3*np.nanstd(ratio[np.isfinite(ratio)])))&(ratio>(np.nanmedian(ratio)-3*np.nanstd(ratio[np.isfinite(ratio)])))&(ratio!=0)&(np.isfinite(ratio))
                    ratio = ratio[mask0]
                    w_ = np.nan_to_num(1/np.array(sky_err)[mask0],nan=0)
                    # int_fit_ = np.polyfit(self.isol_sky_lines[mask0],ratio,deg,w=w_)

                    # plt.scatter(self.isol_sky_lines[mask0],ratio,c=ifum_utils.normalize(w_))
                    # plt.show()

                    # diff = ratio-np.poly1d(int_fit_)(self.isol_sky_lines[mask0])
                    # mask = (ratio<(np.poly1d(int_fit_)(self.isol_sky_lines[mask0])+1.5*np.std(diff)))&(ratio>(np.poly1d(int_fit_)(self.isol_sky_lines[mask0])-1.5*np.std(diff)))
                    # w = np.nan_to_num(1/np.array(sky_errs[c_m])[mask0][mask],nan=0)
                    # int_fit = np.polyfit(self.isol_sky_lines[mask0][mask],ratio[mask],deg,w=w)

                    fit_mask,int_fit = ifum_utils.sigma_clip(self.isol_sky_lines[mask0],ratio,deg,w_,sigma=3.0)
                    # plt.scatter(self.isol_sky_lines[mask0],ratio,c=ifum_utils.normalize(w_))
                    # plt.scatter(self.isol_sky_lines[mask0][fit_mask],ratio[fit_mask],c="red",marker="x",alpha=0.5)
                    # plt.show()

                    fit_mask_,int_fit = ifum_utils.sigma_clip(self.isol_sky_lines[mask0][fit_mask],ratio[fit_mask],deg+1,w_[fit_mask],sigma=1.0)
                    # plt.scatter(self.isol_sky_lines[mask0][fit_mask],ratio[fit_mask],c=ifum_utils.normalize(w_[fit_mask]))
                    # plt.scatter(self.isol_sky_lines[mask0][fit_mask][fit_mask_],ratio[fit_mask][fit_mask_],c="red",marker="x",alpha=0.5)
                    # plt.plot(self.isol_sky_lines[mask0][fit_mask],np.poly1d(int_fit)(self.isol_sky_lines[mask0][fit_mask]),c="red")
                    # plt.show()
                
                    full_intensity = np.vstack((full_intensity,np.poly1d(int_fit)(wl)*intensity[m]))
                    # full_intensity[c_m] = np.poly1d(int_fit)(wl)*intensity[m]
                    # except:
                    #     print("error in intensity fit")
                    #     full_intensity[c_m] = np.nan
                else:
                    full_intensity = np.vstack((full_intensity,np.repeat(np.nan,shape)))
                    # full_intensity[c_m] = np.nan

        # plt.figure(figsize=(300,10))
        # for m in np.arange(self.total_masks):
        #     if m<276:
        #         plt.plot(wl,full_intensity[m],color="blue",alpha=0.05)
        #     else:
        #         plt.plot(wl,full_intensity[m],color="red",alpha=0.05)
        # plt.show()

        # fits.writeto("spectrabins_int.fits",full_intensity,overwrite=True)

        npzdir = os.path.join(os.path.relpath("out"),self.datafilename+"_spectra.npz")
        save_dict = {'wl': wl,
                     'intensity': full_intensity}

        np.savez(npzdir, **save_dict)

    def _viz(self) -> None:
        npzdir = os.path.join(os.path.relpath("out"),self.datafilename+"_spectra.npz")
        npzdata = np.load(npzdir)

        plt.figure(figsize=(300,10))
        sky_mean = np.nanmedian(npzdata["intensity"],axis=0)
        for m in np.arange(self.total_masks):
            plt.plot(npzdata["wl"],npzdata["intensity"][m]-sky_mean,color="gray",alpha=0.5)
        plt.show()