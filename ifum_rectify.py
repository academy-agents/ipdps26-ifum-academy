import numpy as np
import scipy
import math
import os
import ifum_utils
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve
import matplotlib.pyplot as plt
import seaborn as sns

class Rectify():
    '''
    Class to ...

    Attributes:
        

    Methods:
        
    '''
    def __init__(self, color: str, datafilename: str, arcfilename: str, flatfilename: str,
                 bad_masks, total_masks: int, mask_groups: int):
        self.color = color
        self.datafilename = datafilename
        self.arcfilename = arcfilename
        self.flatfilename = flatfilename
        self.datadir = os.path.join(os.path.relpath("out"),self.datafilename+self.color+".fits")
        self.arcdir = os.path.join(os.path.relpath("out"),self.arcfilename+self.color+".fits")
        self.flatdir = os.path.join(os.path.relpath("out"),self.flatfilename+"_withbias_"+self.color+".fits")
        self.cmraymask = os.path.join(os.path.relpath("out"),self.datafilename+self.color+"_cmray_mask.fits")
        self.trace_data = os.path.join(os.path.relpath("out"),self.datafilename+self.color+"_trace_fits.npz")
        self.trace_arc = os.path.join(os.path.relpath("out"),self.arcfilename+self.color+"_trace_fits.npz")
        self.trace_flat = os.path.join(os.path.relpath("out"),self.flatfilename+self.color+"_trace_fits.npz")
        self.bad_mask = bad_masks[0] if color=="b" else bad_masks[1]
        self.total_masks = total_masks
        self.mask_groups = mask_groups
        self.sky_lines = [7794.112,7808.467,7821.503,7913.708*0.9+7912.252*0.1,#7949.204,
                        7964.650,7993.332,8014.059,8025.668*0.5+8025.952*0.5,8052.020,
                        8061.979*0.5+8062.377*0.5,8310.719,8344.602*0.9+8343.033*0.1,#8382.392*0.8+8379.903*0.1+8380.737*0.1,
                        8399.170,8415.231,8430.174,8465.208*0.6+8465.509*0.4,8493.389,
                        8504.628*0.6+8505.054*0.4,8778.333*0.7+8776.294*0.3,8791.186*0.9+8790.993*0.1,8827.096*0.9+8825.448*0.1,8885.850,
                        8903.114,8919.533*0.5+8919.736*0.5,8943.395,8957.922*0.5+8958.246*0.5,8988.366,
                        9001.115*0.5+9001.577*0.5,9037.952*0.5+9038.162*0.5,9049.232*0.6+9049.845*0.4,9374.318*0.1+9375.961*0.9,9419.729,
                        9439.650,9458.528,9476.748*0.4+9476.912*0.6,9502.815,9519.182*0.4+9519.527*0.6,
                        9552.534,9567.090*0.6+9567.588*0.4,9607.726,9620.630*0.7+9621.300*0.3]
        self.sky_lines_guess_ = [46,61,75,169,#205,
                                221,251,272,284,311,
                                321,575,609,#647,
                                665,681,697,732,761,
                                772,1050,1064,1100,1160,
                                1178,1195,1219,1235,1265,
                                1279,1316,1328,1662,1707,
                                1727,1747,1766,1793,1810,
                                1844,1859,1901,1914]
        

        self.emission_lines = {"Ar": [7948.176,8006.157,8014.786,8103.693,8115.311,
                                    8264.522,8408.21,8424.648,8521.442,
                                    8667.944,9122.967,#9194.638,9291.531,8346.420,8592.624,8605.776
                                    9224.499,9354.22,9657.786],#9657.786,9331.05, 
                            "Ne": [8206.043,8495.3591,8634.6472],#8377.6070,8862.497,,8377.6070,
                            "He": [],
                            "Be": [],
                            "Li": []}
        self.emission_colors = (["red"]*len(self.emission_lines["Ar"])
                            +["#A020F0"]*len(self.emission_lines["Ne"])
                            +["orange"]*len(self.emission_lines["He"])
                            +["blue"]*len(self.emission_lines["Be"])
                            +["green"]*len(self.emission_lines["Li"]))
        self.clear_emission_lines = [item for items in self.emission_lines.values() for item in items]

    def air_to_vacuum(self,wavelengths):
        s = 1e4/(wavelengths)
        n = 1 + 0.00008336624212083+0.02408926869968/(130.1065924522-s**2)+0.0001599740894897/(38.92568793293-s**2)
        return wavelengths*n

    def optimize_centers(self,arc_or_data="arc",sig_mult=1.5) -> None: # re-fit gaussian centers with better trace masks
        if arc_or_data == "arc":
            npzdir = self.trace_arc
            npzfile = np.load(npzdir)
            data = fits.open(self.arcdir)[0].data
            maskdir = os.path.join(os.path.relpath("out"),self.arcfilename+self.color+"_mask.fits")
            mask_data = fits.open(maskdir)[0].data
            cmrays = False
        else:
            npzdir = self.trace_data
            npzfile = np.load(npzdir)
            data = fits.open(self.datadir)[0].data
            maskdir = os.path.join(os.path.relpath("out"),self.arcfilename+self.color+"_mask.fits")
            mask_data = fits.open(maskdir)[0].data
            cmrays = True
        centers = npzfile["centers"]
        traces_sigma = np.load(self.trace_flat)["traces_sigma"]
        masks_l = np.arange(self.total_masks//2)+1

        if np.intersect1d(self.bad_mask,np.arange(self.total_masks//2)).size > 0:
            _, _, ind2 = np.intersect1d(self.bad_mask,np.arange(self.total_masks//2),return_indices=True)
            for mask_ in ind2:
                centers = np.insert(centers, mask_, np.nan, axis=1)

        intensities = np.empty((self.total_masks//2,data.shape[1]))
        if cmrays:
            cmray_data = fits.open(self.cmraymask)[0].data
            for i,m in enumerate(masks_l):
                a = ifum_utils.get_spectrum_simple(data,mask_data,m,cmray_data)
                intensities[i] = a
        else:
            for i,m in enumerate(masks_l):
                a = ifum_utils.get_spectrum_simple(data,mask_data,m)
                intensities[i] = a
        x = np.arange(data.shape[1])

        # get centers for each peak area for each mask
        centers_opt = np.zeros_like(centers)
        for peak in range(len(centers)):
            for i,mask in enumerate(masks_l):
                if (mask-1) not in self.bad_mask:
                    offset = math.ceil(sig_mult*np.median(np.poly1d(traces_sigma[int(mask-1)])(np.arange(data.shape[0]))))
                    try:
                        if (centers[peak][i]+offset)<(~np.isnan(intensities[i])).cumsum(0).argmax(0):
                            mask_area = ((x)>(centers[peak][i]-(offset+1)))&((x)<(centers[peak][i]+(offset+1)))    
                            mask_area = mask_area&(~np.isnan(intensities[i]))
                            p0 = [0,np.max(intensities[i][mask_area]),np.argmax(intensities[i][mask_area])+np.min(x[mask_area]),offset/3]
                            popt,_ = scipy.optimize.curve_fit(ifum_utils.gauss,x[mask_area],intensities[i][mask_area],p0=p0)                            
                        else:
                            mask_area = x>((~np.isnan(intensities[i])).cumsum(0).argmax(0)-offset*2)
                            mask_area = mask_area&(~np.isnan(intensities[i]))
                            p0 = [0,np.max(intensities[i][mask_area]),np.argmax(intensities[i][mask_area])+np.min(x[mask_area]),offset/3]
                            popt,_ = scipy.optimize.curve_fit(ifum_utils.gauss,x[mask_area],intensities[i][mask_area],p0=p0)
                        centers_opt[peak][i] = popt[2]
                    except:
                        # print(f"bad 1D peak fit: color {self.color}, mask {mask}, peak {peak}")
                        centers_opt[peak][i] = np.nan
                else:
                    centers_opt[peak][i] = np.nan
        
        save_dict = dict(npzfile)
        save_dict["centers_opt"] = centers
        save_dict["rect_int"] = intensities
        np.savez(npzdir, **save_dict)


    def rectify(self,arc_or_data="arc") -> None:
        if arc_or_data == "arc":
            npzdir = self.trace_arc
            npzfile = np.load(npzdir)
            data = fits.open(self.arcdir)[0].data
        else:
            npzdir = self.trace_data
            npzfile = np.load(npzdir)
            data = fits.open(self.datadir)[0].data
        centers = npzfile["centers_opt"]

        # if np.intersect1d(self.bad_mask,np.arange(self.total_masks//2)).size > 0:
        #     _, _, ind2 = np.intersect1d(self.bad_mask,np.arange(self.total_masks//2),return_indices=True)
        #     for mask_ in ind2:
        #         centers = np.insert(centers, mask_, np.nan, axis=1)
        
        full_shifts = np.empty((centers.shape))
        masks_split = np.array(np.split(np.arange(self.total_masks//2),self.mask_groups))
        centers_split = np.array(np.split(centers,self.mask_groups,axis=1))
        masks_l = np.arange(self.total_masks//2)+1

        for i in range(centers.shape[0]):
            bad_fits = ~np.isnan(centers[i])
            # bad_fits = (~np.isin(masks_l, (self.bad_mask+1)))&(~np.isnan(centers[i]))
            # bad_fits = bad_fits&(abs((centers[i]-(np.poly1d(np.polyfit(masks_l[bad_fits],centers[i][bad_fits],2))(masks_l))))<3)
            error_from_basic_fit = np.nanmean((centers[i] - np.poly1d(np.polyfit(masks_l[bad_fits],centers[i][bad_fits],3))(masks_l))**2)
            bad_fits_s = np.array(np.split(bad_fits,self.mask_groups))

            # plt.figure(figsize=(8,3))
            # plt.title(error_from_basic_fit)
            # plt.scatter(masks_l,centers[i],color="red")
            # plt.scatter(masks_l[bad_fits],centers[i][bad_fits],color="blue")
            # plt.plot(masks_l,np.poly1d(np.polyfit(masks_l[bad_fits],centers[i][bad_fits],2))(masks_l))
            # plt.show()

            # mse is usually <1, if it's this large, something is wrong in the centers
            if error_from_basic_fit < 1.5:
                print(np.sum(bad_fits_s==False),error_from_basic_fit)
                
                poly = []
                poly_fits = []
                for j in range(self.mask_groups):
                    segment_fit = np.polyfit(masks_split[j][bad_fits_s[j]],centers_split[j,i][bad_fits_s[j]],1)
                    res = scipy.optimize.minimize(ifum_utils.miniminize_double_linear_func,
                                x0 = np.array([segment_fit[0],segment_fit[1],segment_fit[1]]), 
                                args = np.array([masks_split[j][bad_fits_s[j]],centers_split[j,i][bad_fits_s[j]]]))
                    y_fit = ifum_utils.double_linear_func(res.x,masks_split[j])
                    
                    poly.append(y_fit)
                    poly_fits.append(centers_split[j,i]-y_fit)
                    
                poly = np.array(poly)
                poly_fits = np.array(poly_fits)
                full_shifts[i] = poly.flatten()
            else:
                print("BAD center")
                full_shifts[i] = np.nan
        full_shifts = full_shifts[~np.isnan(full_shifts)].reshape(-1,self.total_masks//2)
        
        x = np.arange(0,data.shape[1],1)
        x_s = np.empty((self.total_masks//2,data.shape[1]))
        for mask in masks_l:
            fit = np.polyfit(full_shifts[:,mask-1],full_shifts[:,0]-full_shifts[:,mask-1],2)
            x_s[mask-1] = x+np.poly1d(fit)(x)
        


        # use mask file to get slightly better intensities
        # if arc_or_data == "arc":
        #     maskdir = os.path.join(os.path.relpath("out"),self.arcfilename+self.color+"_mask.fits")
        #     cmrays = False
        # else:
        #     maskdir = os.path.join(os.path.relpath("out"),self.datafilename+self.color+"_mask.fits")
        #     cmrays = True
        # mask_data = fits.open(maskdir)[0].data

        # intensities = np.empty((self.total_masks//2,data.shape[1]))
        # if cmrays:
        #     cmray_data = fits.open(self.cmraymask)[0].data
        #     for i,m in enumerate(masks_l):
        #         a = ifum_utils.get_spectrum_simple(data,mask_data,m,cmray_data)
        #         intensities[i] = a
        # else:
        #     for i,m in enumerate(masks_l):
        #         a = ifum_utils.get_spectrum_simple(data,mask_data,m)
        #         intensities[i] = a

        save_dict = dict(npzfile)
        save_dict["rect_x"] = x_s
        # save_dict["rect_int"] = intensities
        np.savez(npzdir, **save_dict)

        # print(intensities.shape)
        # plt.imshow(intensities,origin="lower")
        # plt.axis("off")
        # plt.show()

        # plt.imshow(x_s,cmap="twilight_shifted",origin="lower")
        # plt.axis("off")
        # plt.show()

    def calib(self) -> None:
        npz_data = np.load(self.trace_data)
        npz_arc = np.load(self.trace_arc)

        # IMPORTANT: x's are considered to be the same for now
        data_xs = npz_data["rect_x"]
        data_intensities = npz_data["rect_int"]
        arc_xs = npz_data["rect_x"]
        arc_intensities = npz_arc["rect_int"]

        delta_func = np.zeros(data_xs.shape[1])
        delta_func[np.arange(len(delta_func))[self.sky_lines_guess_]] = 1
        gauss_kernal = Gaussian1DKernel(stddev=1)
        delta_func = convolve(delta_func,gauss_kernal)
        lag = ifum_utils.get_lag(data_intensities[0],delta_func)
        sky_lines_guess = self.sky_lines_guess_+lag

        sky_lines_xs = []
        sky_lines_errs = []
        masks_l = np.arange(self.total_masks//2)+1
        for m in masks_l:
            if (m-1) not in self.bad_mask:
                sky_lines_x = []
                sky_lines_err = []
                offset = 7
        
                for i,line in enumerate(self.sky_lines):
                    mask_area = ((data_xs[m-1])>(sky_lines_guess[i]-offset))&((data_xs[m-1])<(sky_lines_guess[i]+offset))
                    mask_area = mask_area&(~np.isnan(data_intensities[m-1]))
                    p0 = [0,np.nanmax(data_intensities[m-1][mask_area]),np.argmax(data_intensities[m-1][mask_area])+np.nanmin(data_xs[m-1][mask_area]),5/3]
                    try:
                        popt,pcov = scipy.optimize.curve_fit(ifum_utils.gauss,data_xs[m-1][mask_area],data_intensities[m-1][mask_area],p0=p0)
                        perr = np.sqrt(np.diag(pcov))
                        gauss_x = np.linspace(data_xs[m-1][mask_area][0],data_xs[m-1][mask_area][-1],100)
                        sky_lines_x.append(popt[2])
                        sky_lines_err.append(perr[2])
                    except:
                        print("BAD FIT")
                        sky_lines_x.append(0)
                        sky_lines_err.append(np.inf)

                sky_lines_xs.append(sky_lines_x)
                sky_lines_errs.append(sky_lines_err)

        centers = np.average(sky_lines_xs,axis=0,weights=1./np.array(sky_lines_errs))
        stds = np.nanstd(sky_lines_xs,axis=0)*np.nanmean(sky_lines_errs,axis=0)
        stds[~np.isfinite(stds)] = 2*np.nanmax(stds[np.isfinite(stds)])
        stds = 1-ifum_utils.normalize(stds)

        # fit, put arc lines on top, then fit with all
        best_fit = np.polyfit(self.sky_lines,centers,3,w=stds)
        arc_lines_x0 = np.poly1d(best_fit)(self.clear_emission_lines)

        arc_lines_xs = []
        arc_lines_errs = []
        for m in masks_l:
            if (m-1) not in self.bad_mask:
                arc_lines_x = []
                arc_lines_err = []
                offset = 7
        
                for i,line in enumerate(self.clear_emission_lines):
                    mask_area = ((arc_xs[m-1])>(arc_lines_x0[i]-offset))&((arc_xs[m-1])<(arc_lines_x0[i]+offset))
                    mask_area = mask_area&(~np.isnan(arc_intensities[m-1]))
                    p0 = [0,np.nanmax(arc_intensities[m-1][mask_area]),np.nanargmax(arc_intensities[m-1][mask_area])+np.nanmin(arc_xs[m-1][mask_area]),5/3]
                    try:
                        popt,pcov = scipy.optimize.curve_fit(ifum_utils.gauss,arc_xs[m-1][mask_area],arc_intensities[m-1][mask_area],p0=p0)
                        perr = np.sqrt(np.diag(pcov))
                        gauss_x = np.linspace(arc_xs[m-1][mask_area][0],arc_xs[m-1][mask_area][-1],100)
                        arc_lines_x.append(popt[2])
                        arc_lines_err.append(perr[2])
                    except:
                        print("BAD")
                        arc_lines_x.append(0)
                        arc_lines_err.append(np.inf)
                arc_lines_xs.append(arc_lines_x)
                arc_lines_errs.append(arc_lines_err)

        arc_centers = np.average(arc_lines_xs,axis=0,weights=1./np.array(arc_lines_errs))
        arc_stds = np.nanstd(arc_lines_xs,axis=0)*np.nanmean(arc_lines_errs,axis=0)
        arc_stds[~np.isfinite(arc_stds)] = 2*np.nanmax(arc_stds[np.isfinite(arc_stds)])
        arc_stds = 1-ifum_utils.normalize(arc_stds)
    


        # plt.figure(figsize=(15,3))
        # # plt.title(f"{data_filename+color}")
        # arc_colors = sns.color_palette("Blues",as_cmap=True)((arc_stds+.1)*0.8)
        # sky_colors = sns.color_palette("Oranges",as_cmap=True)((stds+.1)*0.8)
        # plt.vlines(arc_centers,0,1.9,color=arc_colors,alpha=0.7)
        # plt.vlines(centers,0,1.9,color=sky_colors,alpha=0.7)
        # for i,line in enumerate(self.clear_emission_lines):
        #     plt.text(arc_centers[i],1.9, 
        #             str(f"arc {line:.3f}Å"), color="white",
        #             ha='center',va='top',fontsize=8,rotation='vertical',backgroundcolor=arc_colors[i])
        # for i,line in enumerate(self.sky_lines):
        #     plt.text(centers[i],1.9, 
        #             str(f"sky {line:.3f}Å"), color="white",
        #             ha='center',va='top',fontsize=8,rotation='vertical',backgroundcolor=sky_colors[i])
        # plt.plot(arc_xs[0],ifum_utils.normalize(arc_intensities[0]),color="blue")
        # plt.plot(data_xs[0],ifum_utils.normalize(data_intensities[0]),color="orange")
        # plt.xlim(600,800)
        # plt.ylim(-0.05,1.95)
        # plt.axis("off")
        # plt.tight_layout()
        # plt.savefig("arc_sky_lines.png",dpi=300,bbox_inches="tight")
        # plt.show()



        # BAD BAD BAD CODING
        best_fit_for_shift = np.polyfit(arc_centers[~np.isnan(arc_centers)],
                                        np.array(self.clear_emission_lines)[~np.isnan(arc_centers)],
                                        4,
                                        w=np.nan_to_num(arc_stds[~np.isnan(arc_centers)]))
        args = best_fit_for_shift[:-1]
        shift0 = best_fit_for_shift[-1]

        def deg4poly(x,shift):
            d,c,b,a = args
            return shift+a*x+b*x**2+c*x**3+d*x**4
    
        fit_shift,_ = scipy.optimize.curve_fit(deg4poly,centers,self.sky_lines,p0=shift0)
        shift = fit_shift-shift0
        centers = centers+shift
        arc_stds = arc_stds*5
        # print(shift)

        # ONLY arc
        # full_colors = np.array(emission_colors)
        # full_wls = np.array(clear_emission_lines)
        # full_centers = np.array(arc_centers)
        # full_stds = np.array(arc_stds)

        # ONLY sky
        # full_colors = np.repeat("orange",len(sky_lines))
        # full_wls = np.array(sky_lines)
        # full_centers = np.array(centers)
        # full_stds = np.array(stds)

        # both
        full_colors = np.concatenate([self.emission_colors,np.repeat("orange",len(self.sky_lines))])
        full_wls = np.concatenate([self.clear_emission_lines,self.sky_lines])
        full_centers = np.concatenate([arc_centers,centers])
        full_stds = np.concatenate([arc_stds,stds])

        mask = ~np.isnan(full_centers)
        full_colors = full_colors[mask]
        full_centers = full_centers[mask]
        full_wls = full_wls[mask]
        full_stds = full_stds[mask]
        full_stds = np.nan_to_num(full_stds)

        full_best_fit = np.polyfit(full_centers,full_wls,4,w=full_stds)

        # plt.figure(figsize=(12,4)).set_facecolor("lightgray")
        # # plt.title(f"{data_filename+color} RESIDUALS")
        # plt.ylabel("actual - predicted wavelength (Å)")
        # plt.xlabel("x")
        # plt.scatter(full_centers,full_wls-np.poly1d(full_best_fit)(full_centers),c=full_colors,s=30*full_stds,alpha=0.5)
        # plt.axhline(0,c="gray")
        # plt.show()
    
        # wl_x = self.air_to_vacuum(np.poly1d(full_best_fit)(arc_xs))
        full_best_fit[-1] += shift
        wl_x = self.air_to_vacuum(np.poly1d(full_best_fit)(data_xs))

        save_dict = dict(npz_data)
        save_dict["rect_x"] = data_xs
        save_dict["rect_wl"] = wl_x
        save_dict["wl_calib"] = full_best_fit
        np.savez(self.trace_data, **save_dict)

    def _viz(self) -> None:
        npz_data = np.load(self.trace_data)
        npz_arc = np.load(self.trace_arc)

        data_wls = npz_data["rect_wl"]
        data_intensities = npz_data["rect_int"]
        arc_wls = npz_data["rect_wl"]
        arc_intensities = npz_arc["rect_int"]

        plt.figure(figsize=(50,5))
        plt.ylabel("intensity")
        plt.xlabel("wavelength")
        c = plt.get_cmap("viridis")(np.arange(276)/276)
        for m in range(276):
            plt.plot(data_wls[m],ifum_utils.normalize(data_intensities[m]),color=c[m],alpha=0.05)
        c = plt.get_cmap("magma")(np.arange(276)/276)
        for m in range(276):
            plt.plot(arc_wls[m],ifum_utils.normalize(arc_intensities[m]),color=c[m],alpha=0.05)
        plt.show()