import numpy as np
import scipy
import math
import os
from .utils import *
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve
import matplotlib.pyplot as plt
from parsl.app.app import python_app

# https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/407/1157#/browse

class Rectify():
    '''
    Class to ...

    Attributes:
        

    Methods:
        
    '''
    def __init__(
        self,
        datadir,
        arcdir,
        flatdir_biased,
        cmraymask,
        trace_data,
        trace_arc,
        trace_flat,
        wavelength,
        bad_mask,
        total_masks: int, 
        mask_groups: int
    ):
        self.datadir = datadir
        self.arcdir = arcdir
        self.flatdir = flatdir_biased
        self.cmraymask = cmraymask
        self.trace_data = trace_data
        self.trace_arc = trace_arc
        self.trace_flat = trace_flat
        self.bad_mask = bad_mask
        self.total_masks = total_masks
        self.mask_groups = mask_groups

        if wavelength == "far red":
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

        elif wavelength == "blue":

            self.sky_lines = [4026.1914,
                            4861.3203,
                            # 5197.9282,
                            5577.3467,
                            5867.5522,
                            5889.9590,
                            5895.9321,
                            5915.3076,
                            5932.8643,
                            5953.3589*0.5+5953.4897*0.5]
            self.sky_lines_guess_ = [7,
                                    844,
                                    # 1180,
                                    1558,
                                    1849,
                                    1873,
                                    1879,
                                    1898,
                                    1916,
                                    1936]
            
            self.emission_lines = {"Ar": [#4044.418,
                                        #   4158.590, #?
                                        #   4164.180,
                                        #   4181.884,
                                        #   4190.713,
                                        #   4191.029,
                                        #   4198.317,
                                        #   4200.674,
                                        #   4251.185,
                                        #   4259.362,
                                        #   4266.286,
                                        #   4272.169,
                                        #   4300.101,
                                        #   4333.561,
                                        #   4335.338,
                                        #   4345.168,
                                        #   4510.733,
                                        #   4522.323,
                                        #   4596.097,
                                        #   4628.441,
                                        #   4702.316,
                                        #   5151.391,
                                        #   5162.285,
                                        5187.746,
                                        #   5221.271,
                                        #   5421.352,
                                        # 5451.652,
                                        5495.874,
                                        5506.113,
                                        5558.702,
                                        5572.541,
                                        #   5606.733,
                                        5650.704,
                                        #   5739.520,
                                        #   5834.263,
                                        #   5860.310,
                                        #   5882.624,
                                        #   5888.584,
                                        #   5912.085,
                                        #   5928.813,
                                        #   5942.669, #?
                                        #   5987.302
                                        ],
                                "Ne": [#4042.642,
                                        #   4064.036,
                                        #   4080.148,
                                        #   4131.0613,
                                        #   4164.8079,
                                        #   4174.3667,
                                        #   4175.2197,
                                        #   4198.1018,
                                        #   4268.0086,
                                        #   4269.7223,
                                        #   4270.2252,
                                        #   4274.6617,
                                        #   4275.5590,
                                        #   4306.2508,
                                        #   4334.1267,
                                        #   4336.2268,
                                        #   4363.524,
                                        #   4395.556,
                                        #   4416.817,
                                        #   4421.5553,
                                        #   4422.5205,
                                        #   4424.8065,
                                        #   4425.400,
                                        #   4433.7239,
                                        #   4460.175,
                                        #   4465.6544,
                                        #   4466.8120,
                                        #   4475.656,
                                        #   4483.190,
                                        #   4488.0926,
                                        #   4537.7545,
                                        #   4704.3949,
                                        #   4708.8594,
                                        #   4710.0650,
                                        #   4712.0633,
                                        #   4715.344,
                                        #   4788.9258,
                                        #   4827.338,
                                        #   4884.9170,
                                        #   4957.0335,
                                        #   5341.0938,
                                        5400.5616,
                                        5852.4878,
                                        #   5881.8950,
                                        #   6029.9968 #?
                                        ],
                                "He": [4026.1914,
                                        4120.8154,
                                        4143.761,
                                        #   4168.967,
                                        4387.9296,
                                        # 4471.4802,
                                        4713.1457,
                                        4921.9313,
                                        5015.6783,
                                        5047.738,
                                        5875.621
                                        ],
                                "Be": [#4407.935,
                                        #   4572.66605
                                        ],
                                "Li": [#4132.56,
                                        #   4273.07,
                                        #   4602.83,
                                        #   4971.66,
                                        #   4971.75
                                        ]}

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

    def optimize_centers(
        self,
        arc_maskdir,
        output,
        arc_or_data="arc",
        sig_mult=3,
        fix_sparse=False,
    ) -> None:
        mask_data = fits.open(arc_maskdir)[0].data
        if arc_or_data == "arc":
            npzdir = self.trace_arc
            npzfile = np.load(npzdir)
            data = fits.open(self.arcdir)[0].data
            cmrays = False
        else:
            npzdir = self.trace_data
            npzfile = np.load(npzdir)
            data = fits.open(self.datadir)[0].data
            cmrays = True

        centers = npzfile["centers"]
        print(centers)
        traces_sigma = np.load(self.trace_flat)["traces_sigma"]
        print(traces_sigma)
        masks_l = np.arange(self.total_masks//2)+1

        if np.intersect1d(self.bad_mask,np.arange(self.total_masks//2)).size > 0:
            _, _, ind2 = np.intersect1d(self.bad_mask,np.arange(self.total_masks//2),return_indices=True)
            for mask_ in ind2:
                centers = np.insert(centers, mask_, np.nan, axis=1)

        intensities = np.empty((self.total_masks//2,data.shape[1]))
        if cmrays:
            cmray_data = fits.open(self.cmraymask)[0].data
            for i,m in enumerate(masks_l):
                a = get_spectrum_simple_withnan(data,mask_data,m,cmray_data)
                intensities[i] = a
        else:
            for i,m in enumerate(masks_l):
                a = get_spectrum_simple(data,mask_data,m)
                intensities[i] = a
        x = np.arange(data.shape[1])

        # plt.plot(x,intensities[0])
        # plt.show()
        # plt.plot(x,intensities[10])
        # plt.show()
        # plt.plot(x,intensities[100])
        # plt.show()

        # sometimes centers are sparse. this fixes this
        if fix_sparse:
            # get first lag, so that quadrant areas are optimized
            lags_0 = np.empty(intensities.shape[0])
            for i,intensity in enumerate(intensities):
                lags_0[i] = get_lag(intensity,intensities[0])
            quadrants = 4
            expectation = (centers.shape[0]//quadrants)*2 # right now, doubles!
            # centers = np.empty((0,centers.shape[1]))
            # print(centers.shape)
            # print(intensities.shape)
            quadrants_x = np.array_split(x,quadrants)
            for quadrant in range(quadrants):
                quad_mask = quadrants_x[quadrant]
                centers_nearby = centers[:,0][(centers[:,0]>quad_mask[0])&(centers[:,0]<quad_mask[-1])]
                # print(centers_nearby)
                if len(centers_nearby)<expectation:
                    quadrant_ref_a = intensities[0]

                    quad_lags = np.empty(intensities.shape[0])
                    norm_intensities = np.empty((intensities.shape[0],quad_mask.size))
                    for i,intensity in enumerate(intensities):
                        quad_mask_ = np.intersect1d(np.union1d(quad_mask, quad_mask+int(lags_0[i])),
                                                    np.arange(intensities.shape[1]))
                        quad_lag = get_lag(intensity[quad_mask_],quadrant_ref_a[quad_mask_])
                        quad_lags[i] = quad_lag
                        try:
                            norm_intensities[i] = intensity[quad_mask+quad_lag]
                        except:
                            norm_intensities[i] = np.interp(quad_mask, quad_mask-quad_lag, intensity[quad_mask])
                        # plt.plot(quad_mask,norm_intensities[i],alpha=0.01)
                        
                    ref_intensity = np.nanmedian(norm_intensities,axis=0)
                    # plt.plot(quad_mask,ref_intensity,color="red")
                    peaks,_ = scipy.signal.find_peaks(ref_intensity,
                                                      distance=np.nanmean(traces_sigma)*sig_mult)

                    peaks += quad_mask[0]
                    nearby_idxs = []
                    for ctr in centers_nearby:
                        if np.abs(peaks[(np.abs(peaks - ctr)).argmin()]-ctr)<np.nanmean(traces_sigma)*sig_mult:
                            nearby_idxs.append((np.abs(peaks - ctr)).argmin())
                    peaks = np.delete(peaks,nearby_idxs)

                    # for peak in peaks:
                    #     plt.axvline(peak,color="red")
                    
                    peaks -= quad_mask[0]
                    if len(peaks) > expectation:
                        top_args = ref_intensity[peaks].argsort()[-expectation:][::-1]
                        peaks = np.sort(peaks[top_args])
                    peaks += quad_mask[0]
                                        
                    # for peak in peaks:
                    #     plt.axvline(peak,color="green")                
                    # plt.show()

                    new_centers = []
                    for peak in peaks:
                        new_centers.append(peak+quad_lags)
                    new_centers = np.array(new_centers)
                    centers = np.vstack((centers,new_centers))

            centers[centers<0] = np.nan

        # print(centers)

        # get centers for each peak area for each mask
        # if ~np.array_equiv(npzfile["init_traces"],npzfile["traces"]):
        centers_opt = np.zeros_like(centers)
        for peak in range(len(centers)):
            for i,mask in enumerate(masks_l):
                print(f"Mask {i}, {type(mask)}, {type(self.bad_mask)}")
                if (mask-1) not in self.bad_mask:
                    print(f"Mask {i}, {mask}: {traces_sigma[int(mask-1)]}")
                    offset = math.ceil(sig_mult*np.median(np.poly1d(traces_sigma[int(mask-1)])(np.arange(data.shape[0]))))
                    try:
                        if (centers[peak][i]+offset)<(~np.isnan(intensities[i])).cumsum(0).argmax(0):
                            mask_area = ((x)>(centers[peak][i]-(offset+1)))&((x)<(centers[peak][i]+(offset+1)))    
                            mask_area = mask_area&(~np.isnan(intensities[i]))

                            # plt.title(centers[peak][i])
                            # offset_=30
                            # mask_area_ = ((x)>(centers[peak][i]-(offset_+1)))&((x)<(centers[peak][i]+(offset_+1)))    
                            # plt.plot(x[mask_area_],intensities[i][mask_area_])
                            # plt.axvline(centers[peak][i])
                            # plt.show()

                            p0 = [0,np.max(intensities[i][mask_area]),np.argmax(intensities[i][mask_area])+np.min(x[mask_area]),offset/3]
                            popt,pcov = scipy.optimize.curve_fit(gauss,x[mask_area],intensities[i][mask_area],p0=p0)                            
                        else:
                            mask_area = x>((~np.isnan(intensities[i])).cumsum(0).argmax(0)-offset*2)
                            mask_area = mask_area&(~np.isnan(intensities[i]))
                            p0 = [0,np.max(intensities[i][mask_area]),np.argmax(intensities[i][mask_area])+np.min(x[mask_area]),offset/3]
                            popt,pcov = scipy.optimize.curve_fit(gauss,x[mask_area],intensities[i][mask_area],p0=p0)
                        x_err = np.sqrt(np.diag(pcov))[3]
                        
                        if x_err < np.median(np.poly1d(traces_sigma[int(mask-1)])(np.arange(data.shape[0])))/3:
                            centers_opt[peak][i] = popt[2]
                        else:
                            # keep old peak (maybe nan?)
                            centers_opt[peak][i] = np.nan#centers[peak][i]
                    except:
                        # print(f"bad 1D peak fit: color {self.color}, mask {mask}, peak {peak}")
                        centers_opt[peak][i] = np.nan
                else:
                    centers_opt[peak][i] = np.nan
        # else:
            # centers_opt = centers

        save_dict = {}
        save_dict["centers_opt"] = centers_opt
        save_dict["rect_int"] = intensities
        np.savez(output, **save_dict)


    def rectify(self, centers_file, output, arc_or_data="arc") -> None:
        if arc_or_data == "arc":
            data = fits.open(self.arcdir)[0].data
        else:
            data = fits.open(self.datadir)[0].data
        
        npzfile = np.load(centers_file)
        centers = npzfile["centers_opt"]

        # sort low to high
        sorted_indices = np.argsort(np.nanmedian(centers,axis=1))
        centers = centers[sorted_indices]

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
            # bad_fits = bad_fits&(abs((centers[i]-(np.poly1d(np.polyfit(masks_l[bad_fits],centers[i][bad_fits],3))(masks_l))))<3)

            poly_sig_mask, fit = sigma_clip(masks_l[bad_fits],
                                                       centers[i][bad_fits],
                                                       3,weight=np.ones_like(masks_l[bad_fits]),
                                                       sigma=3,iter=10,include=0.9)
            bad_fits[bad_fits] = poly_sig_mask
            # do it again, in case not all points were catched for a good line (high outliers can still come about with high sigma)
            poly_sig_mask, fit = sigma_clip(masks_l[bad_fits],
                                                       centers[i][bad_fits],
                                                       3,weight=np.ones_like(masks_l[bad_fits]),
                                                       sigma=1.5,iter=10,include=0.9)
            bad_fits[bad_fits] = poly_sig_mask
            
            error_from_basic_fit = np.nanmean((centers[i][bad_fits] - np.poly1d(np.polyfit(masks_l[bad_fits],centers[i][bad_fits],3))(masks_l[bad_fits]))**2)
            bad_fits_s = np.array(np.split(bad_fits,self.mask_groups))
            # print(i,centers[i][0],error_from_basic_fit)
            
            # plt.figure(figsize=(8,3))
            # plt.title(error_from_basic_fit)
            # plt.scatter(masks_l,centers[i],color="red")
            # plt.scatter(masks_l[bad_fits],centers[i][bad_fits],color="blue")
            # plt.plot(masks_l,np.poly1d(np.polyfit(masks_l[bad_fits],centers[i][bad_fits],2))(masks_l))
            # plt.show()

            # mse is usually <1, if it's this large, something is wrong in the centers
            # if more than len(bad_masks) are found to be bad fits, skip this emission line too
            if np.all(np.sum(bad_fits_s, axis=1) > bad_fits_s.shape[0]/2) and error_from_basic_fit < 3: # np.all(np.sum(~bad_fits_s, axis=1) <= 3*len(self.bad_mask))
                # print(np.sum(bad_fits_s==False),error_from_basic_fit,np.sum(~bad_fits_s, axis=1))
                
                poly = []
                poly_fits = []
                for j in range(self.mask_groups):

                    # segment_fit = np.polyfit(masks_split[j][bad_fits_s[j]],centers_split[j,i][bad_fits_s[j]],1)

                    group_mask, segment_fit = sigma_clip(masks_split[j][bad_fits_s[j]],
                                                                    centers_split[j,i][bad_fits_s[j]],
                                                                    1,weight=np.ones_like(masks_split[j][bad_fits_s[j]]),
                                                                    sigma=1,iter=10)

                    res = scipy.optimize.minimize(miniminize_double_linear_func,
                                x0 = np.array([segment_fit[0],segment_fit[1],segment_fit[1]]), 
                                args = np.array([masks_split[j][bad_fits_s[j]][group_mask],centers_split[j,i][bad_fits_s[j]][group_mask]]))
                    y_fit = double_linear_func(res.x,masks_split[j])

                    # plt.scatter(masks_split[j][bad_fits_s[j]],centers_split[j,i][bad_fits_s[j]],color="red")
                    # plt.scatter(masks_split[j][bad_fits_s[j]][group_mask],centers_split[j,i][bad_fits_s[j]][group_mask],color="blue")
                    # plt.plot(masks_split[j],y_fit,color="black")
                    
                    poly.append(y_fit)
                    poly_fits.append(centers_split[j,i]-y_fit)
                # plt.show()    
                poly = np.array(poly)
                poly_fits = np.array(poly_fits)
                full_shifts[i] = poly.flatten()

                print("GOOD emission line:",np.nanmean(centers[i][bad_fits]))
            else:
                print("BAD emission line:",np.nanmean(centers[i][bad_fits]))
                full_shifts[i] = np.nan
        full_shifts = full_shifts[~np.isnan(full_shifts)].reshape(-1,self.total_masks//2)

        x = np.arange(0,data.shape[1],1)
        x_s = np.empty((self.total_masks//2,data.shape[1]))
        for mask in masks_l:
            # plt.scatter(full_shifts[:,mask-1],full_shifts[:,0]-full_shifts[:,mask-1])

            _, fit = sigma_clip(full_shifts[:,mask-1],full_shifts[:,0]-full_shifts[:,mask-1],
                                           3,weight=np.ones_like(full_shifts[:,mask-1]),sigma=3,iter=10)

            # fit = np.polyfit(full_shifts[:,mask-1],full_shifts[:,0]-full_shifts[:,mask-1],2)

            # plt.plot(full_shifts[:,mask-1],np.poly1d(fit)(full_shifts[:,mask-1]))

            x_s[mask-1] = x+np.poly1d(fit)(x)
        # plt.show()


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
        save_dict["rect_int"] = npzfile["rect_int"]
        np.savez(output, **save_dict)

        # print(intensities.shape)
        # plt.imshow(intensities,origin="lower")
        # plt.axis("off")
        # plt.show()

        # plt.imshow(x_s,cmap="twilight_shifted",origin="lower")
        # plt.axis("off")
        # plt.show()

    def calib(
        self, 
        rect_data, 
        rect_arc,
        output,
        sig_mult=3, 
        use_sky=True, 
        deg=4,
    ) -> None:
        npz_data = np.load(rect_data)
        npz_arc = np.load(rect_arc)
        traces_sigma = np.load(self.trace_flat)["traces_sigma"]

        # IMPORTANT: x's are considered to be the same for now
        data_xs = npz_data["rect_x"]
        data_intensities = npz_data["rect_int"]
        arc_xs = npz_data["rect_x"]
        arc_intensities = npz_arc["rect_int"]

        delta_func = np.zeros(data_xs.shape[1])
        delta_func[np.arange(len(delta_func))[self.sky_lines_guess_]] = 1
        gauss_kernal = Gaussian1DKernel(stddev=1)
        delta_func = convolve(delta_func,gauss_kernal)
        lag = get_lag(data_intensities[0],delta_func)
        sky_lines_guess = self.sky_lines_guess_+lag

        masks_l = np.arange(self.total_masks//2)+1
        if use_sky:
            sky_lines_xs = []
            sky_lines_errs = []
            for m in masks_l:
                if (m-1) not in self.bad_mask:
                    sky_lines_x = []
                    sky_lines_err = []
                    offset = math.ceil(sig_mult*np.median(np.poly1d(traces_sigma[int(m-1)])(np.arange(data_xs.shape[1]))))
            
                    for i,line in enumerate(self.sky_lines):
                        mask_area = ((data_xs[m-1])>(sky_lines_guess[i]-offset))&((data_xs[m-1])<(sky_lines_guess[i]+offset))
                        mask_area = mask_area&(~np.isnan(data_intensities[m-1]))
                        try:
                            p0 = [0,np.nanmax(data_intensities[m-1][mask_area]),np.argmax(data_intensities[m-1][mask_area])+np.nanmin(data_xs[m-1][mask_area]),5/3]
                            popt,pcov = scipy.optimize.curve_fit(gauss,data_xs[m-1][mask_area],data_intensities[m-1][mask_area],p0=p0)
                            perr = np.sqrt(np.diag(pcov))
                            gauss_x = np.linspace(data_xs[m-1][mask_area][0],data_xs[m-1][mask_area][-1],100)
                            sky_lines_x.append(popt[2])
                            sky_lines_err.append(perr[2])
                        except:
                            print("BAD PEAK FIT:",m,line)
                            sky_lines_x.append(0)
                            sky_lines_err.append(np.inf)

                    sky_lines_xs.append(sky_lines_x)
                    sky_lines_errs.append(sky_lines_err)                

            # print(sky_lines_errs)
            centers = np.average(sky_lines_xs,axis=0,weights=1./np.array(sky_lines_errs))
            # print(centers)
            # stds = np.nanstd(sky_lines_xs,axis=0)#*np.nanmedian(sky_lines_errs,axis=0)
            # print(stds)
            stds = np.nanmedian(sky_lines_errs,axis=0)
            # print(stds)
            stds[~np.isfinite(stds)] = 2*np.nanmax(stds[np.isfinite(stds)])
            stds = 1-normalize(stds)

            # fit, put arc lines on top, then fit with all
            best_fit = np.polyfit(self.sky_lines,centers,3,w=stds)
            arc_lines_x0 = np.poly1d(best_fit)(self.clear_emission_lines)
        else:
            best_fit = np.polyfit(self.sky_lines,sky_lines_guess,3)
            arc_lines_x0 = np.poly1d(best_fit)(self.clear_emission_lines)
        # TEMP: get a slightly better best_fit
        # best_arc_lines = [4921.9313,5015.6783,5875.621]
        # arc_lines_x0 = np.poly1d(best_fit)(best_arc_lines)

        # arc_lines_xs = []
        # arc_lines_errs = []
        # for m in masks_l:
        #     if (m-1) not in self.bad_mask:
        #         arc_lines_x = []
        #         arc_lines_err = []
        #         offset = math.ceil(sig_mult*np.median(np.poly1d(traces_sigma[int(m-1)])(np.arange(data_xs.shape[1]))))
        
        #         for i,line in enumerate(best_arc_lines):
        #             mask_area = ((arc_xs[m-1])>(arc_lines_x0[i]-offset))&((arc_xs[m-1])<(arc_lines_x0[i]+offset))
        #             mask_area = mask_area&(~np.isnan(arc_intensities[m-1]))
        #             try:
        #                 p0 = [0,np.nanmax(arc_intensities[m-1][mask_area]),np.nanargmax(arc_intensities[m-1][mask_area])+np.nanmin(arc_xs[m-1][mask_area]),5/3]
        #                 popt,pcov = scipy.optimize.curve_fit(ifum_utils.gauss,arc_xs[m-1][mask_area],arc_intensities[m-1][mask_area],p0=p0)
        #                 perr = np.sqrt(np.diag(pcov))
        #                 gauss_x = np.linspace(arc_xs[m-1][mask_area][0],arc_xs[m-1][mask_area][-1],100)
        #                 arc_lines_x.append(popt[2])
        #                 arc_lines_err.append(perr[2])
        #             except:
        #                 print("BAD arc fit:",m,line)
        #                 arc_lines_x.append(0)
        #                 arc_lines_err.append(np.inf)
        #         arc_lines_xs.append(arc_lines_x)
        #         arc_lines_errs.append(arc_lines_err)

        # centers_ = np.average(arc_lines_xs,axis=0,weights=1./np.array(arc_lines_errs))
        # print(centers_,best_arc_lines)
        # best_fit = np.polyfit(np.concatenate((best_arc_lines, self.sky_lines)),np.concatenate((centers_, centers)),3)
        # prelim_best = np.polyfit(centers_,best_arc_lines,3)

        # arc_lines_x0 = np.poly1d(best_fit)(self.clear_emission_lines)

        # self._viz_wl(prelim_calib=prelim_best)


        arc_lines_xs = []
        arc_lines_errs = []
        for m in masks_l:
            if (m-1) not in self.bad_mask:
                arc_lines_x = []
                arc_lines_err = []
                offset = math.ceil(sig_mult*np.median(np.poly1d(traces_sigma[int(m-1)])(np.arange(data_xs.shape[1]))))
        
                for i,line in enumerate(self.clear_emission_lines):
                    mask_area = ((arc_xs[m-1])>(arc_lines_x0[i]-offset))&((arc_xs[m-1])<(arc_lines_x0[i]+offset))
                    mask_area = mask_area&(~np.isnan(arc_intensities[m-1]))
                    try:
                        p0 = [0,np.nanmax(arc_intensities[m-1][mask_area]),np.nanargmax(arc_intensities[m-1][mask_area])+np.nanmin(arc_xs[m-1][mask_area]),5/3]
                        popt,pcov = scipy.optimize.curve_fit(gauss,arc_xs[m-1][mask_area],arc_intensities[m-1][mask_area],p0=p0)
                        perr = np.sqrt(np.diag(pcov))
                        gauss_x = np.linspace(arc_xs[m-1][mask_area][0],arc_xs[m-1][mask_area][-1],100)
                        arc_lines_x.append(popt[2])
                        arc_lines_err.append(perr[2])
                    except:
                        print("BAD arc fit:",m,line)
                        arc_lines_x.append(0)
                        arc_lines_err.append(np.inf)
                arc_lines_xs.append(arc_lines_x)
                arc_lines_errs.append(arc_lines_err)
        arc_lines_errs = np.array(arc_lines_errs)
        arc_lines_errs = np.nan_to_num(arc_lines_errs,np.max(arc_lines_errs[np.isfinite(arc_lines_errs)]))
        arc_centers = np.average(arc_lines_xs,axis=0,weights=1./np.array(arc_lines_errs))
        arc_stds = np.nanmedian(arc_lines_errs,axis=0)#np.nanstd(arc_lines_xs,axis=0)*np.nanmean(arc_lines_errs,axis=0)
        arc_stds[~np.isfinite(arc_stds)] = 2*np.nanmax(arc_stds[np.isfinite(arc_stds)])
        arc_stds = 1-normalize(arc_stds)
    


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
        if use_sky:
            best_fit_for_shift = np.polyfit(arc_centers[~np.isnan(arc_centers)],
                                            np.array(self.clear_emission_lines)[~np.isnan(arc_centers)],
                                            4,
                                            )#w=np.nan_to_num(arc_stds[~np.isnan(arc_centers)]))
            # plt.scatter(arc_centers[~np.isnan(arc_centers)],np.array(self.clear_emission_lines)[~np.isnan(arc_centers)])
            # plt.scatter(arc_centers[~np.isnan(arc_centers)],np.poly1d(best_fit_for_shift)(arc_centers[~np.isnan(arc_centers)]))
            # plt.show()
            args = best_fit_for_shift[:-1]
            shift0 = best_fit_for_shift[-1]

            def deg4poly(x,shift):
                d,c,b,a = args
                return shift+a*x+b*x**2+c*x**3+d*x**4
        
            self.sky_lines = self.sky_lines[1:]
            centers = centers[1:]
            stds = stds[1:]

            fit_shift,_ = scipy.optimize.curve_fit(deg4poly,centers,self.sky_lines,p0=shift0)
            shift = fit_shift-shift0
            centers = centers+shift
            arc_stds = arc_stds*5

            full_colors = np.concatenate([self.emission_colors,np.repeat("brown",len(self.sky_lines))])
            full_wls = np.concatenate([self.clear_emission_lines,self.sky_lines])
            full_centers = np.concatenate([arc_centers,centers])
            full_stds = np.concatenate([arc_stds,stds])
        else:
            shift = 0

            # ONLY arc
            full_colors = np.array(self.emission_colors)
            full_wls = np.array(self.clear_emission_lines)
            full_centers = np.array(arc_centers)
            full_stds = np.array(arc_stds)

        # ONLY sky
        # full_colors = np.repeat("orange",len(sky_lines))
        # full_wls = np.array(sky_lines)
        # full_centers = np.array(centers)
        # full_stds = np.array(stds)

        mask = ~np.isnan(full_centers)
        full_colors = full_colors[mask]
        full_centers = full_centers[mask]
        full_wls = full_wls[mask]
        full_stds = full_stds[mask]
        full_stds = np.nan_to_num(full_stds)

        # full_best_fit = np.polyfit(full_centers,full_wls,deg)#,w=full_stds)
        full_best_fit, _ = ransac(full_centers,full_wls,deg,max_iter=1000,threshold=0.1)

        # plt.figure().set_facecolor("lightgray")
        # # plt.title(f"{data_filename+color} RESIDUALS")
        # plt.ylabel("predicted wavelength (Å)")
        # plt.xlabel("x")
        # plt.scatter(full_centers,full_wls,color="black",alpha=1)
        # plt.scatter(full_centers,np.poly1d(full_best_fit)(full_centers),c=full_colors,alpha=0.5)
        # plt.show()

        # plt.figure(figsize=(12,4)).set_facecolor("lightgray")
        # # plt.title(f"{data_filename+color} RESIDUALS")
        # plt.ylabel("actual - predicted wavelength (Å)")
        # plt.xlabel("x")
        # plt.scatter(full_centers,full_wls-np.poly1d(full_best_fit)(full_centers),c=full_colors,s=30*full_stds+10,alpha=0.5)
        # plt.axhline(0,c="gray")
        # plt.show()
    
        # wl_x = self.air_to_vacuum(np.poly1d(full_best_fit)(arc_xs))
        full_best_fit[-1] += shift
        wl_x = self.air_to_vacuum(np.poly1d(full_best_fit)(data_xs))

        save_dict = {}
        save_dict["rect_x"] = data_xs
        save_dict["rect_wl"] = wl_x
        save_dict["wl_calib"] = full_best_fit
        np.savez(output, **save_dict)

    def _viz(self) -> None:
        npz_data = np.load(self.trace_data)
        npz_arc = np.load(self.trace_arc)

        data_wls = npz_data["rect_wl"]
        data_intensities = npz_data["rect_int"]
        arc_wls = npz_data["rect_wl"]
        arc_intensities = npz_arc["rect_int"]

        # print(arc_wls.shape,arc_intensities.shape)

        # plt.figure(figsize=(50,5))
        plt.ylabel("intensity")
        plt.xlabel("wavelength")
        c = plt.get_cmap("viridis")(np.arange(276)/276)
        for m in range(276):
            plt.plot(data_wls[m],normalize(data_intensities[m]),color=c[m],alpha=0.05)
        c = plt.get_cmap("magma")(np.arange(276)/276)
        for m in range(276):
            plt.plot(arc_wls[m],normalize(arc_intensities[m]),color=c[m],alpha=0.05)
        # plt.show()

    def _viz_wl(self,prelim_calib=None) -> None:
        arc_data = fits.open(self.arcdir)[0].data
        arc_maskdir = os.path.join(os.path.relpath("out"),self.arcfilename+self.color+"_mask.fits")
        arc_mask_data = fits.open(arc_maskdir)[0].data

        data = fits.open(self.datadir)[0].data
        maskdir = os.path.join(os.path.relpath("out"),self.arcfilename+self.color+"_mask.fits")
        mask_data = fits.open(maskdir)[0].data

        m = 1
        intensity_arc = get_spectrum_simple_withnan(arc_data,arc_mask_data,m)
        intensity = get_spectrum_simple_withnan(data,mask_data,m,fits.open(self.cmraymask)[0].data)
        x = np.arange(data.shape[1])

        if prelim_calib is None:
            print("activated")
            prelim_calib = np.polyfit(self.sky_lines_guess_,self.sky_lines,3)
        
        plt.scatter(self.sky_lines_guess_,self.sky_lines)
        plt.plot(self.sky_lines_guess_,np.poly1d(prelim_calib)(self.sky_lines_guess_),alpha=0.5)
        plt.show()

        plt.figure(figsize=(100,10))
        plt.vlines(self.clear_emission_lines,0,1,colors=self.emission_colors,alpha=0.7)
        for i,line in enumerate(self.clear_emission_lines):
            plt.text(line,0.05, 
                    str(f"arc {line:.3f}Å"), color="black",
                    ha='center',va='top',fontsize=8,rotation='vertical')
        plt.plot(np.poly1d(prelim_calib)(x),normalize(intensity_arc),color="blue",alpha=0.5)
        plt.plot(np.poly1d(prelim_calib)(x),normalize(intensity),color="darkorange",alpha=0.5)
        plt.show()

    def viz_rect(self):
        import matplotlib
        cmap = matplotlib.colormaps.get_cmap('gray').copy()
        cmap.set_bad(color='maroon')
        
        npz_data = np.load(self.trace_data)
        npz_arc = np.load(self.trace_arc)

        data_xs = npz_arc["rect_wl"]
        data_intensities = npz_data["rect_int"]
        arc_xs = npz_arc["rect_wl"]
        arc_intensities = npz_arc["rect_int"]

        x = np.arange(np.min(data_xs),np.max(data_xs),1)

        plt.figure(dpi=100)#.set_facecolor("lightgray")
        plt.subplot(5,1,3)
        plt.title("transformation gradient",weight="bold",color="#542E26")
        plt.imshow(arc_xs,cmap="twilight_shifted",origin="lower")
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.subplot(5,1,1)
        plt.title("1D arc lamp spectra",color="#854001")
        plt.imshow(arc_intensities,cmap=cmap,origin="lower",vmin=0,vmax=150)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.subplot(5,1,4)
        plt.title("rectified arc lamp spectra",color="#854001")
        norm_intensities = np.empty(arc_intensities.shape)
        for i,x_ss in enumerate(arc_xs):
            norm_intensities[i] =  np.interp(x, x_ss, arc_intensities[i])
        plt.imshow(norm_intensities,cmap=cmap,origin="lower",vmin=0,vmax=150)    
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

        # visualize the results
        plt.subplot(5,1,2)
        plt.title("1D on-sky science spectra",color="#854001")
        plt.imshow(data_intensities,cmap=cmap,origin="lower",vmin=0,vmax=50)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.subplot(5,1,5)
        plt.title("rectified on-sky science spectra",color="#854001")
        snorm_intensities = np.empty(data_intensities.shape)
        for i,x_ss in enumerate(data_xs):
            snorm_intensities[i] =  np.interp(x, x_ss, data_intensities[i])
        plt.imshow(snorm_intensities,cmap=cmap,origin="lower",vmin=0,vmax=50)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.tight_layout()
        plt.savefig("testim.png",dpi=1500,bbox_inches='tight',pad_inches=0.1)
        plt.close()



@python_app
def optimize_center_app(
    rectify_args,
    arc_maskdir, 
    arc_or_data,
    fix_sparse,
    outputs=()
):
    rectify = Rectify(**rectify_args)
    return rectify.optimize_centers(
        arc_maskdir=arc_maskdir,
        output=outputs[0],
        arc_or_data=arc_or_data,
        fix_sparse=fix_sparse
    )

@python_app
def rectify_app(rectify_args, centers_file, arc_or_data, outputs=()):
    rectify = Rectify(**rectify_args)
    return rectify.rectify(centers_file, outputs[0], arc_or_data)

@python_app
def calib_app(rectify_args, rect_data, rect_arc, use_sky, outputs=()):
    rectify = Rectify(**rectify_args)
    return rectify.calib(rect_data, rect_arc, outputs[0], use_sky=use_sky)