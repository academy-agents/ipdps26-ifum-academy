import os
import numpy as np
from astropy.io import fits
import scipy
import math
import matplotlib.pyplot as plt
import scipy.optimize
from .utils import *
from parsl.app.app import python_app

from sklearn.cluster import DBSCAN



class Mask():
    def __init__(self, color: str, flatfilename: str, bad_masks, 
                 total_masks: int, mask_groups: int):
        self.color = color
        self.flatfilename = flatfilename
        self.flatdir = os.path.join(os.path.abspath("out"),self.flatfilename+"_withbias_"+self.color+".fits")
        self.maskdir = os.path.join(os.path.abspath("out"),self.flatfilename+self.color+"_mask.fits")
        self.datadir = os.path.join(os.path.abspath("out"),self.flatfilename+self.color+"_trace_fits.npz")
        self.bad_mask = bad_masks[0] if color=="b" else bad_masks[1]
        self.total_masks = total_masks
        self.mask_groups = mask_groups

    def first_guess(self, deg, median_filter = (1,9)) -> np.ndarray:
        with fits.open(self.flatdir) as flatf:
            flat_data = flatf[0].data
        flat_data = scipy.ndimage.median_filter(flat_data,size=median_filter)

        expected_peaks = int(self.total_masks//2-len(self.bad_mask))

        x = np.arange(0,flat_data.shape[1],3) # sample every 3rd pixel, will speed things up!
        all_peaks = np.array([])
        all_x = np.array([])
        mask_peaks = np.empty(shape=(len(x),expected_peaks))
        mask_polys = np.empty(shape=(expected_peaks,deg+1))

        num_det_peaks = []
        for i,idx in enumerate(x):
            # fit continuum
            y = np.arange(flat_data.shape[0])
            col = flat_data[:,idx]
            cont_fit = np.polyfit(y,col,deg)
            for j in range(10):
                polymask = col<(np.poly1d(cont_fit)(y)+np.std(np.poly1d(cont_fit)(y)))
                cont_fit = np.polyfit(y[polymask],col[polymask],deg)
            flat_flat = col/(np.poly1d(cont_fit)(y))

            # detect peaks (# expected - # bad masks)
            peaks,_ = scipy.signal.find_peaks(flat_flat,
                                            #   height=np.percentile(flat_flat,30), # maybe??? assumes uncalibrated peak heights...
                                              distance=len(y)/self.total_masks/2,
                                              width=[len(y)/self.total_masks/2,len(y)/self.total_masks*2])
            
            # append all peaks for later
            all_peaks = np.append(all_peaks,peaks)
            all_x = np.append(all_x,np.repeat(idx,len(peaks)))

            # take top (# of expected peaks) to include
            num_det_peaks.append(len(peaks))
            if len(peaks) >= expected_peaks:
                top_args = flat_flat[peaks].argsort()[-expected_peaks:][::-1]
                peaks = peaks[top_args]
                mask_peaks[i,:] = np.sort(peaks)
            else:
                mask_peaks[i,:] = np.nan

        all_peaks_flat = np.column_stack([all_x,all_peaks])

        threshold = 1.5
        pot_bad_masks = []
        for i,mask_dots in enumerate(mask_peaks.T):
            nanmask = ~np.isnan(mask_dots)
            best_fit, best_mask = ransac(x[nanmask],mask_dots[nanmask],deg,max_iter=1000,threshold=threshold)

            # optimize using all detected peaks
            distances = np.abs(all_peaks_flat[:,1] - np.poly1d(best_fit)(all_peaks_flat[:,0]))
            all_mask = distances < threshold*1.5 #expand a bit
            better_fit = np.polyfit(all_peaks_flat[:,0][all_mask],all_peaks_flat[:,1][all_mask],deg)
            better_fit_std = np.std(all_peaks_flat[:,1][all_mask]-np.poly1d(better_fit)(all_peaks_flat[:,0][all_mask]))

            mask_polys[i] = better_fit
            
            if better_fit_std>threshold:
                pot_bad_masks.append(i)

        for mask in self.bad_mask:
            mask_polys = np.insert(mask_polys, mask, np.nan, axis=0)

        return mask_polys

    def mask_poly(self, mask_polys, n) -> None:        
        with fits.open(self.flatdir) as flatf:
            flat_data = flatf[0].data
        flat_data = scipy.ndimage.median_filter(flat_data,size=(1,3))

        weight = np.log10(np.median(flat_data,axis=0))
        weight = np.array(weight - np.min(weight))[::3]
        lines = np.random.choice(np.arange(0,flat_data.shape[1],3),
                                 size=n, 
                                 replace=False, 
                                 p=weight/np.sum(weight))
    
        gauss_centers_full = np.empty((0,self.total_masks//2))
        gauss_sigmas_full = np.empty((0,self.total_masks//2))
        gauss_amps_full = np.empty((0,self.total_masks//2))
        
        masks_split = np.array(np.split(np.arange(self.total_masks//2),self.mask_groups))

        x_s = lines
        multi_fits = []
        for x in x_s:
            multi_fits.append(
                multi_gauss_fit(x,mask_polys,masks_split,flat_data,self.bad_mask)
            )

        print("submitted")
        
        bad_lines = []
        for i,future in enumerate(multi_fits):
            if future.result() is None:
                bad_lines.append(x_s[i])
            else:
                centers,sigmas,amps = future.result()
                gauss_centers_full = np.vstack((gauss_centers_full,np.array(centers).flatten()))
                gauss_sigmas_full = np.vstack((gauss_sigmas_full,np.array(sigmas).flatten()))
                gauss_amps_full = np.vstack((gauss_amps_full,np.array(amps).flatten()))

        save_dict = {'x': x_s[np.isin(x_s, bad_lines, invert=True)],
                     'centers': gauss_centers_full,
                     'sigmas': gauss_sigmas_full, 
                     'amps': gauss_amps_full}

        np.savez(self.datadir, **save_dict)

        return None

    def plot_trace_fits(self,center_deg,sigma_deg) -> None:
        npzfile = np.load(self.datadir)
        x = npzfile["x"]
        centers = npzfile["centers"]
        sigmas = npzfile["sigmas"]
        amps = npzfile["amps"]

        fig,ax = plt.subplots(2,1, figsize=(10,10))
        fig.suptitle(f"Residuals for {self.flatfilename}{self.color}",weight="bold")

        for i in range(self.total_masks//2):
            if i not in self.bad_mask:
                ax[0].set_title(f"Centers Fit; deg = {center_deg}")
                fit_eq = np.polyfit(x,centers[:,i],center_deg,w=amps[:,i])
                residual = np.poly1d(fit_eq)(x)-centers[:,i]
                ax[0].scatter(x,residual,alpha=0.05,c=amps[:,i],cmap="magma")

                ax[1].set_title(f"Sigmas Fit; deg = {sigma_deg}")
                fit_eq = np.polyfit(x,sigmas[:,i],sigma_deg,w=amps[:,i])
                residual = np.poly1d(fit_eq)(x)-sigmas[:,i]
                ax[1].scatter(x,residual,alpha=0.05,c=amps[:,i],cmap="magma")

        ax[0].set_xlabel("x")
        ax[0].set_ylabel("center residual")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("sigma residual")
        ax[0].set_ylim(-0.2,0.2)
        ax[1].set_ylim(-0.2,0.2)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(10)
        plt.close()
        
    def get_flat_traces(self,center_deg,sigma_deg) -> None:
        npzfile = np.load(self.datadir)
        x = npzfile["x"]
        centers = npzfile["centers"]
        sigmas = npzfile["sigmas"]
        amps = npzfile["amps"]

        traces = np.empty(shape=(self.total_masks//2,center_deg+1))
        traces_sigma = np.empty(shape=(self.total_masks//2,sigma_deg+1))
        for i in range(self.total_masks//2):
            if i not in self.bad_mask:
                fit_eq = np.polyfit(x,np.array(centers)[:,i],center_deg,w=np.array(amps)[:,i])
                fit_eq_sig = np.polyfit(x,np.array(sigmas)[:,i],sigma_deg,w=np.array(amps)[:,i])
                traces[i,:] = fit_eq
                traces_sigma[i,:] = fit_eq_sig
            else:
                traces[i,:] = np.nan
                traces_sigma[i,:] = np.nan

        save_dict = dict(npzfile)
        save_dict["traces"] = traces
        save_dict["traces_sigma"] = traces_sigma
        np.savez(self.datadir, **save_dict)
    
    def create_mask(self,sig_mult,mode="flat",copy=None) -> None:
        if mode=="flat":
            npzfile = np.load(self.datadir)
            save_dir = self.maskdir
            traces_sigma = npzfile["traces_sigma"]   
        else:
            file = os.path.join(os.path.abspath("out"),mode+self.color+"_trace_fits.npz")
            npzfile = np.load(file)
            save_dir = os.path.join(os.path.abspath("out"),mode+self.color+"_mask.fits")
            init_traces = npzfile["init_traces"]
            traces_sigma = npzfile["init_traces_sigma"]
        traces = npzfile["traces"]
        flat_data = fits.open(self.flatdir)[0].data

        if mode!="flat" and np.array_equiv(traces,init_traces): # if traces are the same, do not waste time creating mask. instead use arc's
            fits.writeto(save_dir, data=fits.open(self.maskdir)[0].data, overwrite=True)
        elif copy is not None and np.array_equiv(traces,init_traces):
            arcmask = os.path.join(os.path.abspath("out"),copy+self.color+"_mask.fits")
            fits.writeto(save_dir, data=fits.open(arcmask)[0].data, overwrite=True)
        else:
            new_mask = np.zeros(flat_data.shape)
            for mask in range(self.total_masks//2):
                if mask not in self.bad_mask:
                    for i in range(flat_data.shape[1]):
                        mask_center_i = round(np.poly1d(traces[mask])(i))
                        mask_sig_i = round(sig_mult*np.poly1d(traces_sigma[mask])(i))
                        new_mask[mask_center_i-mask_sig_i:mask_center_i+mask_sig_i+1,i] = mask+1
                # else:
                    # takes last mask, shifts up, considers that a proxy; does not overlap other masks
                    # bad_mask = new_mask==mask
                    # bad_mask = np.vstack((np.zeros((mask_sig_i+1,bad_mask.shape[1])),bad_mask[:-(mask_sig_i+1)]))
                    # new_mask[(bad_mask==1)&(new_mask!=mask)] = mask+1

            fits.writeto(save_dir, data=new_mask, overwrite=True)


    
    def optimize_trace(self,filename,sig_mult,cmrays=False,expected_peaks=30,optimize=True) -> None:
        arcdir = os.path.join(os.path.abspath("out"),filename+self.color+".fits")
        data = fits.open(arcdir)[0].data

        if cmrays:
            cmraydir = os.path.join(os.path.abspath("out"),filename+self.color+"_cmray_mask.fits")
            cmray_data = fits.open(cmraydir)[0].data

        mask_data = fits.open(self.maskdir)[0].data

        npzfile = np.load(self.datadir)
        traces = npzfile["traces"]
        traces_sigma = npzfile["traces_sigma"]

        masks_l = np.unique(mask_data)[1:]

        # returns intensity projections for each mask, along with cross-correlation based "lags"
        intensities = np.empty((self.total_masks//2-len(self.bad_mask),data.shape[1]))
        lags = np.empty(self.total_masks//2-len(self.bad_mask))
        
        x = np.arange(0,data.shape[1],1)
        if cmrays: 
            ref_a = get_spectrum_simple(data,mask_data,1,cmray_data)
            for i,m in enumerate(masks_l):
                a = get_spectrum_simple(data,mask_data,m,cmray_data)
                intensities[i] = a
                lags[i] = get_lag(a,ref_a)
        else:
            ref_a = get_spectrum_simple(data,mask_data,1)
            for i,m in enumerate(masks_l):
                a = get_spectrum_simple(data,mask_data,m)
                intensities[i] = a
                lags[i] = get_lag(a,ref_a)

        norm_intensities = np.empty(intensities.shape)
        for i,intensity in enumerate(intensities):
            norm_intensities[i] = np.interp(x, x-lags[i], intensity)
            # plt.plot(norm_intensities[i],alpha=0.03,color="gray")
        ref_intensity = np.nanmedian(norm_intensities,axis=0)
        # plt.plot(ref_intensity,color="darkorange")
        # plt.show()

        # gets peak areas for the reference intensity given a certain percentile cutoff
        # perc_cut = 95
        # masked = x[ref_intensity>np.nanpercentile(ref_intensity,perc_cut)]
        # peak_areas = np.split(masked,np.where(np.diff(masked) != x[1]-x[0])[0]+1)
    
        # peak_xs = np.empty((len(peak_areas),len(intensities)))
        # for i,peak_area in enumerate(peak_areas):
        #     for j,a in enumerate(intensities):
        #         peak_xs[i][j] = ifum_utils.get_peak_center(peak_area,4,4,x,a,lags[j])


        # splits into 3 quadrants to better account for quadratic diffs (is 3 good?)
        quadrants = 3
        quadrants_x = np.array_split(x,quadrants)
        peak_xs = np.empty((0,len(intensities)))
        peak_ints = np.array([])
        for quadrant in range(quadrants):
            quad_mask = quadrants_x[quadrant]
            quadrant_ref_a = intensities[0]

            quad_lags = np.empty(intensities.shape[0])
            norm_intensities = np.empty((intensities.shape[0],quad_mask.size))
            for i,intensity in enumerate(intensities):
                quad_mask_ = np.intersect1d(np.union1d(quad_mask,quad_mask+int(lags[i])),
                                            np.arange(intensities.shape[1]))
                quad_lag = get_lag(intensity[quad_mask_],quadrant_ref_a[quad_mask_])
                quad_lags[i] = quad_lag
                try:
                    norm_intensities[i] = intensity[quad_mask+quad_lag]
                except:
                    norm_intensities[i] = np.interp(quad_mask, quad_mask-quad_lag, intensity[quad_mask])
                # plt.plot(quad_mask,norm_intensities[i],alpha=0.01)
                
            ref_intensity = np.nanmedian(norm_intensities,axis=0)
            # plt.plot(quad_mask,ref_intensity)

            peaks,_ = scipy.signal.find_peaks(ref_intensity,
                                              distance=np.nanmean(traces_sigma)*sig_mult*8)
            expected_p = int(len(peaks)*0.5) # only include top 50%
            if len(peaks) > expected_p:
                top_args = ref_intensity[peaks].argsort()[-expected_p:][::-1]
                peaks = np.sort(peaks[top_args])

            # for peak in peaks:
            #     plt.axvline(peak+quad_mask[0],color="red")

            filtered_peaks = []
            threshold = np.nanmean(traces_sigma)*sig_mult*8*4
            for peak,intens in zip(peaks,ref_intensity[peaks]):
                if len(filtered_peaks)==0 or peak-filtered_peaks[-1]>threshold:
                    filtered_peaks.append(peak)
                else:
                    if intens > ref_intensity[filtered_peaks[-1]]:
                        filtered_peaks[-1] = peak

            # for peak in filtered_peaks:
            #     plt.axvline(peak+quad_mask[0],color="green")
            peaks = np.array(filtered_peaks)
            
            peak_xs_ = np.empty((len(peaks),len(intensities)))
            for i,peak in enumerate(peaks):
                for j,a in enumerate(intensities):
                    peak_xs_[i][j] = peak + lags[j] + quad_mask[0]
            peak_xs = np.vstack((peak_xs,peak_xs_))
            peak_ints = np.append(peak_ints,ref_intensity[peaks]) 

        # only chooses up to expected peaks #
        if peak_xs.shape[0] > expected_peaks:
            top_args = np.array(peak_ints).argsort()[-expected_peaks:][::-1]
            peak_xs = peak_xs[top_args,:]
        else:
            print(f"{filename}{self.color}: only {peak_xs.shape[0]}/{expected_peaks} peaks detected")

        # for peak in peak_xs:
        #     plt.axvline(peak[0],color="blue")
        # plt.show()

        # peaks,_ = scipy.signal.find_peaks(ref_intensity,
        #                                   distance=np.nanmean(traces_sigma)*sig_mult*8)
        # if len(peaks) > expected_peaks:
        #     top_args = ref_intensity[peaks].argsort()[-expected_peaks:][::-1]
        #     peaks = np.sort(peaks[top_args])
        # peak_xs = np.empty((len(peaks),len(intensities)))
        # for i,peak in enumerate(peaks):
        #     for j,a in enumerate(intensities):
        #         peak_xs[i][j] = peak + lags[j]

        # plt.figure(figsize=(100,3))
        # plt.plot(ref_intensity)
        # plt.vlines(peaks,plt.gca().get_ylim()[0],plt.gca().get_ylim()[1])
        # plt.show()
        # print(peak_xs)
        
        # get centers for each peak area for each mask
        centers = np.empty((len(peaks),len(intensities)))
        for peak in range(len(peaks)):
            for i,mask in enumerate(masks_l):
                offset = math.ceil(sig_mult*np.median(np.poly1d(traces_sigma[int(mask-1)])(np.arange(data.shape[0]))))
                try:
                    if (peak_xs[peak][i]+offset)<(~np.isnan(intensities[i])).cumsum(0).argmax(0):
                        mask_area = ((x)>(peak_xs[peak][i]-(offset+1)))&((x)<(peak_xs[peak][i]+(offset+1)))    
                        mask_area = mask_area&(~np.isnan(intensities[i]))
                        p0 = [0,np.max(intensities[i][mask_area]),np.argmax(intensities[i][mask_area])+np.min(x[mask_area]),offset/3]
                        popt,_ = scipy.optimize.curve_fit(gauss,x[mask_area],intensities[i][mask_area],p0=p0)                            
                    else:
                        mask_area = x>((~np.isnan(intensities[i])).cumsum(0).argmax(0)-offset*2)
                        mask_area = mask_area&(~np.isnan(intensities[i]))
                        p0 = [0,np.max(intensities[i][mask_area]),np.argmax(intensities[i][mask_area])+np.min(x[mask_area]),offset/3]
                        popt,_ = scipy.optimize.curve_fit(gauss,x[mask_area],intensities[i][mask_area],p0=p0)
                    centers[peak][i] = popt[2]
                except:
                    # print(f"bad 1D peak fit: color {self.color}, mask {mask}, peak {peak}")
                    centers[peak][i] = np.nan


        if optimize:
            results_arr = np.empty((self.total_masks//2-len(self.bad_mask),centers.shape[0],7))
            fit_xs = np.empty((self.total_masks//2-len(self.bad_mask),centers.shape[0]))
            fit_ys = np.empty((self.total_masks//2-len(self.bad_mask),centers.shape[0]))
            rotations = np.empty((self.total_masks//2-len(self.bad_mask),centers.shape[0]))

            if cmrays:
                data[cmray_data==1] = np.nan
        
            for i,mask in enumerate(masks_l):
                for line in np.arange(centers.shape[0]):
                    try:
                        x0,y0 = centers[line,i],np.poly1d(traces[int(mask-1)])(centers[line,i])
                        x_off,y_off = round(1.75*np.poly1d(traces_sigma[int(mask-1)])(x0)),round(2.5*np.poly1d(traces_sigma[int(mask-1)])(x0))
                        
                        adata_ = data[round(y0)-y_off:round(y0)+y_off+1,round(x0)-x_off:round(x0)+x_off+1]
                        
                        theta = np.pi
                        sig_x = 2
                        sig_y = np.poly1d(traces_sigma[int(mask-1)])(x0)
                        
                        a = np.cos(theta)**2/(2*sig_x**2)+np.sin(theta)**2/(2*sig_y**2)
                        b = -1.*np.sin(theta)*np.cos(theta)/(2*sig_x**2)+np.sin(theta)*np.cos(theta)/(2*sig_y**2)
                        c = np.sin(theta)**2/(2*sig_x**2)+np.cos(theta)**2/(2*sig_y**2)
                        
                        
                        var = [0,0,1,a,b,c,0]
                        args = [x0,y0,x_off,y_off,adata_,False]
                        
                        result = scipy.optimize.minimize(minimize_gauss_2d, var, args=(args))
                        results_arr[i,line] = result.x
                        fit_xs[i,line] = x0+result.x[0]
                        fit_ys[i,line] = y0+result.x[1]
                        rotations[i,line] = 0.5*np.arctan(2*result.x[4]/(result.x[3]-result.x[5]))

                        # args[-1] = True
                        # res = ifum_utils.minimize_gauss_2d(result.x,args=(args))
                        # print(0.5*np.arctan(2*result.x[4]/(result.x[3]-result.x[5])),res)

                        # if rotations[i,line] < 0.75 or rotations[i,line] > 1.5 or res > 1:
                        #     # print(f"bad 2D gauss fit: color {self.color}, mask {mask}, peak {peak}")
                        #     results_arr[i,line] = np.nan
                        #     fit_xs[i,line] = np.nan
                        #     fit_ys[i,line] = np.nan
                        #     rotations[i,line] = np.nan

                    except:
                        # print(f"bad 2D gauss fit: color {self.color}, mask {mask}, peak {peak}")
                        results_arr[i,line] = np.nan
                        fit_xs[i,line] = np.nan
                        fit_ys[i,line] = np.nan
                        rotations[i,line] = np.nan

            center_traces = np.empty(shape=(masks_l.shape[0],traces.shape[1]+1))
            for i,mask in enumerate(masks_l):
                var = [0,0]
                args = [fit_xs[i],fit_ys[i],traces[int(mask-1)]]
                result = scipy.optimize.minimize(minimize_poly_dist, var, args=(args))
                center_traces[i] = np.polymul(traces[int(mask-1)],result.x)

            if np.intersect1d(self.bad_mask,np.arange(self.total_masks//2)).size > 0:
                _, _, ind2 = np.intersect1d(self.bad_mask,np.arange(self.total_masks//2),return_indices=True)
                for mask_ in ind2:
                    center_traces = np.insert(center_traces, mask_, np.nan, axis=0)
        else:
            # still will get the fits for the centers, without optimizing 2d gauss
            fit_xs = np.empty((self.total_masks//2-len(self.bad_mask),centers.shape[0]))
            fit_ys = np.empty((self.total_masks//2-len(self.bad_mask),centers.shape[0]))

            if cmrays:
                data[cmray_data==1] = np.nan
        
            for i,mask in enumerate(masks_l):
                for line in np.arange(centers.shape[0]):
                    x0,y0 = centers[line,i],np.poly1d(traces[int(mask-1)])(centers[line,i])
                    fit_xs[i,line] = x0
                    fit_ys[i,line] = y0
    
            center_traces = traces
            results_arr = np.nan
            rotations = np.nan
        # plot_masks = [0,1,2,3]
        # plt.figure(dpi=2000)
        # plt.imshow(data[:150,:],origin="lower")
        # for i in plot_masks:
        #     plt.scatter(fit_xs[i],fit_ys[i],s=0.05,color="orange")
        #     plt.plot(np.arange(data.shape[1]),np.poly1d(center_traces[i])(np.arange(data.shape[1])),lw=0.5,alpha=0.5)
        # plt.axis("off")
        # plt.show(block=False)
        # plt.pause(10)
        # plt.close()

                
        save_dict = {'init_traces': traces,
                     'init_traces_sigma': traces_sigma,
                     'centers': centers,
                     'results_arr': results_arr,
                     'fit_xs': fit_xs,
                     'fit_ys': fit_ys, 
                     'rotations': rotations,
                     'traces': center_traces}

        np.savez(os.path.join(os.path.abspath("out"),filename+self.color+"_trace_fits.npz"), **save_dict)
        


    def get_rots(self,arcfilename,datafilename,optimize=True) -> None:
        arc_npz = np.load(os.path.join(os.path.abspath("out"),arcfilename+self.color+"_trace_fits.npz"))
        data_npz = np.load(os.path.join(os.path.abspath("out"),datafilename+self.color+"_trace_fits.npz"))
        arc_rot = arc_npz["rotations"]
        arc_xs = arc_npz["fit_xs"]
        data_rot = data_npz["rotations"]
        data_xs = data_npz["fit_xs"]

        if optimize:
            deg = 2
            rot_fit = np.empty(shape=(self.total_masks//2-len(self.bad_mask),deg+1))
            for i in np.arange(self.total_masks//2-len(self.bad_mask)):
                all_rots = np.concatenate([arc_rot[i],data_rot[i]])
                all_xs = np.concatenate([arc_xs[i],data_xs[i]])
                bad0 = (np.isnan(all_rots))|(np.isnan(all_xs))
                rot_fit_poly = np.polyfit(all_xs[~bad0],all_rots[~bad0],deg)
                bad = abs(all_rots-np.poly1d(rot_fit_poly)(all_xs))>np.std(all_rots-np.poly1d(rot_fit_poly)(all_xs))
                bad = bad|bad0
                rot_fit_poly = np.polyfit(all_xs[~bad],all_rots[~bad],deg)
                rot_fit[i] = np.poly1d(rot_fit_poly)
                
                # plt.title(i)
                # plt.scatter(arc_xs[i],arc_rot[i])
                # plt.scatter(data_xs[i],data_rot[i])
                # fx = np.linspace(0,2048,1000)
                # plt.plot(fx,np.poly1d(rot_fit_poly)(fx))
                # plt.show()
            if np.intersect1d(self.bad_mask,np.arange(self.total_masks//2)).size > 0:
                _, _, ind2 = np.intersect1d(self.bad_mask,np.arange(self.total_masks//2),return_indices=True)
                for mask_ in ind2:
                    rot_fit = np.insert(rot_fit, mask_, np.nan, axis=0)
        else:
            rot_fit = np.nan

        save_dict = dict(data_npz)
        save_dict["rotation_traces"] = rot_fit
        np.savez(os.path.join(os.path.abspath("out"),datafilename+self.color+"_trace_fits.npz"), **save_dict)

    def _viz(self,datafilename,sig_mult,masks) -> None:
        data_dir = os.path.join(os.path.abspath("out"),datafilename+self.color+".fits")
        data = fits.open(data_dir)[0].data

        data_npz = np.load(os.path.join(os.path.abspath("out"),datafilename+self.color+"_trace_fits.npz"))
        sigs = data_npz["init_traces_sigma"]
        traces = data_npz["traces"]
        rots = data_npz["rotation_traces"]

        plt.figure(dpi=2500)
        plt.imshow(data[:150,:],vmax=1500,vmin=0,origin="lower")
        for mask in masks:
            l = sig_mult*np.poly1d(sigs[mask])(np.arange(0,2048))
            m = np.tan(np.pi/2-np.poly1d(rots[mask])(np.arange(0,2048)))
            delta_x = np.sin(np.arctan((-1/m)))*l
            delta_y = np.cos(np.arctan((-1/m)))*l
            print(delta_x,delta_y)
            for pix in np.arange(0,2048):
                plt.plot([pix-delta_x[pix],pix+delta_x[pix]],[np.poly1d(traces[mask])(pix)-delta_y[pix],np.poly1d(traces[mask])(pix)+delta_y[pix]],lw=0.05,alpha=0.5,color="orange")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(60)
        plt.close()

    def _viz_(self,datafilename,color,mask) -> None:
        import os
        data_file = os.path.join(os.path.abspath("out"),datafilename+color+".fits")
        # cmray_file = os.path.join(os.path.abspath("out"),datafilename+color+"_cmray_mask.fits")
        data_npz = np.load(os.path.join(os.path.abspath("out"),datafilename+self.color+"_trace_fits.npz"))

        data = fits.open(data_file)[0].data
        # cmray_mask = fits.open(cmray_file)[0].data

        # data[cmray_mask==1] = np.nan

        centers = data_npz["centers"]
        fit_xs = data_npz["fit_xs"]
        fit_ys = data_npz["fit_ys"]
        center_fits = data_npz["init_traces"]
        center_fits_data = data_npz["traces"]
        sig_fits = data_npz["init_traces_sigma"]
        results_arr = data_npz["results_arr"]

        line = 2
        if ~np.isnan(centers[line,mask]):
            print("line",line)
            x0,y0 = centers[line,mask],np.poly1d(center_fits[mask])(centers[line,mask])
            x_off,y_off = round(1.75*np.poly1d(sig_fits[mask])(x0)),round(2.5*np.poly1d(sig_fits[mask])(x0))
            
            data_ = data[round(y0)-y_off:round(y0)+y_off+1,round(x0)-x_off:round(x0)+x_off+1]
        
            # var = [0,0,1,a,b,c,0]
            args = [x0,y0,x_off,y_off,data_,True]

        print(results_arr.shape)

        x_,y_,A,a,b,c,H = results_arr[mask][line]
        x0,y0,x_off,y_off,data__,plot = args

        x = np.arange(x0-x_off,x0+x_off+0.9,1.)
        y = np.arange(y0-y_off,y0+y_off+0.9,1.)
        xv,yv = np.meshgrid(x,y)

        gauss_points = gauss_2d(xv,yv,x0+x_,y0+y_,A,a,b,c,H)
        image_points = (data__-np.nanmin(data__))/(np.nanmax(data__)-np.nanmin(data__))

        res = np.nanmean((image_points-gauss_points)**2)


        vmin,vmax = np.nanmin(image_points),np.nanmax(image_points)
        plt.figure(dpi=200,figsize=(10,4.5)).set_facecolor("#C7B299")
        plt.suptitle("'red' data",weight="bold",color="darkred")
        plt.subplot(2,7,1)
        plt.title("gaussian",weight="bold")
        plt.imshow(gauss_points,origin="lower",cmap="magma",vmin=vmin,vmax=vmax)
        cbar = plt.colorbar(orientation="horizontal", pad=0.0, shrink=0.9)
        cbar.set_ticks([0,0.5,1])
        plt.axis("equal")
        plt.axis("off")
        plt.subplot(2,7,2)
        plt.title("data",weight="bold")
        plt.imshow(image_points,origin="lower",cmap="magma",vmin=vmin,vmax=vmax)
        cbar = plt.colorbar(orientation="horizontal", pad=0.0, shrink=0.9)
        cbar.set_ticks([0,0.5,1])
        plt.axis("equal")
        plt.axis("off")
        plt.subplot(2,7,3)
        plt.title("residual",weight="bold")
        vmin,vmax = np.nanmin(image_points-gauss_points),np.nanmax(image_points-gauss_points)
        vminmax = .35
        plt.imshow(image_points-gauss_points,origin="lower",cmap="PuOr",vmin=-vminmax,vmax=vminmax)
        aspect = 20
        cbar = plt.colorbar(orientation="horizontal", pad=0.0, shrink=0.9)
        cbar.set_ticks([-0.3, 0.0, 0.3])
        plt.axis("equal")
        plt.axis("off")



        line = 10
        if ~np.isnan(centers[line,mask]):
            print("line",line)
            x0,y0 = centers[line,mask],np.poly1d(center_fits[mask])(centers[line,mask])
            x_off,y_off = round(1.75*np.poly1d(sig_fits[mask])(x0)),round(2.5*np.poly1d(sig_fits[mask])(x0))
            
            data_ = data[round(y0)-y_off:round(y0)+y_off+1,round(x0)-x_off:round(x0)+x_off+1]
        
            var = [0,0,1,a,b,c,0]
            args = [x0,y0,x_off,y_off,data_,True]

        print(results_arr[mask][line])

        x_,y_,A,a,b,c,H = results_arr[mask][line]
        x0,y0,x_off,y_off,data__,plot = args

        x = np.arange(x0-x_off,x0+x_off+0.9,1.)
        y = np.arange(y0-y_off,y0+y_off+0.9,1.)
        xv,yv = np.meshgrid(x,y)

        gauss_points = gauss_2d(xv,yv,x0+x_,y0+y_,A,a,b,c,H)
        image_points = (data__-np.nanmin(data__))/(np.nanmax(data__)-np.nanmin(data__))

        res = np.nanmean((image_points-gauss_points)**2)

        vmin,vmax = np.nanmin(image_points),np.nanmax(image_points)
        plt.subplot(2,7,5)
        plt.title("gaussian",weight="bold")
        plt.imshow(gauss_points,origin="lower",cmap="magma",vmin=vmin,vmax=vmax)
        cbar = plt.colorbar(orientation="horizontal", pad=0.0, shrink=0.9)
        cbar.set_ticks([0,0.5,1])
        plt.axis("equal")
        plt.axis("off")
        plt.subplot(2,7,6)
        plt.title("data",weight="bold")
        plt.imshow(image_points,origin="lower",cmap="magma",vmin=vmin,vmax=vmax)
        cbar = plt.colorbar(orientation="horizontal", pad=0.0, shrink=0.9)
        cbar.set_ticks([0,0.5,1])
        plt.axis("equal")
        plt.axis("off")
        plt.subplot(2,7,7)
        plt.title("residual",weight="bold")
        vmin,vmax = np.nanmin(image_points-gauss_points),np.nanmax(image_points-gauss_points)
        vminmax = .35
        plt.imshow(image_points-gauss_points,origin="lower",cmap="PuOr",vmin=-vminmax,vmax=vminmax)
        aspect = 20
        cbar = plt.colorbar(orientation="horizontal", pad=0.0, shrink=0.9)
        cbar.set_ticks([-0.3, 0.0, 0.3])
        plt.axis("equal")
        plt.axis("off")
        import os


        data_file = os.path.join(os.path.abspath("out"),datafilename+color+".fits")
        data = fits.open(data_file)[0].data
        data = data[25:125,850:1250]

        plt.subplot(2,2,3)
        plt.title("mask 3 (lower λ)",weight="bold")
        plt.imshow(data,origin="lower",vmin=np.percentile(data,1),vmax=np.percentile(data,99),cmap="Greys_r")
        mask = 3
        plt.plot(np.arange(850,1250)-850,np.poly1d(center_fits[mask])(np.arange(850,1250))-25,lw=0.5,c="red")
        plt.plot(np.arange(850,1250)-850,np.poly1d(center_fits_data[mask])(np.arange(850,1250))-25,lw=0.5,c="gold")

        for line in np.arange(5):
            if line == 2 or line == 28:
                plt.scatter(fit_xs[mask][line]-850,fit_ys[mask][line]-25,color="gold",s=100,marker="x")
            else:
                plt.scatter(fit_xs[mask][line]-850,fit_ys[mask][line]-25,color="sienna",s=100,marker="x")

        plt.axis("off")



        plt.subplot(2,2,4)

        data_file = os.path.join(os.path.abspath("out"),datafilename+color+".fits")
        data = fits.open(data_file)[0].data
        data = data[50:125,-500:]
        plt.title("mask 3 (higher λ)",weight="bold")

        plt.imshow(data,origin="lower",vmin=np.percentile(data,1),vmax=np.percentile(data,99),cmap="Greys_r")
        mask = 3
        plt.plot(np.arange(2048-500,2048)-2048+500,np.poly1d(center_fits[mask])(np.arange(2048-500,2048))-50,lw=0.5,c="red")
        plt.plot(np.arange(2048-500,2048)-2048+500,np.poly1d(center_fits_data[mask])(np.arange(2048-500,2048))-50,lw=0.5,c="gold")

        for line in np.arange(centers.shape[0]-5,centers.shape[0]):
            print(line)
            if line == 2 or line == 27:
                plt.scatter(fit_xs[mask][line]-2048+500,fit_ys[mask][line]-50,color="gold",s=100,marker="x")
            else:
                plt.scatter(fit_xs[mask][line]-2048+500,fit_ys[mask][line]-50,color="sienna",s=100,marker="x")

        plt.axis("off")

        plt.tight_layout(pad=1)
        # plt.savefig("im06.png",dpi=400,bbox_inches='tight',pad_inches=0.1)
        plt.show()




@python_app
def minimize_gauss_fit(xs, ys, p0, lbounds, hbounds):
    import scipy

    try:
        popt,_ = scipy.optimize.curve_fit(
            gauss_added,
            xs,
            ys,
            p0=p0,
            bounds=(lbounds,hbounds),
            nan_policy="omit",
            method="trf",
            maxfev=1000
        )
        return popt
    except:
        return None

def multi_gauss_fit(x,mask_polys,masks_split,flat_data,bad_mask):
    cutoffs = [5]
    for i in np.arange(1,len(masks_split)):
        first = mask_polys[masks_split[i-1]][~np.isnan(mask_polys[masks_split[i-1]]).any(axis=1)][-1]
        last = mask_polys[masks_split[i]][~np.isnan(mask_polys[masks_split[i]]).any(axis=1)][0]
        cutoffs.append(int((np.poly1d(first)(x)+np.poly1d(last)(x))/2))
    cutoffs.append(flat_data.shape[0]-5)
    cutoffs = np.array(cutoffs)

    # double check that 0::6 to be sure that it is working for non-STD formats
    continuum,_ = scipy.optimize.curve_fit(f_2,cutoffs[0::6],flat_data[cutoffs[0::6],x])

    # print(continuum,flush=True)

    fit_futures = []
    for i, mask_group in enumerate(masks_split):
        p0 = []
        lbounds = []
        hbounds = []
        
        # append quadratic shift guess
        p0.append(continuum[0])
        l,h = generate_bounds(continuum[0],.3)
        lbounds.append(l)
        hbounds.append(h)
        p0.append(continuum[1])
        l,h = generate_bounds(continuum[1],.3)
        lbounds.append(l)
        hbounds.append(h)
        p0.append(continuum[2])
        lbounds.append(-np.inf)
        hbounds.append(np.inf)

        for mask in mask_group:
            if mask not in bad_mask:
                # print(f"mask {mask} for x {x}", flush=True)
                # print(*continuum, flush=True)
                # print(np.poly1d(mask_polys[mask])(x), flush=True)

                # append amplitude guess
                p0.append(abs(flat_data[int(np.round(np.poly1d(mask_polys[mask])(x))),x]-f_2(int(np.round(np.poly1d(mask_polys[mask])(x))),*continuum)))
                lbounds.append(0)
                hbounds.append(np.max(flat_data[cutoffs[i]:cutoffs[i+1],x])*2)
                # append center guess
                p0.append(np.poly1d(mask_polys[mask])(x))
                lbounds.append(np.poly1d(mask_polys[mask])(x)-2) # +/- 2 pixels from first guess
                hbounds.append(np.poly1d(mask_polys[mask])(x)+2)
                # append sigma guess
                p0.append(3.)
                lbounds.append(1.5)
                hbounds.append(5.5)

        # print(f"{x} optimization started: mask group {i}", flush=True)

        xs = np.arange(cutoffs[i],cutoffs[i+1])
        ys = flat_data[cutoffs[i]:cutoffs[i+1],x]

        fit_futures.append(
            minimize_gauss_fit(xs, ys, p0, lbounds, hbounds)
        )

    # print(f"{x}: all fits submitted", flush=True)
    # print(fit_futures)

    return fit_futures
    
def mask_poly(mask_polys, n, flatdir, total_masks, mask_groups, bad_mask, datadir) -> None:
    import numpy as np
    import scipy
    from astropy.io import fits

    with fits.open(flatdir) as flatf:
        flat_data = flatf[0].data
    flat_data = scipy.ndimage.median_filter(flat_data,size=(1,3))

    weight = np.log10(np.median(flat_data,axis=0))
    weight = np.array(weight - np.min(weight))[::3]
    lines = np.random.choice(np.arange(0,flat_data.shape[1],3),
                                size=n, 
                                replace=False, 
                                p=weight/np.sum(weight))

    gauss_centers_full = np.empty((0,total_masks//2))
    gauss_sigmas_full = np.empty((0,total_masks//2))
    gauss_amps_full = np.empty((0,total_masks//2))
    
    masks_split = np.array(np.split(np.arange(total_masks//2),mask_groups))

    x_s = lines
    multi_fits = []
    for x in x_s:
        multi_fits.append(
            multi_gauss_fit(x,mask_polys,masks_split,flatdir,bad_mask)
        )
        # print(f"fit {x} submitted", flush=True)

    # print("all fits submitted", flush=True)
    
    bad_lines = []
    for i,future in enumerate(multi_fits):
        # print(i,future, flush=True)
        if future.result() is None:
        # if future is None:
            bad_lines.append(x_s[i])
        else:
            centers,sigmas,amps = future.result()
            # centers,sigmas,amps = future
            print(i,"complete", flush=True)
            gauss_centers_full = np.vstack((gauss_centers_full,np.array(centers).flatten()))
            gauss_sigmas_full = np.vstack((gauss_sigmas_full,np.array(sigmas).flatten()))
            gauss_amps_full = np.vstack((gauss_amps_full,np.array(amps).flatten()))

    save_dict = {'x': x_s[np.isin(x_s, bad_lines, invert=True)],
                 'centers': gauss_centers_full,
                 'sigmas': gauss_sigmas_full,
                 'amps': gauss_amps_full}

    np.savez(datadir, **save_dict)

    return None

# @python_app
def flat_mask_app(dep_futures,mask_args,polydeg=3,sampling=5):
    mask = Mask(**mask_args)
    [f.result() for f in dep_futures]

    mask_polys0 = mask.first_guess(polydeg)
    mask_poly(mask_polys0,
              sampling,
              mask.flatdir,
              mask.total_masks,
              mask.mask_groups,
              mask.bad_mask,
              mask.datadir)

    # mask.mask_poly(mask_polys0,sampling)
    return None





# NEW APPROACH
# want to run all multi-gauss fits in parallel at once
# submit for different files, for different x values, for different mask groups

# PARALLELIZE FIRST GUESS
@python_app
def first_guess_app(dep_futures, mask_args, polydeg=3):
    mask = Mask(**mask_args)
    [f.result() for f in dep_futures]

    return mask.first_guess(polydeg)

def flat_mask_wrapper(first_guesses, mask_args_s, polydeg=3, sampling=40):
    colors = ["b", "r"]
    all_fit_futures = {"b": {}, "r": {}}
    original_lines = {"b": {}, "r": {}}

    # given both mask_args and first_guesses are double dictionaries
    # where keys are colors and each subkey is a flat filename
    # we will iterate through colors and then through each flat file
    # and submit multi-gauss fits for each x value in the flat file

    for color in colors:
        first_guesses_c = first_guesses[color]
        mask_args_s_c = mask_args_s[color]

        for mask_args in mask_args_s_c.values():
            # print(f"IN FUNCTION: {mask_args}", flush=True)
            mask = Mask(**mask_args)
            all_fit_futures[color][mask.flatfilename] = []

            first_guess = first_guesses_c[mask.flatfilename].result()

            # init mask, get sampled lines
            with fits.open(mask.flatdir) as flatf:
                flat_data = flatf[0].data
            flat_data = scipy.ndimage.median_filter(flat_data, size=(1, 3))
            weight = np.log10(np.median(flat_data, axis=0))
            weight = np.array(weight - np.min(weight))[::3]
            lines = np.random.choice(
                np.arange(0, flat_data.shape[1], 3),
                size=sampling,
                replace=False,
                p=weight/np.sum(weight)
            )
            original_lines[color][mask.flatfilename] = lines

            masks_split = np.array(np.split(np.arange(mask.total_masks//2), mask.mask_groups))

            x_s = lines
            for x in x_s:
                # print(mask.bad_mask, flush=True)
                fit_futures = multi_gauss_fit(
                    x, first_guess, masks_split, flat_data, mask.bad_mask
                )
                all_fit_futures[color][mask.flatfilename].append(fit_futures)
                # print(f"fit {x} submitted", flush=True)

            # print("all fits submitted", flush=True)

            # show length of fit futures for each color and flat file
            # print(f"{mask.flatfilename}{color}: {len(all_fit_futures[color][mask.flatfilename])} x {len(all_fit_futures[color][mask.flatfilename][0])}", flush=True)

    # print("both colors & all files submitted", flush=True)

    for color in colors:
        for flat_filename, fit_futures in all_fit_futures[color].items():
            mask = Mask(**mask_args_s[color][flat_filename])
            # print(mask_args_s[color][flat_filename], flush=True)
            # print(f"{flat_filename}{color}: {fit_futures}")
            # print(fit_futures==all_fit_futures[color][flat_filename])

            # for each mask group, retrieve results from futures
            gauss_centers_full = np.empty((0,mask.total_masks//2))
            gauss_sigmas_full = np.empty((0,mask.total_masks//2))
            gauss_amps_full = np.empty((0,mask.total_masks//2))

            for i, fit_futs in enumerate(fit_futures):
                # print(f"{flat_filename}{color}: vertical slice {i}", flush=True)

                all_centers = np.full(mask.total_masks//2, np.nan)
                all_sigmas = np.full(mask.total_masks//2, np.nan)
                all_amps = np.full(mask.total_masks//2, np.nan)

                for j,future in enumerate(fit_futs):
                    # print(f"{flat_filename}{color}: {i} - mask group {j}", flush=True)
                    popt = future.result()
                    # print(popt, flush=True)

                    if popt is None:
                        # print(f"{flat_filename}{color}: bad fit at slice {i}", flush=True)
                        all_centers.append(np.full(mask.total_masks//2, np.nan))
                        all_sigmas.append(np.full(mask.total_masks//2, np.nan))
                        all_amps.append(np.full(mask.total_masks//2, np.nan))
                        # go to next fit_futs
                        continue

                    else:
                        center = np.array(popt[3:][1::3])
                        sigma = np.array(popt[3:][2::3])
                        amp = np.array(popt[3:][0::3])

                        # print(center.shape, sigma.shape, amp.shape, flush=True)


                        # Get the mask indices for this group
                        mask_indices = masks_split[j]
                        
                        # Skip bad masks within this group
                        valid_indices = [idx for idx in range(len(mask_indices)) 
                                        if mask_indices[idx] not in mask.bad_mask]
                        
                        # Place the results in the correct positions in the output arrays
                        for k, valid_idx in enumerate(valid_indices):
                            mask_idx = mask_indices[valid_idx]
                            all_centers[mask_idx] = center[k]
                            all_sigmas[mask_idx] = sigma[k]
                            all_amps[mask_idx] = amp[k]

                        # if np.intersect1d(mask.bad_mask, masks_split[i]).size > 0:
                        #     _, _, ind2 = np.intersect1d(mask.bad_mask, masks_split[i], return_indices=True)
                        #     for mask_ in ind2:
                        #         center = np.insert(center, mask_, np.nan, axis=0)
                        #         sigma = np.insert(sigma, mask_, np.nan, axis=0)
                        #         amp = np.insert(amp, mask_, np.nan, axis=0)

                        # print(center.shape, sigma.shape, amp.shape, flush=True)



                        # all_centers = np.concatenate((all_centers, center))
                        # all_sigmas = np.concatenate((all_sigmas, sigma))
                        # all_amps = np.concatenate((all_amps, amp))

                    # print(np.array(all_centers), flush=True)
                # print(np.array(all_centers).shape, flush=True)

                gauss_centers_full = np.vstack((gauss_centers_full,np.array(all_centers).flatten()))
                gauss_sigmas_full = np.vstack((gauss_sigmas_full,np.array(all_sigmas).flatten()))
                gauss_amps_full = np.vstack((gauss_amps_full,np.array(all_amps).flatten()))

                # print(gauss_centers_full.shape, flush=True)
            # print(gauss_centers_full.shape, flush=True)

            x_s = original_lines[color][flat_filename]
            # print(x_s.shape,gauss_centers_full.shape, gauss_sigmas_full.shape, gauss_amps_full.shape, flush=True)
            save_dict = {'x': x_s,#[np.isin(x_s, mask.bad_lines, invert=True)],
                        'centers': gauss_centers_full,
                        'sigmas': gauss_sigmas_full,
                        'amps': gauss_amps_full}

            np.savez(mask.datadir, **save_dict)

            # # retrieve results from futures
            # bad_lines = []
            # for future in fit_futures:
            #     if future.result() is None:
            #         bad_lines.append(x_s[i])
            #     else:
            #         centers, sigmas, amps = future.result()
            #         all_centers.append(centers)
            #         all_sigmas.append(sigmas)
            #         all_amps.append(amps)

#         centers = []
#         sigmas = []
#         amps = []
#         for i, mask_group in enumerate(masks_split):
#             popt = fit_futures[i].result()
            
#             # print(f"{x} optimization finished: mask group {i}", flush=True)
#             # print(f"{x}: popt: {popt}", flush=True)

#             if popt is None:
#                 # print(f"{x}: bad fit for mask group {i}", flush=True)
#                 return None

#             center = np.array(popt[3:][1::3])
#             sigma = np.array(popt[3:][2::3])
#             amp = np.array(popt[3:][0::3])

#             if np.intersect1d(bad_mask,mask_group).size > 0:
#                 _, _, ind2 = np.intersect1d(bad_mask,mask_group,return_indices=True)
#                 for mask_ in ind2:
#                     center = np.insert(center, mask_, np.nan, axis=0)
#                     sigma = np.insert(sigma, mask_, np.nan, axis=0)
#                     amp = np.insert(amp, mask_, np.nan, axis=0)

#             centers.append(center)
#             sigmas.append(sigma)
#             amps.append(amp)

#         return np.array(centers).flatten(),np.array(sigmas).flatten(),np.array(amps).flatten()









# with fits.open(flatdir) as flatf:
#         flat_data = flatf[0].data
#     flat_data = scipy.ndimage.median_filter(flat_data,size=(1,3))

#     weight = np.log10(np.median(flat_data,axis=0))
#     weight = np.array(weight - np.min(weight))[::3]
#     lines = np.random.choice(np.arange(0,flat_data.shape[1],3),
#                                 size=n, 
#                                 replace=False, 
#                                 p=weight/np.sum(weight))

#     gauss_centers_full = np.empty((0,total_masks//2))
#     gauss_sigmas_full = np.empty((0,total_masks//2))
#     gauss_amps_full = np.empty((0,total_masks//2))
    
#     masks_split = np.array(np.split(np.arange(total_masks//2),mask_groups))

#     x_s = lines
#     multi_fits = []
#     for x in x_s:
#         multi_fits.append(
#             multi_gauss_fit(x,mask_polys,masks_split,flatdir,bad_mask)
#         )
#         print(f"fit {x} submitted", flush=True)

#     print("all fits submitted", flush=True)
    
#     bad_lines = []
#     for i,future in enumerate(multi_fits):
#         # print(i,future, flush=True)
#         if future.result() is None:
#         # if future is None:
#             bad_lines.append(x_s[i])
#         else:
#             centers,sigmas,amps = future.result()
#             # centers,sigmas,amps = future
#             print(i,"complete", flush=True)
#             gauss_centers_full = np.vstack((gauss_centers_full,np.array(centers).flatten()))
#             gauss_sigmas_full = np.vstack((gauss_sigmas_full,np.array(sigmas).flatten()))
#             gauss_amps_full = np.vstack((gauss_amps_full,np.array(amps).flatten()))

#     save_dict = {'x': x_s[np.isin(x_s, bad_lines, invert=True)],
#                  'centers': gauss_centers_full,
#                  'sigmas': gauss_sigmas_full,
#                  'amps': gauss_amps_full}

#     np.savez(datadir, **save_dict)

#     return None

            
            
            # flat_mask_app(first_guess, mask_args, polydeg, sampling)




@python_app
def flat_mask_app(first_guess, mask_args, polydeg=3, sampling=5):
    mask = Mask(**mask_args)

    mask_polys0 = first_guess.result()
    mask_poly(mask_polys0,
              sampling,
              mask.flatdir,
              mask.total_masks,
              mask.mask_groups,
              mask.bad_mask,
              mask.datadir)

    return None


# def flat_mask_app(dep_futures, mask_args, polydeg=3, sampling=5):
#     """
#     Orchestrates the mask fitting process by submitting a fine-grained task
#     for every (x, mask_group) pair.
#     """
#     # 1. Initial setup
#     mask = Mask(**mask_args)
#     [f.result() for f in dep_futures]
#     mask_polys0 = mask.first_guess(polydeg)

#     # 2. Prepare data common to all fits to avoid repeated I/O
#     print("Preparing data for parallel fits...", flush=True)
#     with fits.open(mask.flatdir) as flatf:
#         flat_data = flatf[0].data
#     flat_data = scipy.ndimage.median_filter(flat_data, size=(1, 3))

#     weight = np.log10(np.median(flat_data, axis=0))
#     weight = np.array(weight - np.min(weight))[::3]
#     lines = np.random.choice(np.arange(0, flat_data.shape[1], 3),
#                              size=sampling,
#                              replace=False,
#                              p=weight / np.sum(weight))
    
#     masks_split = np.array(np.split(np.arange(mask.total_masks // 2), mask.mask_groups), dtype=object)

#     # 3. SCATTER: Loop through all x and mask_groups to submit all tasks
#     all_futures = {} # Use a dictionary to store futures, keyed by x
#     for x in lines:
#         # Pre-calculate cutoffs and continuum for this x value
#         cutoffs = [5]
#         for i in np.arange(1, len(masks_split)):
#             first = mask_polys0[masks_split[i-1]][~np.isnan(mask_polys0[masks_split[i-1]]).any(axis=1)][-1]
#             last = mask_polys0[masks_split[i]][~np.isnan(mask_polys0[masks_split[i]]).any(axis=1)][0]
#             cutoffs.append(int((np.poly1d(first)(x) + np.poly1d(last)(x)) / 2))
#         cutoffs.append(flat_data.shape[0] - 5)
#         cutoffs = np.array(cutoffs)
#         continuum, _ = scipy.optimize.curve_fit(f_2, cutoffs[0::6], flat_data[cutoffs[0::6], x])

#         fit_futures_for_x = []
#         for i, mask_group in enumerate(masks_split):
#             # Prepare parameters for this specific (x, mask_group) pair
#             p0, lbounds, hbounds = [], [], []
#             p0.extend(continuum)
#             l, h = generate_bounds(continuum[0], .3); lbounds.append(l); hbounds.append(h)
#             l, h = generate_bounds(continuum[1], .3); lbounds.append(l); hbounds.append(h)
#             lbounds.append(-np.inf); hbounds.append(np.inf)

#             for mask_idx in mask_group:
#                 if mask_idx not in mask.bad_mask:
#                     p0.append(abs(flat_data[int(np.round(np.poly1d(mask_polys0[mask_idx])(x))), x] - f_2(int(np.round(np.poly1d(mask_polys0[mask_idx])(x))), *continuum)))
#                     lbounds.append(0)
#                     hbounds.append(np.max(flat_data[cutoffs[i]:cutoffs[i+1], x]) * 2)
#                     p0.append(np.poly1d(mask_polys0[mask_idx])(x))
#                     lbounds.append(np.poly1d(mask_polys0[mask_idx])(x) - 2)
#                     hbounds.append(np.poly1d(mask_polys0[mask_idx])(x) + 2)
#                     p0.append(3.); lbounds.append(1.5); hbounds.append(5.5)
            
#             xs = np.arange(cutoffs[i], cutoffs[i+1])
#             ys = flat_data[cutoffs[i]:cutoffs[i+1], x]

#             # Submit the fine-grained task
#             future = minimize_gauss_fit(xs, ys, p0, lbounds, hbounds)
#             fit_futures_for_x.append(future)
        
#         all_futures[x] = fit_futures_for_x
#         print(f"Submitted {len(fit_futures_for_x)} mask group fits for x={x}", flush=True)

#     # 4. Return all futures and context to the main script for processing
#     return {
#         "all_futures": all_futures,
#         "lines": lines,
#         "masks_split": masks_split,
#         "bad_mask": mask.bad_mask,
#         "datadir": mask.datadir,
#         "total_masks": mask.total_masks
#     }



@python_app
def create_flatmask_app(dep_futures,mask_args,center_deg,sigma_deg,sig_mult):
    mask = Mask(**mask_args)
    # [f.result() for f in dep_futures]
    mask.get_flat_traces(center_deg,sigma_deg)
    mask.create_mask(sig_mult)
    return None

@python_app
def optimize_arc_app(dep_futures,mask_args,arcfilename,sig_mult,expected_peaks,optimize=True):
    mask = Mask(**mask_args)
    [f.result() for f in dep_futures]
    return mask.optimize_trace(arcfilename,sig_mult,False,expected_peaks=expected_peaks,optimize=optimize)

@python_app
def optimize_data_app(dep_futures,mask_args,arcfilename,datafilename,sig_mult,expected_peaks,optimize):
    mask = Mask(**mask_args)
    [f.result() for f in dep_futures]
    mask.optimize_trace(datafilename,sig_mult,True,expected_peaks=expected_peaks,optimize=optimize)
    mask.get_rots(arcfilename,datafilename,optimize=optimize)
    return None

@python_app
def create_mask_app(dep_futures,mask_args,filename,arcfilename,sig_mult,copy=False):
    mask = Mask(**mask_args)
    [f.result() for f in dep_futures]
    if copy:
        mask.create_mask(sig_mult,filename,copy=arcfilename)
    else:
        return mask.create_mask(sig_mult,filename)