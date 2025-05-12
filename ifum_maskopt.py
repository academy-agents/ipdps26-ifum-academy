import os
import numpy as np
from astropy.io import fits
import scipy
import math
import matplotlib.pyplot as plt
import ifum_utils

class Mask():
    '''
    Class to ...

    Attributes:
        

    Methods:
        
    '''
    def __init__(self, color: str, flatfilename: str, bad_masks, 
                 total_masks: int, mask_groups: int):
        self.color = color
        self.flatfilename = flatfilename
        self.flatdir = os.path.join(os.path.relpath("out"),self.flatfilename+"_withbias_"+self.color+".fits")
        self.maskdir = os.path.join(os.path.relpath("out"),self.flatfilename+self.color+"_mask.fits")
        self.datadir = os.path.join(os.path.relpath("out"),self.flatfilename+self.color+"_trace_fits.npz")
        self.bad_mask = bad_masks[0] if color=="b" else bad_masks[1]
        self.total_masks = total_masks
        self.mask_groups = mask_groups

    def first_guess(self, deg, median_filter = (1,9)) -> np.ndarray:
        flat_data = fits.open(self.flatdir)[0].data
        flat_data = scipy.ndimage.median_filter(flat_data,size=median_filter)

        expected_peaks = int(self.total_masks//2-len(self.bad_mask))

        x = np.arange(0,flat_data.shape[1],3)
        mask_peaks = np.empty(shape=(len(x),expected_peaks))
        mask_polys_0 = np.empty(shape=(expected_peaks,deg))
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
            
            # take top (# of expected peaks) to include
            num_det_peaks.append(len(peaks))
            if len(peaks) >= expected_peaks:
                top_args = flat_flat[peaks].argsort()[-expected_peaks:][::-1]
                peaks = peaks[top_args]
                mask_peaks[i,:] = np.sort(peaks)
            else:
                mask_peaks[i,:] = np.nan
        # print(np.mean(num_det_peaks))
        # print(np.median(num_det_peaks))
        # plt.hist(num_det_peaks)
        # plt.show()

        # "repair" the mask dots to better match neighbors
        # # # plt.figure(figsize=(100,10))
        # old_mask_peaks = mask_peaks.copy()
        # cutoff = 3 # adjacent pixel diff (this is high)
        # for i,mask_dots in enumerate(mask_peaks.T):
        #     initmask = ~np.isnan(mask_dots)
        #     # plt.scatter(np.arange(0,flat_data.shape[1],3)[initmask],mask_dots[initmask],alpha=0.3)
        #     # diff = np.concatenate(([0],np.diff(mask_dots)))
        #     # bad_diff = np.abs(diff)>cutoff
        #     # print(len(mask_dots),np.sum(bad_diff))
            
        #     # if not i==0:
        #     #     diff = np.concatenate(([0],np.diff((mask_peaks.T)[i])))
        #     #     bad_diff = np.abs(diff)>cutoff
        #     #     mask_dots_ = mask_dots.copy()
        #     #     mask_dots_[bad_diff] = (mask_peaks.T)[i-1][bad_diff]
        #     #     lower_diff = np.concatenate(([0],np.diff(mask_dots_)))
        #     #     lower_good = np.abs(lower_diff)<=cutoff
        #     #     (mask_peaks.T)[i][(bad_diff)&(~lower_good)] = np.nan
        #     #     for col in np.where(lower_good&bad_diff)[0]:
        #     #         (mask_peaks.T)[i:,col] = np.roll((mask_peaks.T)[i:,col], shift=1)

        #     if i!=(len(mask_peaks.T)-1):
        #         diff = np.concatenate(([0],np.diff((mask_peaks.T)[i])))
        #         bad_diff = np.abs(diff)>cutoff
        #         mask_dots_ = mask_dots.copy()
        #         mask_dots_[bad_diff] = (mask_peaks.T)[i+1][bad_diff]
        #         upper_diff = np.concatenate(([0],np.diff(mask_dots_)))
        #         upper_good = np.abs(upper_diff)<=cutoff
        #         (mask_peaks.T)[i][(bad_diff)&(~upper_good)] = np.nan
        #         # print(np.sum(upper_good&bad_diff))
        #         # plt.figure(figsize=(50,5))
        #         # plt.scatter(np.arange(0,flat_data.shape[1],3),(mask_peaks.T)[i],marker="x",color="red")
        #         # mask_peaks[(upper_good&bad_diff),(i+1):] = mask_peaks[(upper_good&bad_diff),:-(i+1)]
        #         for col in np.where(upper_good&bad_diff)[0]:
        #             # print(col)
        #             (mask_peaks.T)[i:,col] = np.roll((mask_peaks.T)[i:,col], shift=-1)
        #         # plt.scatter(np.arange(0,flat_data.shape[1],3),(mask_peaks.T)[i],color="blue",alpha=0.5)
        #         # plt.show()
        #         # print(upper_diff[bad_diff])
        # plt.show()

        # for i,mask_dots in enumerate(mask_peaks.T):        
        #     if i!=0:
        #         diff = np.concatenate(([0],np.diff((mask_peaks.T)[i])))
        #         bad_diff = np.abs(diff)>cutoff
        #         mask_dots_ = mask_dots.copy()
        #         mask_dots_[bad_diff] = (mask_peaks.T)[i-1][bad_diff]
        #         lower_diff = np.concatenate(([0], np.diff(mask_dots_)))
        #         lower_good = np.abs(lower_diff)<=cutoff
        #         (mask_peaks.T)[i][(bad_diff)&(~lower_good)] = np.nan
        #         for col in np.where(lower_good&bad_diff)[0]:
        #             (mask_peaks.T)[i:,col] = np.roll((mask_peaks.T)[i:,col], shift=1)

        # attempt repair????
        max_dist = 3 # adjacent pixel diff (this is high)
        # for i,mask_dots in enumerate(mask_peaks.T):
        #     if i!=(len(mask_peaks.T)-1):
        #         diff = np.concatenate(([0],np.diff((mask_peaks.T)[i])))
        #         bad_diff = np.abs(diff)>max_dist
        #         mask_dots_ = mask_dots.copy()
        #         mask_dots_[bad_diff] = (mask_peaks.T)[i+1][bad_diff]
        #         upper_diff = np.concatenate(([0],np.diff(mask_dots_)))
        #         upper_good = np.abs(upper_diff)<=max_dist
        #         (mask_peaks.T)[i][(bad_diff)&(~upper_good)] = np.nan
        #         for col in np.where(upper_good&bad_diff)[0]:
        #             # print(col)
        #             (mask_peaks.T)[i:,col] = np.roll((mask_peaks.T)[i:,col], shift=-1)
        # for i,mask_dots in enumerate(mask_peaks.T):
        #     diff = np.concatenate(([0],np.diff((mask_peaks.T)[i])))
        #     bad_diff = np.abs(diff)>max_dist
        #     (mask_peaks.T)[i][bad_diff] = np.nan
            
        for i in range(1,len(mask_peaks.T)-1):
            current_group = mask_peaks.T[i]
            prev_group = mask_peaks.T[i-1]
            next_group = mask_peaks.T[i+1]
            
            for j,peak in enumerate(current_group):
                if np.isnan(peak):
                    continue
                
                dist_to_prev = np.abs(prev_group-peak)
                dist_to_next = np.abs(next_group-peak)
                
                # peak closer to a peak in the previous group
                if np.any(dist_to_prev<=max_dist) and np.min(dist_to_prev)==np.min(dist_to_prev[dist_to_prev<=max_dist]):
                    closest_peak_idx = np.argmin(dist_to_prev)
                    mask_peaks.T[i][j] = prev_group[closest_peak_idx]

                # peak closer to a peak in the next group
                elif np.any(dist_to_next<=max_dist) and np.min(dist_to_next)==np.min(dist_to_next[dist_to_next<=max_dist]):
                    closest_peak_idx = np.argmin(dist_to_next)
                    mask_peaks.T[i][j] = next_group[closest_peak_idx]
        # for i,mask_dots in enumerate(mask_peaks.T):
        #     diff = np.concatenate(([0],np.diff((mask_peaks.T)[i])))
        #     bad_diff = np.abs(diff)>max_dist
        #     (mask_peaks.T)[i][bad_diff] = np.nan


        
        for i,mask_dots in enumerate(mask_peaks.T):
            initmask = ~np.isnan(mask_dots)
            # initmask &= ((np.concatenate(([0],np.diff(mask_dots)))<1.*np.nanstd(np.diff(mask_dots)))&
            #              (np.concatenate(([0],np.diff(mask_dots)))>-1.*np.nanstd(np.diff(mask_dots))))
            x = np.arange(0,flat_data.shape[1],3)[initmask]
            mask_dots = mask_dots[initmask]

            rand_fits = []
            rand_masks = []
            rand_polymasks = []
            rand_stds = []
            for j in range(50):
                rand = np.array([True]*round(0.75*len(mask_dots))+[False]*round(0.25*len(mask_dots)))
                np.random.shuffle(rand)
                polymask, cont_fit = ifum_utils.sigma_clip(x[rand],mask_dots[rand],deg=deg,weight=np.ones_like(x[rand]),sigma=1,iter=5,include=0.5)
                rand_fits.append(cont_fit)
                rand_masks.append(rand)
                rand_polymasks.append(polymask)
                rand_stds.append(np.std(mask_dots[rand][polymask]-np.poly1d(cont_fit)(x[rand][polymask])))
            cont_fit = rand_fits[np.argmin(rand_stds)]
            randmask = rand_masks[np.argmin(rand_stds)]
            polymask = rand_polymasks[np.argmin(rand_stds)]

            # cont_fit = np.polyfit(x,mask_dots,deg)
            # # print(i,np.std(mask_dots-np.poly1d(cont_fit)(x)))
            # try:
            #     for j in range(20):
            #         polymask = ((mask_dots<np.poly1d(cont_fit)(x)+0.5*np.std(np.poly1d(cont_fit)(x)))&
            #                     (mask_dots>np.poly1d(cont_fit)(x)-0.5*np.std(np.poly1d(cont_fit)(x))))
            #         cont_fit = np.polyfit(x[polymask],mask_dots[polymask],deg)
            # except:
            #     # plt.title(i)
            #     # plt.scatter(x,mask_dots,alpha=0.25)
            #     # plt.scatter(x[polymask],mask_dots[polymask],alpha=0.25)
            #     # plt.plot(x,np.poly1d(cont_fit)(x),label=i,alpha=0.5,lw=0.75)
            #     # plt.show()
            #     cont_fit = cont_fit

            if np.std(np.poly1d(cont_fit)(x[randmask][polymask])-mask_dots[randmask][polymask])>3:
                plt.title(i)
                plt.scatter(x,mask_dots,alpha=0.25)
                plt.scatter(x[randmask][polymask],mask_dots[randmask][polymask],alpha=0.25)
                plt.plot(x,np.poly1d(cont_fit)(x),label=i,alpha=0.5,lw=0.75)
                plt.show(block=False)
                plt.pause(1)
                plt.close()

            # if np.std(mask_dots-np.poly1d(cont_fit)(x))>1:
            #     plt.title(i)
            #     plt.scatter(x,(old_mask_peaks.T)[i][initmask],alpha=0.25,color="red",marker="x")
            # plt.scatter(x,mask_dots,alpha=0.05)
            #     plt.scatter(x[polymask],mask_dots[polymask],alpha=0.25)
            # plt.plot(x,np.poly1d(cont_fit)(x),label=i,alpha=0.5,lw=0.75)
            # plt.text(x[0],np.poly1d(cont_fit)(x)[0],str(i+1),va="center",ha="center",color="orange")
            # if i%10==0:
            #     plt.legend()
            #     plt.show()
            #     plt.show()
            mask_polys[i] = cont_fit
        # plt.legend()

        # 2nd run!
        # for i,mask_dots in enumerate(mask_peaks.T):
        #     initmask = ~np.isnan(mask_dots)
        #     x = np.arange(0,flat_data.shape[1],3)
        #     initmask &= (np.abs(mask_dots-np.poly1d(mask_polys_0[i])(x))<10)
        #     x = x[initmask]
        #     mask_dots = mask_dots[initmask]

        #     polymask, cont_fit = ifum_utils.sigma_clip(x,mask_dots,deg,weight=np.ones_like(x),sigma=1,iter=20,include=0.5)

        #     # cont_fit = np.polyfit(x,mask_dots,deg)
        #     # # print(i,np.std(mask_dots-np.poly1d(cont_fit)(x)))
        #     # try:
        #     #     for j in range(20):
        #     #         polymask = ((mask_dots<np.poly1d(cont_fit)(x)+0.5*np.std(np.poly1d(cont_fit)(x)))&
        #     #                     (mask_dots>np.poly1d(cont_fit)(x)-0.5*np.std(np.poly1d(cont_fit)(x))))
        #     #         cont_fit = np.polyfit(x[polymask],mask_dots[polymask],deg)
        #     # except:
        #     #     # plt.title(i)
        #     #     # plt.scatter(x,mask_dots,alpha=0.25)
        #     #     # plt.scatter(x[polymask],mask_dots[polymask],alpha=0.25)
        #     #     # plt.plot(x,np.poly1d(cont_fit)(x),label=i,alpha=0.5,lw=0.75)
        #     #     # plt.show()
        #     #     cont_fit = cont_fit

        #     if np.std(np.poly1d(cont_fit)(x)-mask_dots)>1:
        #         plt.title(i)
        #         plt.scatter(x,mask_dots,alpha=0.25)
        #         plt.scatter(x[polymask],mask_dots[polymask],alpha=0.25)
        #         plt.plot(x,np.poly1d(cont_fit)(x),label=i,alpha=0.5,lw=0.75)
        #         plt.show(block=False)
        #         plt.pause(0.5)
        #         plt.close()

        #     # if np.std(mask_dots-np.poly1d(cont_fit)(x))>1:
        #     #     plt.title(i)
        #     #     plt.scatter(x,(old_mask_peaks.T)[i][initmask],alpha=0.25,color="red",marker="x")
        #     # plt.scatter(x,mask_dots,alpha=0.05)
        #     #     plt.scatter(x[polymask],mask_dots[polymask],alpha=0.25)
        #     # plt.plot(x,np.poly1d(cont_fit)(x),label=i,alpha=0.5,lw=0.75)
        #     # plt.text(x[0],np.poly1d(cont_fit)(x)[0],str(i+1),va="center",ha="center",color="orange")
        #     # if i%10==0:
        #     #     plt.legend()
        #     #     plt.show()
        #     #     plt.show()
        #     mask_polys[i] = cont_fit


        # plt.show()
        for mask in self.bad_mask:
            mask_polys = np.insert(mask_polys, mask, np.nan, axis=0)

        plt.figure(figsize=(100,100))
        plt.imshow(flat_data,origin="lower",cmap="Greys_r")
        for mask in range(self.total_masks//2):
            cont_fit = mask_polys[mask]
            x = np.arange(flat_data.shape[1])
            plt.plot(x,np.poly1d(cont_fit)(x),alpha=0.75,lw=0.75)
            plt.text(x[0],np.poly1d(cont_fit)(x)[0],str(mask+1),va="center",ha="center",color="orange")
        plt.axis("off")
        print("saving...")
        plt.savefig("out.png",dpi=100,bbox_inches='tight')
        plt.close()
        print("saved")

        return mask_polys

    def f_2(self,x,a,b,c):
        return a*x**2+b*x+c
    
    def gauss_added(self,x,*params):
        y = np.zeros_like(x)
        for i in np.arange(3,len(params),3):
            y += params[i]*np.exp(-(x-params[i+1])**2/(2*params[i+2]**2))
        y = y+params[0]*x**2+params[1]*x+params[2]
        return y

    def generate_bounds(self,value,percentile):
        if value>0:
            return value*(1-percentile),value*(1+percentile)
        else:
            return value*(1+percentile),value*(1-percentile)

    def mask_poly(self, mask_polys, n) -> None:
        flat_data = fits.open(self.flatdir)[0].data
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
        bad_lines = []
        for x in x_s:            
            cutoffs = [5]
            for i in np.arange(1,len(masks_split)):
                first = mask_polys[masks_split[i-1]][~np.isnan(mask_polys[masks_split[i-1]]).any(axis=1)][-1]
                last = mask_polys[masks_split[i]][~np.isnan(mask_polys[masks_split[i]]).any(axis=1)][0]
                cutoffs.append(int((np.poly1d(first)(x)+np.poly1d(last)(x))/2))
            cutoffs.append(flat_data.shape[0]-5)
            cutoffs = np.array(cutoffs)
            
            continuum,_ = scipy.optimize.curve_fit(self.f_2,cutoffs[0::6],flat_data[cutoffs[0::6],x])
            
            try:
                centers = []
                sigmas = []
                amps = []
                for i,mask_group in enumerate(masks_split):
                    p0 = []
                    lbounds = []
                    hbounds = []
                    
                    # append quadratic shift guess
                    p0.append(continuum[0])
                    l,h = self.generate_bounds(continuum[0],.3)
                    lbounds.append(l)
                    hbounds.append(h)
                    p0.append(continuum[1])
                    l,h = self.generate_bounds(continuum[1],.3)
                    lbounds.append(l)
                    hbounds.append(h)
                    p0.append(continuum[2])
                    lbounds.append(-np.inf)
                    hbounds.append(np.inf)

                    for mask in mask_group:
                        if mask not in self.bad_mask:
                            # append amplitude guess
                            p0.append(abs(flat_data[int(np.round(np.poly1d(mask_polys[mask])(x))),x]-self.f_2(int(np.round(np.poly1d(mask_polys[mask])(x))),*continuum)))
                            lbounds.append(0)
                            hbounds.append(np.max(flat_data[cutoffs[i]:cutoffs[i+1],x])*2)
                            # append center guess
                            p0.append(np.poly1d(mask_polys[mask])(x))
                            lbounds.append(np.poly1d(mask_polys[mask])(x)-4)
                            hbounds.append(np.poly1d(mask_polys[mask])(x)+4)
                            # append sigma guess
                            p0.append(3.)
                            lbounds.append(1.5)
                            hbounds.append(5.5)
                        
                    popt,_ = scipy.optimize.curve_fit(self.gauss_added,np.arange(cutoffs[i],cutoffs[i+1]),flat_data[cutoffs[i]:cutoffs[i+1],x],p0=p0,bounds=(lbounds,hbounds),nan_policy="omit",method="trf")

                    center = np.array(popt[3:][1::3])
                    sigma = np.array(popt[3:][2::3])
                    amp = np.array(popt[3:][0::3])

                    if np.intersect1d(self.bad_mask,mask_group).size > 0:
                        _, _, ind2 = np.intersect1d(self.bad_mask,mask_group,return_indices=True)
                        for mask_ in ind2:
                            center = np.insert(center, mask_, np.nan, axis=0)
                            sigma = np.insert(sigma, mask_, np.nan, axis=0)
                            amp = np.insert(amp, mask_, np.nan, axis=0)

                    centers.append(center)
                    sigmas.append(sigma)
                    amps.append(amp)

                gauss_centers_full = np.vstack((gauss_centers_full,np.array(centers).flatten()))
                gauss_sigmas_full = np.vstack((gauss_sigmas_full,np.array(sigmas).flatten()))
                gauss_amps_full = np.vstack((gauss_amps_full,np.array(amps).flatten()))
            except:
                bad_lines.append(x)
                print("bad fit at x=",x)

        save_dict = {'x': x_s[np.isin(x_s, bad_lines, invert=True)],
                     'centers': gauss_centers_full,
                     'sigmas': gauss_sigmas_full, 
                     'amps': gauss_amps_full}

        np.savez(self.datadir, **save_dict)

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
            file = os.path.join(os.path.relpath("out"),mode+self.color+"_trace_fits.npz")
            npzfile = np.load(file)
            save_dir = os.path.join(os.path.relpath("out"),mode+self.color+"_mask.fits")
            init_traces = npzfile["init_traces"]
            traces_sigma = npzfile["init_traces_sigma"]
        traces = npzfile["traces"]
        flat_data = fits.open(self.flatdir)[0].data

        if mode!="flat" and np.array_equiv(traces,init_traces): # if traces are the same, do not waste time creating mask. instead use arc's
            fits.writeto(save_dir, data=fits.open(self.maskdir)[0].data, overwrite=True)
        elif copy is not None and np.array_equiv(traces,init_traces):
            arcmask = os.path.join(os.path.relpath("out"),copy+self.color+"_mask.fits")
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
        arcdir = os.path.join(os.path.relpath("out"),filename+self.color+".fits")
        data = fits.open(arcdir)[0].data

        if cmrays:
            cmraydir = os.path.join(os.path.relpath("out"),filename+self.color+"_cmray_mask.fits")
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
            ref_a = ifum_utils.get_spectrum_simple(data,mask_data,1,cmray_data)
            for i,m in enumerate(masks_l):
                a = ifum_utils.get_spectrum_simple(data,mask_data,m,cmray_data)
                intensities[i] = a
                lags[i] = ifum_utils.get_lag(a,ref_a)
        else:
            ref_a = ifum_utils.get_spectrum_simple(data,mask_data,1)
            for i,m in enumerate(masks_l):
                a = ifum_utils.get_spectrum_simple(data,mask_data,m)
                intensities[i] = a
                lags[i] = ifum_utils.get_lag(a,ref_a)

        norm_intensities = np.empty(intensities.shape)
        for i,intensity in enumerate(intensities):
            norm_intensities[i] = np.interp(x, x-lags[i], intensity)
        ref_intensity = np.nanmedian(norm_intensities,axis=0)
    
        # gets peak areas for the reference intensity given a certain percentile cutoff
        # perc_cut = 95
        # masked = x[ref_intensity>np.nanpercentile(ref_intensity,perc_cut)]
        # peak_areas = np.split(masked,np.where(np.diff(masked) != x[1]-x[0])[0]+1)
    
        # peak_xs = np.empty((len(peak_areas),len(intensities)))
        # for i,peak_area in enumerate(peak_areas):
        #     for j,a in enumerate(intensities):
        #         peak_xs[i][j] = ifum_utils.get_peak_center(peak_area,4,4,x,a,lags[j])

        peaks,_ = scipy.signal.find_peaks(ref_intensity,
                                          distance=np.nanmean(traces_sigma)*sig_mult*8)
        if len(peaks) > expected_peaks:
            top_args = ref_intensity[peaks].argsort()[-expected_peaks:][::-1]
            peaks = np.sort(peaks[top_args])
        peak_xs = np.empty((len(peaks),len(intensities)))
        for i,peak in enumerate(peaks):
            for j,a in enumerate(intensities):
                peak_xs[i][j] = peak + lags[j]

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
                        popt,_ = scipy.optimize.curve_fit(ifum_utils.gauss,x[mask_area],intensities[i][mask_area],p0=p0)                            
                    else:
                        mask_area = x>((~np.isnan(intensities[i])).cumsum(0).argmax(0)-offset*2)
                        mask_area = mask_area&(~np.isnan(intensities[i]))
                        p0 = [0,np.max(intensities[i][mask_area]),np.argmax(intensities[i][mask_area])+np.min(x[mask_area]),offset/3]
                        popt,_ = scipy.optimize.curve_fit(ifum_utils.gauss,x[mask_area],intensities[i][mask_area],p0=p0)
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
                        
                        result = scipy.optimize.minimize(ifum_utils.minimize_gauss_2d, var, args=(args))
                        results_arr[i,line] = result.x
                        fit_xs[i,line] = x0+result.x[0]
                        fit_ys[i,line] = y0+result.x[1]
                        rotations[i,line] = 0.5*np.arctan(2*result.x[4]/(result.x[3]-result.x[5]))
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
                result = scipy.optimize.minimize(ifum_utils.minimize_poly_dist, var, args=(args))
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

        np.savez(os.path.join(os.path.relpath("out"),filename+self.color+"_trace_fits.npz"), **save_dict)
        


    def get_rots(self,arcfilename,datafilename,optimize=True) -> None:
        arc_npz = np.load(os.path.join(os.path.relpath("out"),arcfilename+self.color+"_trace_fits.npz"))
        data_npz = np.load(os.path.join(os.path.relpath("out"),datafilename+self.color+"_trace_fits.npz"))
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
        np.savez(os.path.join(os.path.relpath("out"),datafilename+self.color+"_trace_fits.npz"), **save_dict)

    def _viz(self,datafilename,sig_mult,masks) -> None:
        data_dir = os.path.join(os.path.relpath("out"),datafilename+self.color+".fits")
        data = fits.open(data_dir)[0].data

        data_npz = np.load(os.path.join(os.path.relpath("out"),datafilename+self.color+"_trace_fits.npz"))
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