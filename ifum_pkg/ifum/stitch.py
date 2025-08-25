from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
import glob
import os
from astropy.io import fits
from scipy import ndimage
from astropy.convolution import Gaussian2DKernel, convolve
import re
from parsl.app.app import python_app, join_app
from parsl.data_provider.files import File
from .utils import *

from academy.agent import Agent, action, loop
from typing import List, Dict
import asyncio

class Stitch():

    def __init__(self, directory: str, filename: str, files, color: str,
                 datafilename: str, arcfilename: str, flatfilename: str):
        self.directory = directory
        self.filename = filename
        self.files = files
        self.color = color
        self.datafilename = datafilename
        self.arcfilename = arcfilename
        self.flatfilename = flatfilename
        
    def load_and_save(self,bin_to_2x1=True) -> None:
        files = load_files(self.directory,self.filename)
        save_file(files,self.filename,bin_to_2x1)
        return self.filename

    def bias_sub(self) -> np.ndarray:
        import os
        from astropy.io import fits
        from scipy import ndimage
        import numpy as np

        # data_file = os.path.join(os.path.abspath("out"),self.datafilename+"_withbias_"+self.color+".fits")
        # arc_file = os.path.join(os.path.abspath("out"),self.arcfilename+"_withbias_"+self.color+".fits")
        flat_file = os.path.join(os.path.abspath("out"),self.flatfilename+"_withbias_"+self.color+".fits")
        
        # with fits.open(data_file) as dataf, \
        #      fits.open(arc_file) as arc_dataf, \
        with fits.open(flat_file) as flat_dataf:
            
            # data = dataf[0].data
            # arc_data = arc_dataf[0].data
            flat_data = flat_dataf[0].data

        # THINK is there a better way to compute internal noise?
        median_image = ndimage.median_filter(flat_data,size=(1,9))
        internal_noise = flat_data/median_image
        internal_noise = (internal_noise)/(np.percentile(internal_noise,99)-np.min(internal_noise))

        fits.writeto(filename=os.path.join(os.path.abspath("out"),self.flatfilename+"_biasfilter_"+self.color+".fits"), data=internal_noise, overwrite=True)

        return internal_noise

    def write_noise(self, internal_noise, files) -> None:
        import os
        from astropy.io import fits

        for file in files:
            # data_file = os.path.join(os.path.abspath("out"),self.filename+"_withbias_"+self.color+".fits")
            data_file = os.path.join(os.path.abspath("out"),file+"_withbias_"+self.color+".fits")

            with fits.open(data_file) as datahdu:
                data = datahdu[0].data

            denoised = data/internal_noise
            # fits.writeto(filename=os.path.join(os.path.abspath("out"),self.filename+self.color+".fits"), data=denoised, overwrite=True)
            fits.writeto(filename=os.path.join(os.path.abspath("out"),file+self.color+".fits"), data=denoised, overwrite=True)

        return None

    # bl_resize is copied from medium website
    # try https://scikit-image.org/docs/stable/auto_examples/transform/plot_rescale.html instead???
    # replace with the scipy interpolation instead???
    def bl_resize(self, image, height, width):
        """
        `image` is a 2-D numpy array
        `height` and `width` are the desired spatial dimension of the new 2-D array.
        """
        img_height, img_width = image.shape
        
        image = image.ravel()
        
        x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
        y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0
        
        y, x = np.divmod(np.arange(height * width), width)
        
        x_l = np.floor(x_ratio * x).astype('int32')
        y_l = np.floor(y_ratio * y).astype('int32')
        
        x_h = np.ceil(x_ratio * x).astype('int32')
        y_h = np.ceil(y_ratio * y).astype('int32')
        
        x_weight = (x_ratio * x) - x_l
        y_weight = (y_ratio * y) - y_l
        
        a = image[y_l * img_width + x_l]
        b = image[y_l * img_width + x_h]
        c = image[y_h * img_width + x_l]
        d = image[y_h * img_width + x_h]
        
        resized = a * (1 - x_weight) * (1 - y_weight) + \
                b * x_weight * (1 - y_weight) + \
                c * y_weight * (1 - x_weight) + \
                d * x_weight * y_weight
        
        return resized.reshape(height, width)

    def interp_shift_and_sub(self,v_shift,h_shift,scale,data,data_,cutoff,mult,margin=10):
        subtracted = data[margin:-margin,margin:-margin]-data_[margin-v_shift:-margin-v_shift,margin-h_shift:-margin-h_shift]*scale
        subtracted[(subtracted<-cutoff)|(subtracted>cutoff)] = np.nan
        return subtracted

    def perform_shift(self,v_shift,h_shift,scale,data_):
        if v_shift > 0:
            data_n = np.vstack((np.zeros((abs(v_shift),data_.shape[1])),data_[:-(v_shift)]))
        elif v_shift < 0:
            data_n = np.vstack((data_[-(v_shift):],np.zeros((abs(v_shift),data_.shape[1]))))
        else:
            data_n = data_

        if h_shift > 0:
            data_n = np.hstack((np.zeros((abs(h_shift),data_n.shape[0])).T,data_n[:,:-(h_shift)]))
        elif h_shift < 0:
            data_n = np.hstack((data_n[:,-(h_shift):],np.zeros((abs(h_shift),data_n.shape[0])).T))

        return data_n*scale

    def cmray_mask(self, data_files) -> None:
        import os
        from astropy.io import fits
        import numpy as np
        from scipy.ndimage import convolve
        from astropy.convolution import Gaussian2DKernel
        import glob

        # area is size of returned array
        # cutoff ignores pixels >+/<- that value
        area,cutoff,mult = 150,500,2
        if len(data_files)>4:
            use = len(data_files)//2
        else:
            use = len(data_files)-1
        sim_files = data_files.copy()

        sim_files.remove(self.datafilename)
        v_range = np.arange(-2,3,1)
        h_range = np.arange(-2,3,1)
        s_range = np.linspace(0.5,1.5,100)

        with fits.open(os.path.join(os.path.abspath("out"),self.datafilename+self.color+".fits")) as refdata:
            data = refdata[0].data
        
        data_m = np.ones(data.shape)
        fit_params = np.zeros((len(sim_files),4))
        not_sim = []
        for idx,s_file in enumerate(sim_files):
            if len(glob.glob(os.path.join(os.path.abspath("out"),(s_file+"*.fits"))))<2:
                not_sim.append(idx)
            else:
                with fits.open(os.path.join(os.path.abspath("out"),s_file+self.color+".fits")) as sfile:
                    data_ = sfile[0].data
                
                    i1,i2 = int(len(data)/area)+area,int(len(data)/area)+2*area
                    margin = 10
                    i1_,i2_ = i1-margin,i2+margin
                    cut_data = data[i1_:i2_,i1_:i2_]
                    cut_data_ = data_[i1_:i2_,i1_:i2_]
                    ncut_data = self.bl_resize(cut_data,cut_data.shape[0]*mult,cut_data.shape[1]*mult)
                    ncut_data_ = self.bl_resize(cut_data_,cut_data_.shape[0]*mult,cut_data_.shape[1]*mult)
                    
                    mesh = np.empty((v_range.size,h_range.size,s_range.size))
                    for i,v in enumerate(v_range):
                        for j,h in enumerate(h_range):
                            for k,s in enumerate(s_range):
                                mesh[i,j,k] = np.nanstd(self.interp_shift_and_sub(v,h,s,ncut_data,ncut_data_,cutoff,2))
                    best_v = v_range[np.unravel_index(np.argmin(mesh),mesh.shape)[0]]
                    best_h = h_range[np.unravel_index(np.argmin(mesh),mesh.shape)[1]]
                    best_s = s_range[np.unravel_index(np.argmin(mesh),mesh.shape)[2]]
                    best_std = np.nanstd(self.interp_shift_and_sub(best_v,best_h,best_s,ncut_data,ncut_data_,cutoff,2))
                    fit_params[idx] = np.array([best_v,best_h,best_s,best_std])

        sim_files = [k for l, k in enumerate(sim_files) if l not in not_sim]
        fit_params = np.delete(fit_params, (not_sim), axis=0)
        added_data = np.zeros(data.shape)
        best_ = np.array(np.argsort(fit_params[:,3])[:use].astype(int))
        for idx in best_:
            with fits.open(os.path.join(os.path.abspath("out"),sim_files[idx]+self.color+".fits")) as sfile:
                data_ = sfile[0].data
                data_large = self.bl_resize(data_,data_.shape[0]*mult,data_.shape[1]*mult)
                shifted_ = self.perform_shift(int(fit_params[idx,0]),int(fit_params[idx,1]),fit_params[idx,2],data_large)
                shifted_ = self.bl_resize(shifted_,data_.shape[0],data_.shape[1])
                added_data += data-shifted_

        gauss_kernal = Gaussian2DKernel(x_stddev=1,y_stddev=1)
        cmray_conv = convolve(added_data,gauss_kernal)
        cmray_conv_mask = (cmray_conv>(np.median(cmray_conv.flatten())+3*np.std(cmray_conv.flatten())))
        fits.writeto(os.path.join(os.path.abspath("out"),self.datafilename+self.color+"_cmray_mask.fits"), data=1.*cmray_conv_mask, overwrite=True)



    # EVERYTHING BELOW IS FOR AGENT!!!

    def shift_and_scale(self, image, v_shift, h_shift, scale=1.0, order=1):
        # order = 1 -> bilinear interpolation

        shifted_image = ndimage.shift(
            image,
            (v_shift,h_shift),
            order=order,
            mode='constant',
            cval=0.0
        )
        
        return shifted_image * scale

    def subtract_ims(self, v_shift, h_shift, scale, refdata, data, margin, cutoff):
        shifted_data = self.shift_and_scale(data, v_shift, h_shift, scale)
        
        subtracted = refdata[margin:-margin, margin:-margin] - shifted_data[margin:-margin, margin:-margin]
        # turn outliers to NaN (top&bottom cutoff_perc%)
        extreme_mask = (subtracted < (-1*cutoff)) | (subtracted > cutoff)
        subtracted[extreme_mask] = np.nan

        return subtracted

    def cmray_mask_new(self, data_files, area=150) -> None:
        import os
        from astropy.io import fits
        import numpy as np
        from scipy.ndimage import convolve
        from astropy.convolution import Gaussian2DKernel
        import glob

        sim_files = data_files.copy()
        sim_files.remove(self.datafilename)
        print(self.datafilename,sim_files)

        v_range = np.linspace(-2,2,20)
        h_range = np.linspace(-2,2,20)
        s_range = np.linspace(0.5,1.5,100)

        margin = int(np.max([abs(v_range),abs(h_range)])+1)


        with fits.open(os.path.join(os.path.abspath("out"),self.datafilename+self.color+".fits")) as refdata:
            data = refdata[0].data

            random_x = np.random.randint(0,data.shape[1]-area-2*margin)
            random_y = np.random.randint(0,data.shape[0]-area-2*margin)

            cut_data = data[
                random_y+margin:random_y+margin+area,
                random_x+margin:random_x+margin+area
            ]

        valid_sim_files = []
        valid_fit_params = []

        for idx,s_file in enumerate(sim_files):
            if len(glob.glob(os.path.join(os.path.abspath("out"), (s_file+"*.fits")))) < 2:
                continue

            with fits.open(os.path.join(os.path.abspath("out"),s_file+self.color+".fits")) as sfile:
                data_ = sfile[0].data

                cut_data_ = data_[
                    random_y+margin:random_y+margin+area,
                    random_x+margin:random_x+margin+area
                ]

                cutoff = 2*np.nanstd(cut_data)

                mesh = np.empty((v_range.size,h_range.size,s_range.size))
                for i,v in enumerate(v_range):
                    for j,h in enumerate(h_range):
                        for k,s in enumerate(s_range):
                            result = self.subtract_ims(v,h,s,cut_data,cut_data_,margin,cutoff)
                            mesh[i,j,k] = np.nanstd(result)
                best_indicies = np.unravel_index(np.argmin(mesh),mesh.shape)
                best_v = v_range[best_indicies[0]]
                best_h = h_range[best_indicies[1]]
                best_s = s_range[best_indicies[2]]
                best_std = mesh[best_indicies]
                print(best_v,best_h,best_s)
                print(best_std)

                plt.figure(figsize=(10, 5))
                cmap = plt.cm.get_cmap("gray").copy()
                cmap.set_bad('maroon', 1.)
                result = self.subtract_ims(best_v,best_h,best_s,cut_data,cut_data_,margin,cutoff)
                vmin,vmax = -5*np.nanstd(result), 5*np.nanstd(result)
                plt.subplot(1, 3, 1)
                plt.imshow(cut_data-cut_data_,cmap=cmap,vmin=vmin,vmax=vmax)
                plt.colorbar()
                plt.subplot(1, 3, 2)
                plt.imshow(result,cmap=cmap,vmin=vmin,vmax=vmax)
                plt.colorbar()
                plt.subplot(1, 3, 3)
                # show grid search vs and hs at best_s
                plt.imshow(np.min(mesh,axis=2),
                           extent=[h_range[0], h_range[-1], v_range[0], v_range[-1]],
                           origin='lower', aspect='auto')
                plt.show()

                valid_sim_files.append(s_file)
                valid_fit_params.append([best_v, best_h, best_s, best_std])

        valid_fit_params = np.array(valid_fit_params)

        if len(data_files) > 4:
            use = len(data_files) // 2
        else:
            use = len(data_files) - 1
        use = min(use, len(valid_sim_files))
        best_indices = np.argsort(valid_fit_params[:,3])[:use]
        added_data = np.zeros(data.shape)
        added_data = np.zeros(data.shape)
        for idx in best_indices:
            with fits.open(os.path.join(os.path.abspath("out"),valid_sim_files[idx]+self.color+".fits")) as sfile:
                data_ = sfile[0].data

                shifted_scaled = self.shift_and_scale(
                    data_, 
                    valid_fit_params[idx, 0],
                    valid_fit_params[idx, 1],
                    valid_fit_params[idx, 2]
                )
                added_data += data - shifted_scaled
    
        gauss_kernal = Gaussian2DKernel(x_stddev=1, y_stddev=1)
        cmray_conv = convolve(added_data, gauss_kernal)
        cmray_conv_mask = (cmray_conv > (np.median(cmray_conv.flatten()) + 3*np.std(cmray_conv.flatten())))
        # fits.writeto(os.path.join(os.path.abspath("out"), self.datafilename+self.color+"_cmray_mask.fits"), 
        #             data=1.*cmray_conv_mask, overwrite=True)
        

    def optimize_cmray_params(self, cut_data, cut_data_, margin, cutoff, x0 = np.array([0,0,1]), bounds = [(-2, 2), (-2, 2), (0.5, 1.5)], optimize=True):
        def optimize_function(params):
            v_shift, h_shift, scale = params
            result = self.subtract_ims(v_shift, h_shift, scale, cut_data, cut_data_, margin, cutoff)
            return np.nanstd(result)
        
        if not optimize:
            return x0[0], x0[1], x0[2], optimize_function(x0)
        
        result = minimize(
            optimize_function,
            x0,
            bounds=bounds,
            method='L-BFGS-B',
            options={'ftol': 1e-6}
        )

        best_v, best_h, best_s = result.x
        best_std = result.fun

        return best_v, best_h, best_s, best_std

    def cmray_mask_minimize(self, data_files, area=300, x0=np.array([0, 0, 1]), bounds=[(-2, 2), (-2, 2), (0.5, 1.5)]) -> None:
        import os
        from astropy.io import fits
        import numpy as np
        from scipy.ndimage import convolve
        from astropy.convolution import Gaussian2DKernel
        import glob

        sim_files = data_files.copy()
        sim_files.remove(self.datafilename)
        print(self.datafilename,sim_files)

        margin = int(np.max([np.abs(bounds[0]),np.abs(bounds[1])])+1)


        with fits.open(os.path.join(os.path.abspath("out"),self.datafilename+self.color+".fits")) as refdata:
            data = refdata[0].data

            # print(np.std(data), np.mean(data), np.min(data), np.max(data))
            # std of middle 98% of data
            # print(np.std(np.percentile(data.flatten(), [1, 99])))
            # std of middle 50% of data
            # print(np.std(np.percentile(data.flatten(), [25, 75])))
            # std of 1-50% of data
            # BEST METRIC
            # effectively: gets rid of bottom outliers, assumes 50% of the data is approximately noise
            # -> subtracted images should have less than std of 1-50% of data
            print(np.std(np.percentile(data.flatten(), [1, 50])))

            random_x = np.random.randint(0,data.shape[1]-area-2*margin)
            random_y = np.random.randint(0,data.shape[0]-area-2*margin)

            cut_data = data[
                random_y+margin:random_y+margin+area,
                random_x+margin:random_x+margin+area
            ]

        valid_sim_files = []
        valid_fit_params = []

        # fig,ax = plt.subplots(2,len(sim_files),figsize=(2*len(sim_files),4))

        for idx,s_file in enumerate(sim_files):
            if len(glob.glob(os.path.join(os.path.abspath("out"), (s_file+"*.fits")))) < 2:
                continue

            with fits.open(os.path.join(os.path.abspath("out"),s_file+self.color+".fits")) as sfile:
                data_ = sfile[0].data

                cut_data_ = data_[
                    random_y+margin:random_y+margin+area,
                    random_x+margin:random_x+margin+area
                ]

                cutoff = 2*np.nanstd(cut_data)

                best_v, best_h, best_s, best_std = self.optimize_cmray_params(
                    cut_data,
                    cut_data_,
                    margin,
                    cutoff,
                    x0=np.array([0, 0, 1]),
                    bounds=[(-2, 2), (-2, 2), (0.5, 1.5)]
                )

                valid_sim_files.append(s_file)
                valid_fit_params.append([best_v, best_h, best_s, best_std])

                # cmap = plt.cm.get_cmap("PuOr").copy()
                # cmap.set_bad('red', 1.)
                # result = self.subtract_ims(best_v,best_h,best_s,cut_data,cut_data_,margin,cutoff)
                # vmin,vmax = -5*np.nanstd(result), 5*np.nanstd(result)
                # ax[1, idx].set_title(f"{s_file} std: {best_std:.2f}")
                # ax[0, idx].imshow(cut_data-cut_data_,cmap=cmap,vmin=vmin,vmax=vmax)
                # # get rid of axis ticks
                # ax[0, idx].set_xticks([])
                # ax[0, idx].set_yticks([])
                # # fig.colorbar(ax=ax[0, idx], mappable=ax[0, idx].images[0],fraction=0.046, pad=0.04)
                # ax[1, idx].imshow(result,cmap=cmap,vmin=vmin,vmax=vmax)
                # ax[1, idx].set_xticks([])
                # ax[1, idx].set_yticks([])
                # # fig.colorbar(ax=ax[1, idx], mappable=ax[1, idx].images[0],fraction=0.046, pad=0.04)
        # plt.show()

        valid_fit_params = np.array(valid_fit_params)

        # _,ax = plt.subplots()
        # ax.plot(np.arange(len(valid_fit_params)), valid_fit_params[:, 0], label="v shift", color="sienna")
        # ax.plot(np.arange(len(valid_fit_params)), valid_fit_params[:, 1], label="h shift", color="peru")
        # ax2 = ax.twinx()
        # ax2.plot(np.arange(len(valid_fit_params)), valid_fit_params[:, 2], label="scale", color='purple')
        # ax.set_xlabel("sim #")
        # ax.set_ylabel("shift (px)")
        # ax2.set_ylabel("scale")

        # # get all handles and labels from ax2 and plt
        # handles, labels = ax.get_legend_handles_labels()
        # handles2, labels2 = ax2.get_legend_handles_labels()
        # plt.legend(handles+handles2, labels+labels2)
        # plt.show()

        print(valid_fit_params[:,0])
        print(valid_fit_params[:,1])
        print(valid_fit_params[:,2])
        print(valid_fit_params[:,3])

        if len(data_files) > 4:
            use = len(data_files) // 2
        else:
            use = len(data_files) - 1
        use = min(use, len(valid_sim_files))
        best_indices = np.argsort(valid_fit_params[:,3])[:use]
        added_data = np.zeros(data.shape)
        added_data = np.zeros(data.shape)
        for idx in best_indices:
            with fits.open(os.path.join(os.path.abspath("out"),valid_sim_files[idx]+self.color+".fits")) as sfile:
                data_ = sfile[0].data

                shifted_scaled = self.shift_and_scale(
                    data_, 
                    valid_fit_params[idx, 0],
                    valid_fit_params[idx, 1],
                    valid_fit_params[idx, 2]
                )
                added_data += data - shifted_scaled
    
        gauss_kernal = Gaussian2DKernel(x_stddev=1, y_stddev=1)
        cmray_conv = convolve(added_data, gauss_kernal)
        cmray_conv_mask = (cmray_conv > (np.median(cmray_conv.flatten()) + 3*np.std(cmray_conv.flatten())))
        # fits.writeto(os.path.join(os.path.abspath("out"), self.datafilename+self.color+"_cmray_mask.fits"), 
        #             data=1.*cmray_conv_mask, overwrite=True)

    async def cmray_mask_minimize_agent(self, reference_file, target_file, 
                                  area=300, 
                                  x0=np.array([0, 0, 1]), 
                                  bounds=[(-2, 2), (-2, 2), (0.5, 1.5)]) -> None:
        import os
        from astropy.io import fits
        import numpy as np
        from scipy.ndimage import convolve
        from astropy.convolution import Gaussian2DKernel
        import glob

        margin = int(np.max([np.abs(bounds[0]),np.abs(bounds[1])])+1)

        with fits.open(os.path.join(os.path.abspath("out"),reference_file+self.color+".fits")) as refdata:
            data = refdata[0].data
            std_cutoff = np.std(np.percentile(data.flatten(), [1, 50]))

            random_x = np.random.randint(0,data.shape[1]-area-2*margin)
            random_y = np.random.randint(0,data.shape[0]-area-2*margin)

            cut_data = data[
                random_y+margin:random_y+margin+area,
                random_x+margin:random_x+margin+area
            ]

        with fits.open(os.path.join(os.path.abspath("out"),target_file+self.color+".fits")) as sfile:
            data_ = sfile[0].data

            cut_data_ = data_[
                random_y+margin:random_y+margin+area,
                random_x+margin:random_x+margin+area
            ]

            cutoff = 2*np.nanstd(cut_data)

            best_v, best_h, best_s, best_std = self.optimize_cmray_params(
                cut_data,
                cut_data_,
                margin,
                cutoff,
                x0=np.array([0, 0, 1]),
                bounds=[(-2, 2), (-2, 2), (0.5, 1.5)]
            )

        return best_v, best_h, best_s, best_std, std_cutoff

    async def cmray_mask_minimize_agent(self, reference_file, target_file, 
                                area=300, 
                                x0=np.array([0, 0, 1]), 
                                bounds=[(-2, 2), (-2, 2), (0.5, 1.5)]):
        """
        Async version of minimize that can be called from the agent
        Returns best parameters and quality metrics
        """
        # This allows us to call the synchronous method from async context
        import functools
        import concurrent.futures
        
        # Create a thread pool executor for running the synchronous method
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            # Run the synchronous method in a thread and await its result
            result = await loop.run_in_executor(
                pool,
                functools.partial(
                    self._cmray_mask_minimize_agent_sync,
                    reference_file,
                    target_file,
                    area,
                    x0,
                    bounds
                )
            )
        
        return result

    # Non-async implementation to be called from the async wrapper
    def _cmray_mask_minimize_agent_sync(self, reference_file, target_file, 
                                    area=300, 
                                    x0=np.array([0, 0, 1]), 
                                    bounds=[(-2, 2), (-2, 2), (0.5, 1.5)]):
        import os
        from astropy.io import fits
        import numpy as np
        
        margin = int(np.max([np.abs(bounds[0][1]), np.abs(bounds[1][1])])+1)

        with fits.open(os.path.join(os.path.abspath("out"), reference_file+self.color+".fits")) as refdata:
            data = refdata[0].data
            std_cutoff = np.std(np.percentile(data.flatten(), [1, 60]))

            # make random_x and random_y in the center 50% of the image
            random_x = np.random.randint(data.shape[1]*0.25, data.shape[1]*0.75-area-2*margin)
            random_y = np.random.randint(data.shape[0]*0.25, data.shape[0]*0.75-area-2*margin)

            cut_data = data[
                random_y+margin:random_y+margin+area,
                random_x+margin:random_x+margin+area
            ]

        with fits.open(os.path.join(os.path.abspath("out"), target_file+self.color+".fits")) as sfile:
            data_ = sfile[0].data

            cut_data_ = data_[
                random_y+margin:random_y+margin+area,
                random_x+margin:random_x+margin+area
            ]

            cutoff = 2*np.nanstd(cut_data)

            best_v, best_h, best_s, best_std = self.optimize_cmray_params(
                cut_data,
                cut_data_,
                margin,
                cutoff,
                x0=x0,
                bounds=bounds
            )

        return best_v, best_h, best_s, best_std, std_cutoff
    



    async def generate_cmray_mask_from_agent(self, best_fits: dict, full_stds: dict, area=300) -> None:
        import os
        from astropy.io import fits
        import numpy as np
        from scipy.ndimage import convolve
        from astropy.convolution import Gaussian2DKernel
        
        with fits.open(os.path.join(os.path.abspath("out"), self.datafilename+self.color+".fits")) as refdata:
            data = refdata[0].data
        
        if len(best_fits) > 4:
            use = len(best_fits) // 2
        else:
            use = max(1, len(best_fits) - 1)
        
        # sort by best standard deviation
        # print(best_fits, flush=True)
        # print(full_stds, flush=True)        

        sorted_fits = sorted(best_fits.items(), key=lambda item: full_stds[item[0]], reverse=False)
        # print(sorted_fits,flush=True)

        # best fits only!
        added_data = np.zeros(data.shape)
        for target_file, params in sorted_fits[:use]:
            with fits.open(os.path.join(os.path.abspath("out"), target_file+self.color+".fits")) as sfile:
                data_ = sfile[0].data
        
                v_shift, h_shift, scale = params
                shifted_scaled = self.shift_and_scale(data_, v_shift, h_shift, scale)
                added_data += data - shifted_scaled
        
        # create mask
        gauss_kernal = Gaussian2DKernel(x_stddev=1, y_stddev=1)
        cmray_conv = convolve(added_data, gauss_kernal)
        cmray_conv_mask = (cmray_conv > (np.median(cmray_conv.flatten()) + 3*np.std(cmray_conv.flatten())))
        
        # save mask
        fits.writeto(os.path.join(os.path.abspath("out"), self.datafilename+self.color+"_cmray_mask.fits"),
                     data=1.*cmray_conv_mask, overwrite=True)

        # write best fits to a file
        import json
        best_fits_file = os.path.join(os.path.abspath("out"), self.datafilename + self.color + "_best_fits.json")
        with open(best_fits_file, 'w') as f:
            json.dump(best_fits, f, indent=4)

        try:
            # also save the plot of best fits
            fig,ax = plt.subplots(figsize=(10,5))
            fig.set_facecolor("linen")

            vs = {key: value[0] for key, value in best_fits.items()}
            hs = {key: value[1] for key, value in best_fits.items()}
            scales = {key: value[2] for key, value in best_fits.items()}
            ax.plot(best_fits.keys(), list(vs.values()), label="Vertical Shift", marker='o', color='sienna')
            ax.plot(best_fits.keys(), list(hs.values()), label="Horizontal Shift", marker='o', color='peru')
            ax.set_xlabel("Target File")
            ax.set_ylabel("shift (px)")

            ax2 = ax.twinx()
            ax2.set_ylabel("spectrograph temperature (°C)")

            if self.color == "r":
                temps = [17.812, 17.688, 17.5, 17.375, 17.312, 17.25, 17.062, 17.]
            else:
                temps = [17.812, 17.625, 17.5, 17.312, 17.25, 17.125, 17., 16.938]
            temps = [temp for temp, key in zip(temps, best_fits.keys()) if key != self.datafilename]
            ax2.plot(best_fits.keys(), temps, label="Echelle Temperature", marker='o', color='orchid')

            # ax2.plot(best_fits.keys(), list(scales.values()), label="Scale (s)", marker='o', color='orchid')

            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(handles + handles2, labels + labels2, loc="lower center")

            plt.title(f"Reference File {self.datafilename}{self.color} Best Fits",weight="bold",color="maroon")
            plt.savefig(os.path.join(os.path.abspath("out"), self.datafilename + self.color + "_best_fits.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error generating best fits plot for {self.datafilename}{self.color}: {e}")

        # print(self.datafilename,self.color,"generation context saved")
        # print("DO THIS LATER: generate mask correctly")
        
        return None





class CosmicRayTrackingAgent(Agent):
    def __init__(
            self,
    ) -> None:
        super().__init__()
        self.param_hist = {}
        self.best_fits_ref = {}
        self.stitch_instance = None
        self.optimization_results = {}

        # self._current_opt_params = {} # don't think i need rn

    @action
    async def set_stitch_instance(self, stitch_instance: Stitch) -> None:
        self.stitch_instance = stitch_instance

    @action
    async def initial_guess(self, reference_file: str, target_file: str, file_list: str) -> List[float]:
        '''
        cases:
        1 - no reference file, no target file
         start with v,h,s = 0,0,1
         optimize
        2 - reference file, target file
         if ref,target than same v,h,s
         if target,ref then -v,-h,1/s
        3 - reference file, no target file
         if previous target, use v,h,s from previous
         optimize
         if no, use -v,-h,1/s from previous inverted
         optimize
        4 - no reference file, target file (caught by 1)
        '''

        # include self.log.info("...")

        # print()
        # print(self.stitch_instance.color, "initial_guess", reference_file, target_file, file_list)

        optimize = False

        # case 2
        if reference_file in self.best_fits_ref and target_file in self.best_fits_ref[reference_file]:
            # print("case 2.1: file pair already exists (THIS SHOULD NOT HAPPEN)", "[v_shift, h_shift, scale, DO NOT optimize]")
            v_shift, h_shift, scale = self.best_fits_ref[reference_file][target_file]
            return [v_shift, h_shift, scale, optimize]
        if target_file in self.best_fits_ref and reference_file in self.best_fits_ref[target_file]:
            # print("case 2.2: inverse file pair already exists", "[-v_shift, -h_shift, 1/scale, DO NOT optimize]")
            v_shift, h_shift, scale = self.best_fits_ref[target_file][reference_file]
            return [-v_shift, -h_shift, 1/scale, optimize]
        
        # case 3
        if reference_file in self.best_fits_ref:
            # if reference file exists, then previous target pair should too!
            prev_target_idx = file_list.index(target_file) - 1
            if prev_target_idx >= 0:
                prev_target_file = file_list[prev_target_idx]
                if prev_target_file in self.best_fits_ref[reference_file]:
                    # print("case 3.1: using guess from",prev_target_file, "[v_shift, h_shift, scale, optimize]")
                    optimize = True
                    v_shift, h_shift, scale = self.best_fits_ref[reference_file][prev_target_file]
                    return [v_shift, h_shift, scale, optimize]
                
            # if reference file exists, but previous target pair is not consistent
            #  this case occurs when target jumps across reference file
            if prev_target_file == reference_file:
                prev_target_idx = prev_target_idx - 1
                if prev_target_idx < len(file_list) and prev_target_idx >= 0:
                    prev_target_file = file_list[prev_target_idx]
                    if prev_target_file in self.best_fits_ref[reference_file]:
                        # print("case 3.2: inverting guess from",prev_target_file, "[-v_shift, -h_shift, 1/scale, optimize]")
                        optimize = True
                        v_shift, h_shift, scale = self.best_fits_ref[reference_file][prev_target_file]
                        return [-v_shift, -h_shift, 1/scale, optimize]


        if reference_file not in self.best_fits_ref:
            # print("case 1: no reference file pair yet", "[0,0,1,optimize]")
            optimize = True
            return [0,0,1,optimize]

        # case 4 bad
        print("NOT CAUGHT???", reference_file, target_file, file_list)
        print(self.best_fits_ref)

    @action
    async def optimize_parameters(self, reference_file: str, target_file: str,
                                v_shift: float, h_shift: float, scale: float,
                                max_iter: int = 20) -> Dict:
        """
        Run optimization until std_value is lower than std_cutoff
        """
        # Initialize the best parameters
        best_params = {
            "v_shift": v_shift,
            "h_shift": h_shift,
            "scale": scale,
            "std_value": float('inf')
        }

        for iter in range(max_iter):
            # print(self.stitch_instance.color, iter, best_params)

            # make bounds smaller based on previous best
            #  DOUBLE CHECK THIS PARAMETER SPACE MAKES SENSE
            # bounds = [
            #     (best_params["v_shift"] - 0.5, best_params["v_shift"] + 0.5),
            #     (best_params["h_shift"] - 0.5, best_params["h_shift"] + 0.5),
            #     (best_params["scale"] - 0.1, best_params["scale"] + 0.1)
            # ]
            bounds = [
                (-2, 2),
                (-2, 2),
                (0.5, 1.5)
            ]

            # Call the stitch method
            best_v, best_h, best_s, std_value, std_cutoff = await self.stitch_instance.cmray_mask_minimize_agent(
                reference_file,
                target_file,
                area=300,
                x0=np.array([best_params["v_shift"], best_params["h_shift"], best_params["scale"]]),
                bounds=bounds
            )

            # print(std_value, std_cutoff)

            # Store result
            result_key = f"{reference_file}_{target_file}_{iter}"
            self.optimization_results[result_key] = {
                "v_shift": best_v,
                "h_shift": best_h,
                "scale": best_s,
                "std_value": std_value,
                "std_cutoff": std_cutoff
            }

            # Update best parameters if this result is better
            if std_value < best_params["std_value"]:
                best_params["v_shift"] = best_v
                best_params["h_shift"] = best_h
                best_params["scale"] = best_s
                best_params["std_value"] = std_value
            
            # Success condition: std is lower than cutoff
            if std_value < std_cutoff:
                


                # fig,ax = plt.subplots(figsize=(8,5))
                # cmap = plt.cm.get_cmap("PuOr").copy()
                # cmap.set_bad('red', 1.)
                # ref_data = fits.getdata(os.path.join(os.path.abspath("out"), reference_file+self.stitch_instance.color+".fits"))
                # target_data = fits.getdata(os.path.join(os.path.abspath("out"), target_file+self.stitch_instance.color+".fits"))
                # result = self.stitch_instance.subtract_ims(
                #     best_v,
                #     best_h,
                #     best_s,
                #     ref_data,
                #     target_data,
                #     margin = int(np.max([np.abs(bounds[0]),np.abs(bounds[1])])+1),
                #     cutoff = 2*np.nanstd(ref_data)
                # )
                # vmin,vmax = -5*np.nanstd(result), 5*np.nanstd(result)
                # ax.set_title(f"{reference_file}-{target_file} std: {std_value:.2f}")
                # # fig.colorbar(ax=ax[0, idx], mappable=ax[0, idx].images[0],fraction=0.046, pad=0.04)
                # ax.imshow(result,cmap=cmap,vmin=vmin,vmax=vmax)
                # ax.set_xticks([])
                # ax.set_yticks([])
                # # fig.colorbar(ax=ax[1, idx], mappable=ax[1, idx].images[0],fraction=0.046, pad=0.04)
                # plt.show()





                # print("success!", best_params)


                if reference_file not in self.best_fits_ref:
                    self.best_fits_ref[reference_file] = {}
                
                self.best_fits_ref[reference_file][target_file] = [best_v, best_h, best_s]
                
                await self.update_params(
                    reference_file, target_file, best_v, best_h, best_s, std_value
                )
                
                return {"success": True, "params": best_params, "iterations": iter+1}
        
        # No optimal solution found
        if reference_file not in self.best_fits_ref:
            self.best_fits_ref[reference_file] = {}
        
        self.best_fits_ref[reference_file][target_file] = [
            best_params["v_shift"], 
            best_params["h_shift"], 
            best_params["scale"]
        ]
        
        await self.update_params(
            reference_file, 
            target_file, 
            best_params["v_shift"],
            best_params["h_shift"],
            best_params["scale"],
            best_params["std_value"]
        )
        
        return {"success": False, "params": best_params, "iterations": max_iter}

    @action
    async def update_best_fits(self, reference_file: str, target_file: str,
                              v_shift: float, h_shift: float, scale: float,
                              optimize: bool) -> None:
        if reference_file not in self.best_fits_ref:
            self.best_fits_ref[reference_file] = {}
            
        if target_file not in self.best_fits_ref[reference_file]:
            if optimize:
                result = await self.optimize_parameters(
                    reference_file, target_file, v_shift, h_shift, scale
                )
                self.best_fits_ref[reference_file][target_file] = [
                    result["params"]["v_shift"],
                    result["params"]["h_shift"],
                    result["params"]["scale"]
                ]
            else:
                self.best_fits_ref[reference_file][target_file] = [v_shift, h_shift, scale]

                

    # Add this method to get the best fits for a reference file
    @action
    async def get_best_fits_for_reference(self, reference_file: str) -> dict:
        """Get all best fits for a specific reference file"""
        if reference_file not in self.best_fits_ref:
            return {}
        return self.best_fits_ref[reference_file]

    @action
    async def update_params(self, reference_file: str, target_file: str,
                            v_shift: float, h_shift: float, scale: float,
                            std_value: float) -> None:
        if reference_file not in self.param_hist:
            self.param_hist[reference_file] = {}
        
        if target_file not in self.param_hist[reference_file]:
            self.param_hist[reference_file][target_file] = []
        
        self.param_hist[reference_file][target_file].append({
            "v_shift": v_shift,
            "h_shift": h_shift,
            "scale": scale,
            "std_value": std_value
        })
    
    @action
    async def get_params(self):
        return self.param_hist


























@python_app
def load_and_save_app(stitch_args, bin_to_2x1=True):
    stitch = Stitch(**stitch_args)
    return stitch.load_and_save(bin_to_2x1)

@python_app
def combined_bias_app(dep_futures = [], stitch_args = None, files = None):
    import os
    from astropy.io import fits
    import numpy as np

    stitch_args_copy = dict(stitch_args)
    files_copy = list(files)

    [dep.result() for dep in dep_futures]

    stitch = Stitch(**stitch_args_copy)

    internal_bias = stitch.bias_sub()

    fits.writeto(
        filename=os.path.join(
            os.path.abspath("out"),
            stitch_args_copy["flatfilename"]+"_biasfilter_"+stitch_args_copy["color"]+".fits"
            ),
        data=internal_bias,
        overwrite=True
    )

    stitch.write_noise(internal_bias, files_copy)

    return None

# @python_app
# def bias_sub_app(dep_futures,stitch_args):
#     # import concurrent.futures
#     # concurrent.futures.wait(dep_futures,return_when="ALL_COMPLETED")
#     [f.result() for f in dep_futures]

#     stitch = Stitch(**stitch_args)

#     return stitch.bias_sub()

# @python_app
# def noise_app(dep_futures,stitch_args,internal_noise):
#     [f.result() for f in dep_futures]

#     # noise = internal_noise.result()
#     noise = internal_noise

#     stitch = Stitch(**stitch_args)

#     return stitch.write_noise(noise)

@python_app
def cmray_mask_app(dep_futures, stitch_args, data_files) -> None:
    [f.result() for f in dep_futures]

    stitch = Stitch(**stitch_args)
    return stitch.cmray_mask(data_files)

def cmray_mask_new_app(stitch_args, data_files) -> None:
    stitch = Stitch(**stitch_args)
    return stitch.cmray_mask_new(data_files)

def cmray_mask_minimize_app(stitch_args, data_files, area=150, x0=np.array([0, 0, 1]), bounds=[(-2, 2), (-2, 2), (0.5, 1.5)]) -> None:
    stitch = Stitch(**stitch_args)
    return stitch.cmray_mask_minimize(data_files, area, x0, bounds)

@join_app
def cmray_mask_agent_join_app(dep_futures=None, stitch_args=None, data_files=None, area=300):
    import asyncio
    from parsl.data_provider.files import File
    from academy.manager import Manager
    from academy.exchange import LocalExchangeFactory
    from concurrent.futures import ThreadPoolExecutor
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    
    if dep_futures:
        [f.result() for f in dep_futures]


    @python_app
    def create_file_fut(filep):
        return File(filep)
    
    async def process_all_files():
        parsl_futures = []

        async with await Manager.from_exchange_factory(
            factory=LocalExchangeFactory(),
            executors=ThreadPoolExecutor(),
        ) as manager:
            cr_agent = await manager.launch(CosmicRayTrackingAgent())
            
            color = stitch_args["color"]
            # print(f"Processing all {color} files with a single agent")
            
            for reference_file in data_files:
                current_stitch_args = stitch_args.copy()
                current_stitch_args["datafilename"] = reference_file
                stitch = Stitch(**current_stitch_args)
                
                await cr_agent.set_stitch_instance(stitch)
                
                for target_file in [f for f in data_files if f != reference_file]:
                    v_shift, h_shift, scale, optimize = await cr_agent.initial_guess(
                        reference_file, target_file, data_files
                    )
                    
                    await cr_agent.update_best_fits(
                        reference_file, target_file, v_shift, h_shift, scale, optimize
                    )
                
                best_fits = await cr_agent.get_best_fits_for_reference(reference_file)

                full_stds = {}
                for target_file in [f for f in data_files if f != reference_file]:
                    v_shift, h_shift, scale = best_fits[target_file]

                    ref_data = fits.getdata(os.path.join(os.path.abspath("out"), reference_file + color + ".fits"))
                    target_data = fits.getdata(os.path.join(os.path.abspath("out"), target_file + color + ".fits"))

                    margin = int(np.max([np.abs(v_shift), np.abs(h_shift)]) + 1)
                    cutoff = 2 * np.nanstd(ref_data)
                    _,_,_,full_std = stitch.optimize_cmray_params(ref_data, target_data, margin, cutoff, x0=best_fits[target_file], optimize=False)
                    full_stds[target_file] = full_std

                # best_params = await cr_agent.get_params()
                # v_shift, h_shift, scale = best_fits.get(target_file, [0, 0, 1])
                # print(f"Best fits for {reference_file}:")
                # print(best_fits)
                # print(v_shift,h_shift,scale)
                # print(best_params)
                
                await stitch.generate_cmray_mask_from_agent(best_fits, full_stds, area=area)

                output_mask_path = os.path.join(os.path.abspath("out"), reference_file + color + "_cmray_mask.fits")
                file_fut = create_file_fut(output_mask_path)
                parsl_futures.append(file_fut)

            # shutdown???
            await cr_agent.shutdown()
        
        return parsl_futures
    
    results = asyncio.run(process_all_files())
    
    # Run the async function
    return results