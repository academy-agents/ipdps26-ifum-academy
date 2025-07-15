import numpy as np
import glob
import os
from astropy.io import fits
from scipy import ndimage
from astropy.convolution import Gaussian2DKernel, convolve
import re
from parsl.app.app import python_app
from parsl.data_provider.files import File
from .utils import *

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
        return None

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

    def write_noise(self, internal_noise) -> None:
        data_file = os.path.join(os.path.abspath("out"),self.filename+"_withbias_"+self.color+".fits")
        
        with fits.open(data_file) as datahdu:
            data = datahdu[0].data

        denoised = data/internal_noise
        fits.writeto(filename=os.path.join(os.path.abspath("out"),self.filename+self.color+".fits"), data=denoised, overwrite=True)

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
        import glob
        from astropy.io import fits
        import numpy as np
        from astropy.convolution import Gaussian2DKernel, convolve


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

    

@python_app
def load_and_save_app(stitch_args, bin_to_2x1=True):
    stitch = Stitch(**stitch_args)
    return stitch.load_and_save(bin_to_2x1)

@python_app
def bias_sub_app(stitch_args):
    stitch = Stitch(**stitch_args)
    return stitch.bias_sub()

@python_app
def noise_app(stitch_args,internal_noise):
    stitch = Stitch(**stitch_args)
    return stitch.write_noise(internal_noise)

@python_app
def cmray_mask_app(stitch_args, data_files) -> None:
    stitch = Stitch(**stitch_args)
    return stitch.cmray_mask(data_files)
