import numpy as np
import glob
import os
from astropy.io import fits
from scipy import ndimage
from astropy.convolution import Gaussian2DKernel, convolve
import re

class Stitch():
    '''
    Class to stitch data, arc, and flat files together

    Attributes:
        

    Methods:
        load_files: 
        save_file: includes bias subtraction, gain subtraction
    '''
    def __init__(self, directory: str, filename: str, files, color: str,
                 datafilename: str, arcfilename: str, flatfilename: str):
        self.directory = directory
        self.filename = filename
        self.files = files
        self.color = color
        self.datafilename = datafilename
        self.arcfilename = arcfilename
        self.flatfilename = flatfilename

    def load_files(self) -> None:
        # find the appropriate files
        files = glob.glob(os.path.join(self.directory,("*"+self.filename+"*.fits")))

        # creates convenient matrix of the files
        ordered_identity = np.array([["blue1","blue2","blue3","blue4"],
                                 ["red1","red2","red3","red4"]])
        ordered_files = np.empty((2, 4), dtype="object")

        # assigns files correctly within matrix
        for file in files:
            header,data = fits.open(file)[0].header,fits.open(file)[0].data
            self.color,opamp = header["SHOE"],header["OPAMP"]
            i = 0 if self.color=="B" else 1 if self.color=="R" else None
            j = 0 if opamp==1 else 1 if opamp==2 else 2 if opamp==3 else 3 if opamp==4 else None
            ordered_files[i,j] = file

        if ordered_identity[ordered_files==None].size != 0:
            print(f"{len(files)} files")
            print(f"missing files: {ordered_identity[ordered_files==None]}")
            return
        else:
            self.files = ordered_files

    def save_file(self,bin_to_2x1=True) -> None:
        # assumes constant trim section
        # x1,x2 = 0,1024
        # y1,y2 = 0,2056
        
        ordered_data = np.empty((2,4), dtype="object")
        for iy, ix in np.ndindex(self.files.shape):
            file = self.files[iy,ix]
            header,data = fits.open(file)[0].header,fits.open(file)[0].data
            x1,x2,y1,y2 = [int(s) for s in re.findall(r'\d+', header["TRIMSEC"])]
            x1 -= 1
            y1 -= 1
            # accounts for gain
            gain = header["EGAIN"]
            # ordered_data[iy,ix] = data[y1:y2,x1:x2]/gain
            # subtracts the mean of bias x slices from the data
            ordered_data[iy,ix] = data[y1:y2,x1:x2] - np.repeat(np.array([np.mean(data[y1:y2,x2:],axis=1)]).T,data[y1:y2,x1:x2].shape[1],axis=1)
            if bin_to_2x1 and header["BINNING"]=='1x1':
                ordered_data[iy,ix] = ordered_data[iy,ix][:,0::2]+ordered_data[iy,ix][:,1::2]

        # stack images
        total_b = np.vstack((np.hstack((ordered_data[0][3],np.flip(ordered_data[0][2], axis=1))),
                             np.hstack((np.flip(ordered_data[0][0], axis=0),np.flip(ordered_data[0][1], axis=(0,1))))))
        total_r = np.vstack((np.hstack((ordered_data[1][3],np.flip(ordered_data[1][2], axis=1))),
                             np.hstack((np.flip(ordered_data[1][0], axis=0),np.flip(ordered_data[1][1], axis=(0,1))))))

        # save images as fits files
        fits.writeto(os.path.join(os.path.relpath("out"),self.filename+"_withbias_b.fits"), data=total_b, overwrite=True)
        fits.writeto(os.path.join(os.path.relpath("out"),self.filename+"_withbias_r.fits"), data=total_r, overwrite=True)

    def bias_sub(self) -> None:
        data_file = os.path.join(os.path.relpath("out"),self.datafilename+"_withbias_"+self.color+".fits")
        arc_file = os.path.join(os.path.relpath("out"),self.arcfilename+"_withbias_"+self.color+".fits")
        flat_file = os.path.join(os.path.relpath("out"),self.flatfilename+"_withbias_"+self.color+".fits")
        
        data = fits.open(data_file)[0].data
        arc_data = fits.open(arc_file)[0].data
        flat_data = fits.open(flat_file)[0].data

        median_image = ndimage.median_filter(flat_data,size=(1,9))
        internal_noise = flat_data/median_image
        # THINK OF BETTER WAY TO STANDARDIZE??
        internal_noise = (internal_noise)/(np.percentile(internal_noise,99)-np.min(internal_noise))

        # save bias filter along with 
        fits.writeto(os.path.join(os.path.relpath("out"),self.datafilename+self.color+".fits"), data=data/internal_noise, overwrite=True)
        fits.writeto(os.path.join(os.path.relpath("out"),self.arcfilename+self.color+".fits"), data=arc_data/internal_noise, overwrite=True)
        fits.writeto(os.path.join(os.path.relpath("out"),self.flatfilename+self.color+".fits"), data=flat_data/internal_noise, overwrite=True)
        fits.writeto(os.path.join(os.path.relpath("out"),self.flatfilename+"_biasfilter_"+self.color+".fits"), data=internal_noise, overwrite=True)

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
        
        data = fits.open(os.path.join(os.path.relpath("out"),self.datafilename+self.color+".fits"))[0].data
        
        data_m = np.ones(data.shape)
        fit_params = np.zeros((len(sim_files),4))
        not_sim = []
        for idx,s_file in enumerate(sim_files):
            if len(glob.glob(os.path.join(os.path.relpath("out"),(s_file+"*.fits"))))<2:
                not_sim.append(idx)
            else:
                data_ = fits.open(os.path.join(os.path.relpath("out"),s_file+self.color+".fits"))[0].data
            
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
            data_ = fits.open(os.path.join(os.path.relpath("out"),sim_files[idx]+self.color+".fits"))[0].data
            data_large = self.bl_resize(data_,data_.shape[0]*mult,data_.shape[1]*mult)
            shifted_ = self.perform_shift(int(fit_params[idx,0]),int(fit_params[idx,1]),fit_params[idx,2],data_large)
            shifted_ = self.bl_resize(shifted_,data_.shape[0],data_.shape[1])
            added_data += data-shifted_

        gauss_kernal = Gaussian2DKernel(x_stddev=1,y_stddev=1)
        cmray_conv = convolve(added_data,gauss_kernal)
        cmray_conv_mask = (cmray_conv>(np.median(cmray_conv.flatten())+3*np.std(cmray_conv.flatten())))
        fits.writeto(os.path.join(os.path.relpath("out"),self.datafilename+self.color+"_cmray_mask.fits"), data=1.*cmray_conv_mask, overwrite=True)