import numpy as np
from astropy.io import fits
from scipy.ndimage import convolve
from scipy import ndimage
from astropy.convolution import Gaussian2DKernel

from parsl import python_app

from ifum_agent.agents.alignment import AlignmentParameters

def shift_and_scale(image, v_shift, h_shift, scale=1.0, order=1):
    # order = 1 -> bilinear interpolation

    shifted_image = ndimage.shift(
        image,
        (v_shift,h_shift),
        order=order,
        mode='constant',
        cval=0.0
    )
    
    return shifted_image * scale

def subtract_ims(v_shift, h_shift, scale, refdata, data, margin, cutoff):
    shifted_data = shift_and_scale(data, v_shift, h_shift, scale)
    
    subtracted = refdata[margin:-margin, margin:-margin] - shifted_data[margin:-margin, margin:-margin]
    # turn outliers to NaN (top&bottom cutoff_perc%)
    extreme_mask = (subtracted < (-1*cutoff)) | (subtracted > cutoff)
    subtracted[extreme_mask] = np.nan

    return subtracted

def image_diff_deviation(cut_data, cut_data_, margin, cutoff, params):
    v_shift, h_shift, scale = params
    result = subtract_ims(v_shift, h_shift, scale, cut_data, cut_data_, margin, cutoff)
    return np.nanstd(result)   


@python_app
def generate_mask(
    ref_file,
    alignment_params: dict[str, AlignmentParameters],
    outputs=(),
):
    ref_data = fits.getdata(ref_file)
    full_stds = {}
    for target_file, alignment_param in alignment_params.items():
        target_data = fits.getdata(target_file)
        margin = int(np.max([np.abs(alignment_param.v_shift), np.abs(alignment_param.h_shift)]) + 1)
        cutoff = 2 * np.nanstd(ref_data)
        x0 = np.array([alignment_param.v_shift, alignment_param.h_shift, alignment_param.scale])
        full_std = image_diff_deviation(ref_data, target_data, margin, cutoff, params=x0)
        full_stds[target_file] = full_std
    
    if len(alignment_params) > 4:
        use = len(alignment_params) // 2
    else:
        use = max(1, len(alignment_params)-1)       

    sorted_fits = sorted(alignment_params.items(), key=lambda item: full_stds[item[0]], reverse=False)

    # best fits only!
    added_data = np.zeros(ref_data.shape)
    for target_file, params in sorted_fits[:use]:
        with fits.open(target_file) as sfile:
            data_ = sfile[0].data
        shifted_scaled = shift_and_scale(data_, params.v_shift, params.h_shift, params.scale)
        added_data += ref_data - shifted_scaled
    
    # create mask
    gauss_kernal = Gaussian2DKernel(x_stddev=1, y_stddev=1)
    cmray_conv = convolve(added_data, gauss_kernal)
    cmray_conv_mask = (cmray_conv > (np.median(cmray_conv.flatten()) + 3*np.std(cmray_conv.flatten())))
    
    # save mask
    fits.writeto(outputs[0], data=1.*cmray_conv_mask, overwrite=True)